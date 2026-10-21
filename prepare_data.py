"""Prepare one CSV of original offshore wave measurements and aligned wind estimates."""
import argparse
from pathlib import Path

import copernicusmarine as cm
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent
WAVE_ID = 'cmems_mod_glo_wav_my_0.2deg_PT3H-i'
WIND_ID = 'cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H'
WIND_VARIABLES = ['eastward_wind', 'northward_wind']
LATITUDE, LONGITUDE = 13.4, 74.6


def check_times(times, start, end, frequency):
    index = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).as_unit('ns')
    expected = pd.date_range(start, end, freq=frequency, tz='UTC').as_unit('ns')
    if index.has_duplicates or not index.equals(expected):
        raise ValueError(f'Invalid {frequency} timestamps: duplicates={index.has_duplicates}; '
                         f'missing={list(map(str, expected.difference(index)))}; '
                         f'unexpected={list(map(str, index.difference(expected)))}')
    return index


def catalogue(dataset_id, start, end):
    """Inspect real variable names, version and advertised coverage before downloading."""
    result = cm.describe(dataset_id=dataset_id, disable_progress_bar=True).model_dump(mode='json')
    dataset = next(d for p in result['products'] for d in p['datasets'] if d['dataset_id'] == dataset_id)
    options = [(v['label'], s) for v in dataset['versions'] for part in v['parts']
               for s in part['services'] if s['service_short_name'] == 'timeseries']
    version, service = max(options, key=lambda item: item[0])
    available = {v['short_name']: v for v in service['variables']}
    variables = ([v for v in available if v.startswith(('VHM','VMDR','VTM','VTP'))]
                 if dataset_id == WAVE_ID else WIND_VARIABLES)
    required = ['VHM0'] if dataset_id == WAVE_ID else WIND_VARIABLES
    if not set(required).issubset(available):
        raise ValueError(f'Required source variables missing from {dataset_id}')
    for name in variables:
        time = next(c for c in available[name]['coordinates'] if c['coordinate_id'] == 'time')
        lo, hi = [pd.to_datetime(time[key], unit='ms') for key in ('minimum_value','maximum_value')]
        if start < lo or end > hi:
            raise ValueError(f'{name}: requested dates outside catalogue coverage {lo} to {hi}')
    print(f'{dataset_id}: version {version}, variables: {", ".join(variables)}')
    return version, variables


def validate_grid(ds, start, end, kind, latitude, longitude, variables):
    check_times(ds.time.values, start, end, '3h' if kind == 'wave' else '1h')
    if not set(variables).issubset(ds.data_vars):
        raise ValueError('Cached file lacks required variables')
    for coord, target in [('latitude',latitude), ('longitude',longitude)]:
        values = ds[coord].values
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or len(np.unique(values)) != len(values):
            raise ValueError(f'Invalid {coord} grid')
        if not values.min() <= target <= values.max():
            raise ValueError(f'Cached grid does not cover reference {coord}')
    for name in variables:
        if set(ds[name].dims) != {'time','latitude','longitude'}:
            raise ValueError(f'Unexpected dimensions: {name}')


def load_chunk(kind, start, end, latitude, longitude, metadata=None, download=False):
    """Reuse matching NetCDF data; new chunks need explicit download permission."""
    folder = ROOT/'data/raw'/('waves' if kind == 'wave' else 'wind')
    folder.mkdir(parents=True, exist_ok=True)
    version, variables = metadata if metadata else (None, ['VHM0'] if kind == 'wave' else WIND_VARIABLES)
    # Legacy sample names are deliberately supported without their old JSON manifests.
    pattern = f'{kind}_{version}_*.nc' if version else '*.nc'
    for path in sorted(folder.glob(pattern)):
        if '.partial' in path.name:
            continue
        try:
            with xr.open_dataset(path) as source:
                # Other cached years are not candidates for this request.
                if not source.time.size or pd.Timestamp(source.time.values[0]) > start or pd.Timestamp(source.time.values[-1]) < end:
                    continue
                ds = source.sel(time=slice(start, end)).load()
            validate_grid(ds, start, end, kind, latitude, longitude, variables)
        except (OSError, ValueError, KeyError) as error:
            print(f'Cache not suitable: {path.name}: {error}')
            continue
        print(f'Using cache: {path.name}')
        return ds
    if not download:
        raise FileNotFoundError(f'No suitable cached {kind} data for {start} to {end}; downloads are disabled')
    path = folder/f'{kind}_{version}_{start:%Y%m%dT%H}_{end:%Y%m%dT%H}.nc'
    temporary = path.with_name(path.stem+'.partial.nc')
    radius = .21 if kind == 'wave' else .14
    cm.subset(dataset_id=WAVE_ID if kind == 'wave' else WIND_ID, dataset_version=version,
              variables=variables, start_datetime=start.isoformat(), end_datetime=end.isoformat(),
              minimum_latitude=latitude-radius, maximum_latitude=latitude+radius,
              minimum_longitude=longitude-radius, maximum_longitude=longitude+radius,
              coordinates_selection_method='inside', output_directory=folder,
              output_filename=temporary.name, overwrite=True, disable_progress_bar=True)
    with xr.open_dataset(temporary) as source:
        ds = source.load()
    validate_grid(ds, start, end, kind, latitude, longitude, variables)
    temporary.replace(path)
    return ds


def select_wave_cell(ds, latitude=LATITUDE, longitude=LONGITUDE):
    """Retain the verified nearest valid source-cell selection."""
    choices = []
    for lat in ds.latitude.values:
        for lon in ds.longitude.values:
            cell = ds.sel(latitude=lat, longitude=lon)
            if (np.isfinite(cell.VHM0.values) & (cell.VHM0.values >= 0)).any():
                distance = (float(lat)-latitude)**2 + ((float(lon)-longitude)*np.cos(np.deg2rad(latitude)))**2
                choices.append((distance, float(lat), float(lon)))
    if not choices:
        raise ValueError('No valid offshore wave cell in source neighborhood')
    _, lat, lon = min(choices)
    return ds.sel(latitude=lat, longitude=lon), lat, lon


def interpolate_wind(ds, latitude, longitude):
    """Bilinearly interpolate components first, then calculate speed."""
    ds = ds.sortby('latitude').sortby('longitude')
    selected = {}
    for name, target in [('latitude',latitude), ('longitude',longitude)]:
        values = ds[name].values
        if len(values) < 2 or not values[0] <= target <= values[-1]:
            raise ValueError('Wave point outside wind interpolation grid')
        right = min(max(int(np.searchsorted(values, target)), 1), len(values)-1)
        selected[name] = values[right-1:right+1]
    box = ds.sel(selected)[WIND_VARIABLES]
    if any(not np.isfinite(box[v]).all() or (abs(box[v]) > 100).any() for v in WIND_VARIABLES):
        raise ValueError('Missing or invalid neighboring wind measurements; no fallback or filling')
    result = box.interp(latitude=latitude, longitude=longitude, method='linear')
    for name in WIND_VARIABLES:
        if not np.isfinite(result[name]).all() or ((result[name] < box[name].min(['latitude','longitude'])-1e-6) |
                                                  (result[name] > box[name].max(['latitude','longitude'])+1e-6)).any():
            raise ValueError('Invalid bilinear interpolation result')
    result['wind_speed'] = np.hypot(result.eastward_wind, result.northward_wind)
    print('Wind interpolation grid:', {name:values.tolist() for name,values in selected.items()})
    return result


def combine(waves, wind):
    def point_frame(ds):
        frame = ds.to_dataframe().drop(columns=['latitude','longitude'], errors='ignore')
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True), name='timestamp').as_unit('ns')
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError('Duplicate or unsorted timestamps')
        return frame
    wave_frame, wind_frame = point_frame(waves), point_frame(wind)
    for coord in ('latitude','longitude'):
        if float(waves[coord]) != float(wind[coord]):
            raise ValueError('Wave/wind reference coordinates differ')
    missing = wave_frame.index.difference(wind_frame.index)
    if len(missing):
        raise ValueError(f'Missing exact wind timestamps: {list(map(str,missing))}')
    frame = wave_frame.join(wind_frame, how='left', validate='one_to_one')
    frame.insert(0,'longitude',float(waves.longitude))
    frame.insert(0,'latitude',float(waves.latitude))
    return frame


def validate_csv_data(frame, start, end):
    check_times(frame.index, start, end, '3h')
    missing = frame.isna().sum()
    print(f'Timestamps: {len(frame)} / {len(pd.date_range(start,end,freq="3h"))}; '
          f'duplicates: {frame.index.has_duplicates}; missing values: {int(missing.sum())}')
    if missing.any():
        raise ValueError(f'Missing measurements (not filled): {missing[missing > 0].to_dict()}')
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Nonfinite source measurements')
    for name in frame:
        if name.startswith(('VHM','VTM','VTP')) and (frame[name] < 0).any():
            raise ValueError(f'Negative source height/period: {name}')
        if name.startswith('VMDR') and not frame[name].between(0,360).all():
            raise ValueError(f'Invalid source direction: {name}')
    if len(frame[['latitude','longitude']].drop_duplicates()) != 1:
        raise ValueError('Reference coordinates change between rows')
    np.testing.assert_allclose(frame.wind_speed, np.hypot(frame.eastward_wind,frame.northward_wind))


def prepare(full=False, download=False):
    start = pd.Timestamp('2015-01-01' if full else '2022-01-01')
    end = pd.Timestamp('2024-12-31T21:00:00' if full else '2022-01-07T23:00:00')
    wave_end = end.floor('3h')
    wave_meta = catalogue(WAVE_ID,start,wave_end) if download else None
    wind_meta = catalogue(WIND_ID,start,end) if download else None
    frames = []
    reference = None
    for year in range(start.year,end.year+1):
        left = max(start,pd.Timestamp(year,1,1))
        right = min(end,pd.Timestamp(year+1,1,1)-pd.Timedelta(hours=1))
        wave_grid = load_chunk('wave',left,right.floor('3h'),LATITUDE,LONGITUDE,wave_meta,download)
        if reference is None:
            waves,lat,lon = select_wave_cell(wave_grid)
            reference = lat,lon
        else:
            lat,lon = reference
            waves = wave_grid.sel(latitude=lat,longitude=lon)  # Keep exactly the same cell across years.
        wave_variables = [v for v in waves.data_vars if v.startswith(('VHM','VMDR','VTM','VTP'))]
        waves = waves[wave_meta[1] if wave_meta else wave_variables]
        wind_grid = load_chunk('wind',left,right,lat,lon,wind_meta,download)
        wind = interpolate_wind(wind_grid,lat,lon)
        frames.append(combine(waves,wind))
    frame = pd.concat(frames)
    validate_csv_data(frame,start,wave_end)
    suffix = '2015_2024' if full else 'sample_2022'
    output = ROOT/'data'/f'udupi_wave_wind_{suffix}.csv'
    temporary = output.with_suffix('.partial.csv')
    try:
        frame.to_csv(temporary,index_label='timestamp',encoding='utf-8')
        loaded = pd.read_csv(temporary,index_col='timestamp',parse_dates=['timestamp'],float_precision='round_trip')
        loaded.index = pd.DatetimeIndex(loaded.index).as_unit('ns')
        pd.testing.assert_frame_equal(frame,loaded,check_dtype=False,check_freq=False,rtol=1e-12,atol=1e-12)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Reference: {lat}, {lon}; CSV: {output.relative_to(ROOT)}; shape: {frame.shape}')
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-approved',action='store_true',help='Confirm approval and select 2015-2024 instead of the sample')
    parser.add_argument('--download',action='store_true',help='Allow catalogue requests and missing-data downloads; default is offline')
    args = parser.parse_args()
    prepare(full=args.full_approved,download=args.download)


if __name__ == '__main__':
    main()
