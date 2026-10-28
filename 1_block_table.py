"""
Step 1. Make one row per block, with its properties and the heatwave that came with it.
"""
import numpy as np
import pandas as pd

from common import (BLOCK_CSV, ASSOC_CSV, TABLE, RECURRENCE_DAYS, RECURRENCE_KM, haversine_km)


def block_properties(blk):
    """One row per block track."""
    blk = blk.sort_values('Date')
    g = blk.groupby('Flag')
    t = pd.DataFrame({
        'onset': g['Date'].min(),
        'end': g['Date'].max(),
        'lifetime_days': g.size() / 4,                      # 6-hourly steps
        'max_size_1e6km2': g['Size'].max() / 1e6,
        'min_pv': g['Intensity'].min(),                     # most negative = strongest
        'lat': g['Latitude'].mean(),
        'lon': g['Longitude'].mean(),
        'lat_onset': g['Latitude'].first(),
        'lon_onset': g['Longitude'].first(),
    })
    t['year'] = t['onset'].dt.year
    t['doy'] = t['onset'].dt.dayofyear
    t['lon_sin'] = np.sin(np.radians(t['lon']))
    t['lon_cos'] = np.cos(np.radians(t['lon']))
    return t


def add_recurrence(t):
    """Number of other blocks that started in the previous days and were close by."""
    counts = []
    for _, b in t.iterrows():
        prev = t[(t['onset'] < b['onset']) &
                 (t['onset'] >= b['onset'] - pd.Timedelta(days=RECURRENCE_DAYS))]
        d = haversine_km(b['lat_onset'], b['lon_onset'], prev['lat'].values, prev['lon'].values)
        counts.append(int((d <= RECURRENCE_KM).sum()))
    t['recurrence'] = counts
    return t


def add_heatwaves(t, assoc):
    """For each block: did a heatwave come with it, and what was it like (the most intense one)."""
    rows = []
    for _, hw in assoc[assoc['associated']].iterrows():
        for bid in str(hw['matched_block_ids']).split(','):
            if bid.strip():
                rows.append({'Flag': int(float(bid)), 'Flag_HW': hw['Flag_HW'],
                             'hw_max_intensity': hw['hw_max_intensity'],
                             'hw_duration_days': hw['hw_duration_days'],
                             'hw_max_area_1e6': hw['hw_max_area'] / 1e6,
                             'hw_lat': hw['lat_mean'], 'hw_lon': hw['lon_mean']})
    links = pd.DataFrame(rows)
    n_hw = links.groupby('Flag').size().rename('n_heatwaves')
    strongest = links.sort_values('hw_max_intensity').groupby('Flag').last()
    t = t.join(n_hw).join(strongest)
    t['n_heatwaves'] = t['n_heatwaves'].fillna(0).astype(int)
    t['heatwave'] = (t['n_heatwaves'] > 0).astype(int)
    return t


def main():
    blk = pd.read_csv(BLOCK_CSV)
    blk['Date'] = pd.to_datetime(blk['Date'].astype(str), format='%Y%m%d_%H')
    assoc = pd.read_csv(ASSOC_CSV, parse_dates=['onset'])

    t = block_properties(blk)
    t = t[t['year'].isin(assoc['onset'].dt.year.unique())]     # same summers as the heatwaves
    t = add_recurrence(t)
    t = add_heatwaves(t, assoc)
    t.index.name = 'Flag_BLK'
    t.to_csv(TABLE)

    print(f'{len(t)} blocks in {t["year"].nunique()} summers; '
          f'{t["heatwave"].sum()} came with a heatwave ({t["heatwave"].mean() * 100:.0f}%)\n')
    print(f'written {TABLE}')


if __name__ == '__main__':
    main()
