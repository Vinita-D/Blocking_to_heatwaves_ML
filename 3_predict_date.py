"""
 Pick a date. Which blocks were there, and what heatwave did the models expect?

For every block active within a few days of the date, the script shows:
  - its properties (assumed known, as from a forecast) and its region
  - the chance of a heatwave (random forest trained on OTHER summers)
  - the expected heatwave intensity and duration, if one comes
  - what really happened in ERA5, including the number of heatwaves linked to the block
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone

from common import OUT, BLOCK_FEATURES, CLASSIFIERS, REGRESSORS, load_table, region_of

# ============================================================================ settings
DATE = '2021-06-27'                  # the date to look at (YYYY-MM-DD)
WINDOW_DAYS = 0                      # 0 = blocks active on that exact date; 3 = within 3 days


def main():
    OUT.mkdir(exist_ok=True)
    t = load_table().reset_index(drop=True)
    day, w = pd.Timestamp(DATE), pd.Timedelta(days=WINDOW_DAYS)
    now = t[(t['onset'] <= day + w) & (t['end'] >= day - w)]
    print(f'Blocks active within {WINDOW_DAYS} days of {day:%d %b %Y}: {len(now)}\n')
    if now.empty:
        return

    # train on every other summer, so the models never saw this one
    feats = list(BLOCK_FEATURES)
    train = t[t['year'] != day.year]
    clf = clone(CLASSIFIERS['random forest']).fit(train[feats], train['heatwave'])
    hw_train = train[train['heatwave'] == 1]
    reg = {h: clone(REGRESSORS['random forest']).fit(hw_train[feats], hw_train[h])
           for h in ['hw_max_intensity', 'hw_duration_days']}

    rows = []
    for no, (_, b) in enumerate(now.iterrows(), start=1):
        x = b[feats].to_frame().T.astype(float)
        rows.append({
            'No.': no, 'Block ID': b['Flag_BLK'],
            'Dates': f"{b['onset']:%d %b}-{b['end']:%d %b}",
            'Lat': round(b['lat']), 'Lon': round(b['lon']),
            'Region': region_of(b['lon']) or 'Pacific',
            'Days': b['lifetime_days'], 'Size': round(b['max_size_1e6km2'], 1),
            'PV': round(b['min_pv'], 1),
            'Chance (%)': round(clf.predict_proba(x)[0, 1] * 100),
            'Expected K': round(reg['hw_max_intensity'].predict(x)[0], 1),
            'Expected d': round(reg['hw_duration_days'].predict(x)[0], 1),
            'ERA5: heatwaves': int(b['n_heatwaves']),
            'ERA5 K': round(b['hw_max_intensity'], 1) if b['heatwave'] else '-',
            'ERA5 d': b['hw_duration_days'] if b['heatwave'] else '-',
        })
    table = pd.DataFrame(rows)
    print('Size in 10^6 km2, PV = most negative PV anomaly, Expected = if a heatwave comes\n')
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(table.to_string(index=False))

    # figure: the blocks on a map, coloured by the chance of a heatwave
    fig = plt.figure(figsize=(13, 4.5), dpi=150)
    try:
        import cartopy.crs as ccrs
        ax = fig.add_subplot(projection=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.6)
        ax.set_extent([-180, 180, 20, 85], ccrs.PlateCarree())
        kw = dict(transform=ccrs.PlateCarree())
    except ImportError:
        ax = fig.add_subplot()
        ax.set_xlim(-180, 180)
        ax.set_ylim(20, 85)
        kw = {}
    sc = ax.scatter(now['lon'], now['lat'], s=now['max_size_1e6km2'] * 120, c=table['Chance (%)'],
                    cmap='Reds', vmin=0, vmax=100, edgecolor='black', zorder=5, **kw)
    hw = now[now['heatwave'] == 1]
    ax.scatter(hw['hw_lon'], hw['hw_lat'], marker='*', s=250, color='gold', edgecolor='black',
               zorder=6, label='heatwave in ERA5', **kw)
    for (_, b), no in zip(now.iterrows(), table['No.']):
        ax.text(b['lon'] + 3, b['lat'] + 3, str(no), fontsize=12, weight='bold', zorder=7, **kw)
    plt.colorbar(sc, ax=ax, orientation='horizontal', shrink=0.5, pad=0.08, label='Chance of a heatwave (%)')
    ax.legend(loc='lower left')
    ax.set_title(f'Blocks on {day:%d %b %Y} (circle size = block size)', loc='left')
    name = f'blocks_{day:%Y-%m-%d}'
    table.to_csv(OUT / f'{name}.csv', index=False)
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight')
    plt.close(fig)
    print(f'\nwritten {OUT / (name + ".csv")} and {OUT / (name + ".png")}')


if __name__ == '__main__':
    main()
