"""
Settings and helpers shared by all scripts. Change things here, not in the scripts.

Idea ("perfect prognosis"): assume the blocking is known, from a forecast or a climate
projection, and learn from the catalogues what heatwave comes with it.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import PredefinedSplit, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================================ files
BLOCK_CSV = Path('data/blocking_catalogue.csv')            # one row per block and time step
ASSOC_CSV = Path('data/heatwave_block_association.csv')    # one row per heatwave
TABLE = Path('data/block_table.csv')                       # made by 1_block_table.py
OUT = Path('results')

# ============================================================================ settings
SEED = 42
N_FOLDS = 5                     # summers are split into 5 groups for the fair test
RECURRENCE_DAYS = 30            # recurrence: blocks in the previous 30 days ...
RECURRENCE_KM = 2000            # ... whose centre was within 2000 km

# what the models see: properties of the block only
BLOCK_FEATURES = {
    'lifetime_days':   'Block persistence (days)',
    'max_size_1e6km2': 'Block maximum size (10^6 km²)',
    'min_pv':          'Block strength (most negative PV anomaly)',
    'lat':             'Block latitude (°N)',
    'lon_sin':         'Block longitude (sine)',
    'lon_cos':         'Block longitude (cosine)',
    'doy':             'Day of the year at block onset',
    #'recurrence':      f'Blocks nearby in the previous {RECURRENCE_DAYS} days',
}

# what we want to know about the heatwave that comes with the block
HW_TARGETS = {
    'hw_max_intensity': 'Heatwave maximum intensity (K)',
    'hw_duration_days': 'Heatwave duration (days)',
    'hw_max_area_1e6':  'Heatwave maximum area (10^6 km²)',
}

CLASSIFIERS = {
    'logistic regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    'random forest': RandomForestClassifier(n_estimators=500, min_samples_leaf=5, random_state=SEED),
}
REGRESSORS = {
    'linear regression': make_pipeline(StandardScaler(), LinearRegression()),
    'random forest': RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=SEED),
}


# association rule used for the chance baseline (simplified version of the paper's rule, see 4_chance_baseline.py)
ALPHA = 1.2                     # circles of the block and of the heatwave are enlarged by this factor
LAG_DAYS = (-2, -1)             # the block is seen one or two days before the heatwave time step
MIN_STEPS = 4                   # at least 4 six-hourly steps (1 day) of overlap

REGIONS = {'North America': (-170, -50), 'Europe': (-50, 40), 'Asia': (40, 170)}   # as in the paper

# ============================================================================ helpers
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def load_table():
    """The block table made by 1_block_table.py."""
    if not TABLE.exists():
        raise SystemExit(f'{TABLE} not found: run python 1_block_table.py first')
    t = pd.read_csv(TABLE, parse_dates=['onset', 'end'])
    t[list(BLOCK_FEATURES)] = t[list(BLOCK_FEATURES)].astype(float)
    return t


def unseen_summer_folds(years, seed=SEED):
    """Whole summers go into the same group, so a model is never tested on a summer it saw."""
    rng = np.random.default_rng(seed)
    group = {yr: i % N_FOLDS for i, yr in enumerate(rng.permutation(np.unique(years)))}
    return PredefinedSplit(np.array([group[y] for y in years]))


def predict_unseen(model, X, y, years, proba=False, seed=SEED):
    """Predict every row with a model that never saw its summer."""
    method = 'predict_proba' if proba else 'predict'
    p = cross_val_predict(clone(model), X, y, cv=unseen_summer_folds(years, seed), method=method)
    return p[:, 1] if proba else p


def region_of(lon):
    """Region of a block from its mean longitude (None over the Pacific)."""
    for name, (lon0, lon1) in REGIONS.items():
        if lon0 <= lon < lon1:
            return name
    return None


# ============================================================================ chance level
CHANCE_CSV = Path('data/block_chance.csv')       # saved by add_chance_level, so it is computed only once


def block_lookup():
    """For each 6-hourly time: position and radius of all blocks present, and their ids."""
    blk = pd.read_csv(BLOCK_CSV)
    blk['Date'] = pd.to_datetime(blk['Date'].astype(str), format='%Y%m%d_%H')
    return {t: (g['Latitude'].values, g['Longitude'].values, np.sqrt(g['Size'].values / np.pi), g['Flag'].values)
            for t, g in blk.groupby('Date')}


def meets(hw, year, lookup):
    """Blocks that this heatwave, placed in `year` on the same calendar dates, would be associated with.
    Simplified rule: the heatwave is a circle with its mean position and mean area during its whole life."""
    onset = hw['onset'].replace(year=int(year))
    n = int(round(hw['hw_duration_days'] * 4))
    r_hw = np.sqrt(hw['hw_mean_area'] / np.pi)
    best, best_ids = -1, set()
    for lag in LAG_DAYS:
        steps, ids = 0, set()
        for k in range(n):
            b = lookup.get(onset + pd.Timedelta(hours=6 * k) + pd.Timedelta(days=lag))
            if b is None:
                continue
            hit = haversine_km(hw['lat_mean'], hw['lon_mean'], b[0], b[1]) <= ALPHA * (r_hw + b[2])
            if hit.any():
                steps += 1
                ids.update(b[3][hit].astype(int))
        if steps > best:
            best, best_ids = steps, ids
    return best_ids if best >= MIN_STEPS else set()


def add_chance_level(t):
    """Add the CHANCE LEVEL of every block: how often it would meet a heatwave if heatwaves happened at their
    usual times and places but had nothing to do with that block. Each block is paired with the heatwaves of the
    OTHER summers, moved to the calendar dates of its own summer (realistic, but physically unrelated)."""
    t = t.copy()
    if CHANCE_CSV.exists():
        c = pd.read_csv(CHANCE_CSV)
        if len(c) == len(t) and (c['Flag_BLK'].values == t['Flag_BLK'].values).all():
            t['chance'] = c['chance'].values
    if 'chance' not in t:
        assoc = pd.read_csv(ASSOC_CSV, parse_dates=['onset'])
        lookup = block_lookup()
        years = sorted(assoc['onset'].dt.year.unique())
        hits = {}
        for _, hw in assoc.iterrows():
            y0 = hw['onset'].year
            for year in years:
                if year != y0:
                    hits.setdefault((y0, year), set()).update(meets(hw, year, lookup))
        t['chance'] = [np.mean([flag in hits.get((y0, year), set()) for y0 in years if y0 != year])
                       for flag, year in zip(t['Flag_BLK'], t['year'])]
        t[['Flag_BLK', 'chance']].to_csv(CHANCE_CSV, index=False)
    c = np.clip(t['chance'], 1e-3, 1 - 1e-3)
    t['chance_logit'] = np.log(c / (1 - c))                  # the form the models use

    return t




from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

# extra model for the comparison; the figures keep using CLASSIFIERS / REGRESSORS above
_boost = dict(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=10, random_state=SEED)
ALL_CLASSIFIERS = {**CLASSIFIERS, 'gradient boosting': HistGradientBoostingClassifier(**_boost)}
ALL_REGRESSORS = {**REGRESSORS, 'gradient boosting': HistGradientBoostingRegressor(**_boost)}













