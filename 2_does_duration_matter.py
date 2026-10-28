"""
Blocking persistence tests 

Part 1  % of blocking with heatwave occurrences, for different blocking persistence

Part 2  Machine learning model 
Linear regression , multivariation regression and randon forest model trained on blocking heatwave catelouge 
Blocking persistence only trained model 
all blocking properties   lifetime, size, strength, position, time of year, recurrence
Fair test: every blocking is predicted by a model that never saw its summer.
  Scores:
    pair test   take one block with a heatwave and one without. How often does the model
                give the heatwave block the higher chance? (50% = coin toss)
    R2          for the heatwaves: share of the differences in intensity, duration and area
                that is explained (0 = no better than guessing the average)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import r2_score, roc_auc_score

from common import OUT, BLOCK_FEATURES, HW_TARGETS, CLASSIFIERS, REGRESSORS, load_table, predict_unseen

CLASS_EDGES = [0, 6, 10, 15, 100]
CLASS_NAMES = ['up to 6 d', '6 to 10 d', '10 to 15 d', 'over 15 d']
INFORMATION = {'persistence only': ['lifetime_days'], 'all block properties': list(BLOCK_FEATURES)}
COLORS = {'persistence only': '#f0b429', 'all block properties': '#7a56b5'}
SINGLE = {'persistence': ['lifetime_days'], 'size': ['max_size_1e6km2'], 'strength (PV)': ['min_pv'],
          'position': ['lat', 'lon_sin', 'lon_cos'], 'time of year': ['doy']}
def counting(t):
    """Part 1: share of blocks with a heatwave, and the heatwaves, by block lifetime."""
    t = t.assign(lifetime_class=pd.cut(t['lifetime_days'], CLASS_EDGES, labels=CLASS_NAMES))
    g = t.groupby('lifetime_class', observed=True)
    hw = t[t['heatwave'] == 1].groupby('lifetime_class', observed=True)
    table = pd.DataFrame({'blocks': g.size(),
                          'with heatwave (n)': g['heatwave'].sum(),
                          'with heatwave (%)': (g['heatwave'].mean() * 100).round(0),
                          'intensity (K)': hw['hw_max_intensity'].median().round(1),
                          'duration (d)': hw['hw_duration_days'].median().round(1),
                          'area (10^6 km2)': hw['hw_max_area_1e6'].median().round(1)})
    return t, table


def machine_learning(t):
    """Part 2: pair test for 'does a block bring a heatwave', R2 for 'what is the heatwave like'."""
    y, years = t['heatwave'].values, t['year'].values
    pair = {}
    for info, feats in INFORMATION.items():
        for name, model in CLASSIFIERS.items():
            p = predict_unseen(model, t[feats].values, y, years, proba=True)
            pair[(info, name)] = roc_auc_score(y, p) * 100

    h = t[t['heatwave'] == 1]
    r2 = {}
    for target in HW_TARGETS:
        for info, feats in INFORMATION.items():
            for name, model in REGRESSORS.items():
                p = predict_unseen(model, h[feats].values, h[target].values, h['year'].values)
                r2[(target, info, name)] = r2_score(h[target], p)
    return pair, r2


def draw_all(t, pair, r2):
    """Three simple figures, one message each."""
    mean = t['heatwave'].mean() * 100
    by = t.groupby('lifetime_class', observed=True)

        # Figure 1: do more persistent blocks bring heatwaves more often? (counting)
    life, hw = t['lifetime_days'], t['heatwave']
    days = [d for d in range(int(np.ceil(life.min())), 31) if (life >= d).sum() >= 10]
    n = np.array([(life >= d).sum() for d in days])
    share = np.array([hw[life >= d].mean() * 100 for d in days])    # blocks lasting AT LEAST d days
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=150)
    for d, sh, k in zip(days, share, n):
        ax.bar(d, sh, width=0.92, color='#d1495b', alpha=0.85 if k >= 30 else 0.35,
       edgecolor='white', linewidth=0.8)
    ax.axhline(mean, color='black', linestyle='--', linewidth=1)
    ax.text(days[0], share[0] + 0.6, f'all blocks\n{share[0]:.0f}%', ha='center', fontsize=9)
    for d in (10, 15):                                              # label a few bars only
        if d in days:
            ax.text(d, share[days.index(d)] + 0.6, f'{share[days.index(d)]:.0f}%',
                    ha='center', fontsize=10, weight='bold')
    if (n < 30).any():
        ax.text(0.99, 0.04, 'pale bars: fewer than 30 blocks', transform=ax.transAxes,
                ha='right', fontsize=8, color='#555555')
    ax.set_xticks([d for d in days if d % 5 == 0])
    ax.set_ylim(0, share.max() * 1.25)
    ax.set_xlabel('Blocking persistence (days)')
    ax.set_ylabel('% of blocking with heatwave occurrences')
    ax.set_title('', loc='left')
    fig.tight_layout()
    fig.savefig(OUT / 'fig1_persistence.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 2: the machine-learning models, in one figure
    grid = np.linspace(t['lifetime_days'].min(), min(t['lifetime_days'].max(), 25), 100)
    curves = {}
    for name, model in CLASSIFIERS.items():
        fit = clone(model).fit(t[['lifetime_days']].values, t['heatwave'].values)
        c = fit.predict_proba(grid.reshape(-1, 1))[:, 1] * 100
        if name == 'random forest':                 # a forest follows the noise: smooth it for display only
            c = pd.Series(c).rolling(21, center=True, min_periods=1).mean().values
        curves[name] = c
    top = max(c.max() for c in curves.values()) * 1.7
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=150, gridspec_kw={'width_ratios': [1.25, 1]})

    # (a) heatwave occurrence: the chance of a heatwave as a function of blocking persistence
    style = {'logistic regression': '-', 'random forest': '--'}
    for name, c in curves.items():
        ax.plot(grid, c, style[name], color='#c98a00', linewidth=3,
                label=f'{name}{" (smoothed)" if name == "random forest" else ""}, persistence only: '
                      f'{pair[("persistence only", name)]:.0f}%')
    ax.plot([], [], 's', color='#7a56b5',
            label='both models, all block properties: '
                  f'{pair[("all block properties", "logistic regression")]:.0f}% / '
                  f'{pair[("all block properties", "random forest")]:.0f}%')
    yes, no = t[t['heatwave'] == 1]['lifetime_days'], t[t['heatwave'] == 0]['lifetime_days']
    ax.plot(yes, np.full(len(yes), top - 0.8), '|', color='#d1495b', alpha=0.6, markersize=12)
    ax.plot(no, np.full(len(no), -0.8), '|', color='#555555', alpha=0.15, markersize=12)
    ax.text(grid.max(), top - 3.4, 'blocks that brought a heatwave', ha='right', color='#d1495b', fontsize=9, bbox=dict(facecolor='white', edgecolor='none', pad=1.5, alpha=0.85))
    ax.text(grid.max(), 1.6, 'blocks without a heatwave', ha='right', color='#555555', fontsize=9, bbox=dict(facecolor='white', edgecolor='none', pad=1.5, alpha=0.85))
    ax.set_xlim(grid.min() - 0.5, grid.max() + 0.5)
    ax.set_ylim(-2.5, top + 1)
    ax.set_xlabel('Blocking persistence (days)')
    ax.set_ylabel('Chance that the block brings a heatwave (%)')
    ax.set_title('(a) Persistence tells whether a heatwave occurs', loc='left', fontsize=12)
    ax.legend(title='Picks the block that brought the heatwave\n(pair test; guessing = 50%)',
              loc='upper left', bbox_to_anchor=(0.01, 0.92), fontsize=9, title_fontsize=9)
    ax.grid(alpha=0.3)

    # (b) heatwave properties: how much do the models explain?
    best = lambda tg, info: max(r2[(tg, info, m)] for m in REGRESSORS)
    allv = [best(tg, i) for tg in HW_TARGETS for i in INFORMATION]
    lo, hi = min(-0.35, min(allv) - 0.05), max(0.35, max(allv) + 0.08)
    bx.axvspan(lo, 0, color='#f4cccc', alpha=0.6)
    bx.axvline(0, color='black', linewidth=1)
    for k, tg in enumerate(HW_TARGETS):
        a_, b_ = best(tg, 'persistence only'), best(tg, 'all block properties')
        bx.plot([a_, b_], [k, k], color='#999999', linewidth=3, zorder=1)
        bx.scatter([a_], [k], s=170, color='#f0b429', edgecolor='black', zorder=3,
                   label='persistence only' if k == 0 else None)
        bx.scatter([b_], [k], s=170, color='#7a56b5', edgecolor='black', zorder=3,
                   label='all block properties' if k == 0 else None)
    bx.set_yticks(range(3))
    bx.set_yticklabels(['intensity', 'duration', 'area'], fontsize=12)
    bx.set_ylim(2.7, -0.7)
    bx.set_xlim(lo, hi)
    bx.text(lo + 0.01, -0.62, 'worse than guessing\nthe average', fontsize=9, color='#a33', va='top')
    bx.set_xlabel('Share of the differences between heatwaves that is explained ($R^2$)')
    bx.set_title('(b) Persistence does not tell what the heatwave is like', loc='left', fontsize=12)
    bx.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10)
    fig.suptitle('Machine-learning models: logistic / linear regression and random forest',
                 x=0.01, ha='left', fontsize=14, weight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'fig2_ml_models.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 3: does the heatwave change with the blocking persistence? (counting)
    h = t[t['heatwave'] == 1].groupby('lifetime_class', observed=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=150)
    for ax, (col, name, unit) in zip(axes, [('hw_max_intensity', 'intensity', 'K'),
                                            ('hw_duration_days', 'duration', 'days'),
                                            ('hw_max_area_1e6', 'area', '10^6 km²')]):
        med = h[col].median().reindex(CLASS_NAMES).fillna(0)
        n = h.size().reindex(CLASS_NAMES).fillna(0).astype(int)
        ax.bar(range(len(med)), med, color='#7a56b5', edgecolor='black')
        for i, (m, k) in enumerate(zip(med, n)):
            if k == 0:
                continue
            ax.text(i, m + 0.02 * med.max(), f'{m:.1f}', ha='center', fontsize=11, weight='bold')
            ax.text(i, 0.04 * med.max(), f'n={k}', ha='center', color='white', fontsize=8)
        ax.set_xticks(range(len(med)))
        ax.set_xticklabels(CLASS_NAMES, fontsize=8)
        ax.set_ylim(0, med.max() * 1.18)
        ax.set_xlabel('Blocking persistence')
        ax.set_ylabel(f'Median heatwave {name} ({unit})')
    fig.suptitle('The heatwave looks the same whatever the blocking persistence', x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(OUT / 'fig3_heatwave.png', bbox_inches='tight')
    plt.close(fig)



def one_property_at_a_time(t):
    """Pair test with ONE block property at a time: which property alone tells whether a heatwave occurs?"""
    y, years = t['heatwave'].values, t['year'].values
    rows = {name: {m: roc_auc_score(y, predict_unseen(model, t[feats].values, y, years, proba=True)) * 100
                   for m, model in CLASSIFIERS.items()}
            for name, feats in SINGLE.items()}
    return pd.DataFrame(rows).T


def draw_one_property(one):
    one = one.loc[one.max(axis=1).sort_values().index]      # most telling property at the top
    models = {'logistic regression': ('#6baed6', 'ML-regression'),
              'random forest': ('#fd8d3c', 'ML-forest')}
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=150)
    y = np.arange(len(one))
    for k, (model, (color, label)) in enumerate(models.items()):
        pos, v = y + (k - 0.5) * 0.38, one[model].values
        ax.barh(pos, v - 50, left=50, height=0.36, color=color, alpha=0.6, label=label)
        for p, vi in zip(pos, v):
            ax.text(max(vi, 50) + 0.4, p, f'{vi:.0f}%', va='center', fontsize=8)
    ax.axvline(50, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(one.index)
    ax.set_xlim(min(45, one.values.min() - 2), max(70, one.values.max() + 5))
    ax.set_xlabel('Pair test (%). 50% = guessing')
    ax.set_title('ML diagnostics: which blocking properties tell heatwave occurrence?', loc='left', fontsize=11)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'fig4_one_property.png', bbox_inches='tight')
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    t, table = counting(load_table())
    print(f'{len(t)} blocks, {t["heatwave"].sum()} with a heatwave ({t["heatwave"].mean() * 100:.0f}%)\n')
    print('PART 1. Counting: blocks and heatwaves by blocking persistence\n')
    print(table.to_string(), '\n')
    table.to_csv(OUT / 'by_lifetime.csv')

    pair, r2 = machine_learning(t)
    print('PART 2a. Does a block bring a heatwave? Pair test in % (50 = coin toss)\n')
    print(pd.DataFrame({m: {i: round(pair[(i, m)]) for i in INFORMATION} for m in CLASSIFIERS}).to_string(), '\n')
    one = one_property_at_a_time(t)
    print('PART 2a-bis. Which block property ALONE tells whether a heatwave occurs? Pair test in % (50 = guessing)\n')
    print(one.round(0).to_string(), '\n')
    draw_one_property(one)


    print('PART 2b. If it does, what is the heatwave like? R2 (0 = guessing the average)\n')
    short = {'linear regression': 'linear', 'random forest': 'forest'}
    print(pd.DataFrame({f'{i} / {short[m]}': {HW_TARGETS[tg]: round(r2[(tg, i, m)], 2) for tg in HW_TARGETS}
                        for i in INFORMATION for m in REGRESSORS}).to_string(), '\n')
    draw_all(t, pair, r2)
    print(f'written {OUT / "by_lifetime.csv"} and the figures fig1_persistence, fig2_ml_models, fig3_heatwave in {OUT}/')


if __name__ == '__main__':
    main()
