"""
Blocking persistence and heatwaves: a simple machine-learning diagnostic.

Part 1  persistence of blocking leads to more heatwave occurrences or is it just by chance 
        A longer blocking has more time steps in which a heatwave could occur, even if the two are unrelated.
        So the chnaces are computed by counting how often it would meet a heatwave same place same location but for other years summer

Part 2  Machine learning. Linear/logistic regression and random forest are trained on the blocking-heatwave
        catalogue, and tested on summers they never saw ("fair test").
        Scores:

"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.base import clone
from sklearn.metrics import r2_score, roc_auc_score

from common import (OUT, BLOCK_FEATURES, HW_TARGETS, CLASSIFIERS, REGRESSORS, load_table, predict_unseen,
                    add_chance_level, unseen_summer_folds)

CLASS_EDGES = [0, 6, 10, 15, 100]
CLASS_NAMES = ['up to 6 d', '6 to 10 d', '10 to 15 d', 'over 15 d']
INFORMATION = {'persistence only': ['lifetime_days'], 'all block properties': list(BLOCK_FEATURES)}
CHANCE_SETS = {'chance level only': ['chance_logit'], 'chance + persistence': ['chance_logit', 'lifetime_days']}
SINGLE = {'persistence': ['lifetime_days'], 'size': ['max_size_1e6km2'], 'strength (PV)': ['min_pv'],
          'position': ['lat', 'lon_sin', 'lon_cos'], 'time of year': ['doy']}
MODEL_COLORS = {'logistic regression': ('#6baed6', 'ML-regression'), 'random forest': ('#fd8d3c', 'ML-forest')}
PURPLE, GOLD, RED = '#7a56b5', '#f0b429', '#d1495b'


# ============================================================================ numbers
def counting(t):
    t = t.assign(lifetime_class=pd.cut(t['lifetime_days'], CLASS_EDGES, labels=CLASS_NAMES))
    g = t.groupby('lifetime_class', observed=True)
    hw = t[t['heatwave'] == 1].groupby('lifetime_class', observed=True)
    table = pd.DataFrame({'blocks': g.size(),
                          'with heatwave (n)': g['heatwave'].sum(),
                          'with heatwave (%)': (g['heatwave'].mean() * 100).round(0),
                          'expected by chance (%)': (g['chance'].mean() * 100).round(0),
                          'observed / chance': (g['heatwave'].mean() / g['chance'].mean()).round(1),
                          'intensity (K)': hw['hw_max_intensity'].median().round(1),
                          'duration (d)': hw['hw_duration_days'].median().round(1),
                          'area (10^6 km2)': hw['hw_max_area_1e6'].median().round(1)})
    return t, table


def machine_learning(t):
    """pair test for 'does a blocking bring a heatwave', R2 for 'what is the heatwave like'."""
    y, years = t['heatwave'].values, t['year'].values
    pair, pred = {}, {}
    for info, feats in {**INFORMATION, **CHANCE_SETS}.items():
        for name, model in CLASSIFIERS.items():
            pred[(info, name)] = predict_unseen(model, t[feats].values, y, years, proba=True)
            pair[(info, name)] = roc_auc_score(y, pred[(info, name)]) * 100
    h = t[t['heatwave'] == 1]
    r2 = {}
    for target in HW_TARGETS:
        for info, feats in INFORMATION.items():
            for name, model in REGRESSORS.items():
                p = predict_unseen(model, h[feats].values, h[target].values, h['year'].values)
                r2[(target, info, name)] = r2_score(h[target], p)
    return pair, r2, pred


def one_property_at_a_time(t):
    """Pair test with one blocking property at a time."""
    y, years = t['heatwave'].values, t['year'].values
    rows = {name: {m: roc_auc_score(y, predict_unseen(model, t[feats].values, y, years, proba=True)) * 100
                   for m, model in CLASSIFIERS.items()}
            for name, feats in SINGLE.items()}
    return pd.DataFrame(rows).T



def fig_chance(t):
    life, hw = t['lifetime_days'], t['heatwave']
    days = [d for d in range(int(np.ceil(life.min())), 31) if (life >= d).sum() >= 10]
    n = np.array([(life >= d).sum() for d in days])
    share = np.array([hw[life >= d].mean() * 100 for d in days])                  # blocking lasting AT LEAST d days
    chance = np.array([t.loc[life >= d, 'chance'].mean() * 100 for d in days])    # share expected by chance
    ratio = (share / chance)[n >= 30]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    for d, sh, k in zip(days, share, n):
        ax.bar(d, sh, width=0.92, color=RED, alpha=0.85 if k >= 30 else 0.35, edgecolor='white', linewidth=0.8)
    ax.plot(days, chance, 'o-', color='black', linewidth=2, markersize=4,
            label='expected by chance: a longer blocking has more chances')
    ax.text(days[0], share[0] + 0.6, f'all blocking\n{share[0]:.0f}%', ha='center', fontsize=9)
    for d in (10, 15):
        if d in days:
            ax.text(d, share[days.index(d)] + 0.6, f'{share[days.index(d)]:.0f}%', ha='center', fontsize=12)
    ax.text(0.02, 0.80, f'Blocking lead to heatwave {ratio.min():.1f} to {ratio.max():.1f} times more\n'
                        'often than chance, at every length', transform=ax.transAxes, fontsize=10)
    if (n < 30).any():
        ax.text(0.99, 0.04, 'pale bars: fewer than 30 blocking', transform=ax.transAxes, ha='right', fontsize=8, color='#555555')
    ax.set_xticks([d for d in days if d % 5 == 0])
    ax.set_ylim(0, share.max() * 1.25)
    ax.set_xlabel('Blocking persistece (days)')
    ax.set_ylabel('Blocking associated to heatwave (%)')
    ax.set_title('Longer blocking meets more heatwaves, but chance explains much of it', loc='left', fontsize=11)
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT / 'fig1_persistence.png', bbox_inches='tight')
    plt.close(fig)


def fig_training(t):
    """Figure 2: how a model learns from the catalogue and is tested on summers it never saw."""
    y, years = t['heatwave'].values, t['year'].values
    fold = unseen_summer_folds(years).test_fold                    # group of every blocking (a summer stays together)
    rounds = np.unique(fold)
    summers = np.sort(np.unique(years))
    rng = np.random.default_rng(0)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.6), dpi=150, gridspec_kw={'width_ratios': [1.15, 1.2, 1]})

    # (a) the split: in every round one group of summers is kept back for testing
    M = np.array([[fold[years == yr][0] == r for yr in summers] for r in rounds]).astype(int)
    ax1.imshow(M, aspect='auto', cmap=ListedColormap(['#cfcfcf', PURPLE]), interpolation='nearest',
               extent=[summers.min() - 0.5, summers.max() + 0.5, len(rounds) - 0.5, -0.5])
    ax1.set_yticks(range(len(rounds)))
    ax1.set_yticklabels([f'round {r + 1}' for r in rounds])
    ax1.set_xlabel('Summer')
    ax1.set_title('(a) Summers are kept back for testing', loc='left', fontsize=13)
    ax1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color='#cfcfcf'), plt.Rectangle((0, 0), 1, 1, color=PURPLE)],
               labels=['training summers: the model learns from the blocking of these',
                       'test summers: never seen during training'],
               loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=9, frameon=False)

    # (b) training: the model learns the chance of a heatwave from the training catalogue (round 1 shown)
    test = fold == rounds[0]
    X = t[['lifetime_days']].values
    model = clone(CLASSIFIERS['logistic regression']).fit(X[~test], y[~test])
    jitter = lambda k: rng.uniform(-0.035, 0.035, k)
    ax2.scatter(X[~test, 0], y[~test] + jitter((~test).sum()), s=14, color='#999999', alpha=0.5, label='training blocking (round 1)')
    ax2.scatter(X[test, 0], y[test] + jitter(test.sum()), s=30, color=PURPLE, edgecolor='white', linewidth=0.4,
                label='test blocking: unseen summers')
    grid = np.linspace(X.min(), min(X.max(), 25), 100)
    ax2.plot(grid, model.predict_proba(grid.reshape(-1, 1))[:, 1], color='black', linewidth=3,
             label='chance learned from the training summers')
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_yticklabels(['no heatwave', '50%', 'heatwave'])
    ax2.set_xlabel('Blocking persistence (days)')
    ax2.set_ylabel('Heatwave with the blocking / learned chance')
    ax2.set_title('(b) The model learns from the catalogue', loc='left', fontsize=13)
    ax2.legend(loc='center left', fontsize=9, frameon=True)

    # (c) test: every blocking is predicted by a model that never saw its summer
    p = predict_unseen(CLASSIFIERS['logistic regression'], t[list(BLOCK_FEATURES)].values, y, years, proba=True)
    bins = pd.qcut(p, 5, duplicates='drop')
    g = pd.DataFrame({'p': p * 100, 'y': y * 100}).groupby(bins, observed=True).mean()
    top = max(g.p.max(), g.y.max()) * 1.15
    ax3.plot([0, top], [0, top], '--', color='black', linewidth=1, label='perfect prediction')
    ax3.plot(g.p, g.y, 'o-', color=PURPLE, markersize=11, linewidth=2.5, label='logistic regression, all blocking properties')
    ax3.set_xlim(0, top)
    ax3.set_ylim(0, top)
    ax3.set_xlabel('Predicted chance of a heatwave (%)')
    ax3.set_ylabel('Blocking that really brought a heatwave (%)')
    ax3.set_title('(c) Predictions for unseen summers', loc='left', fontsize=13)
    ax3.text(0.03, 0.96, f'pair test: {roc_auc_score(y, p) * 100:.0f}%  (guessing: 50%)\nblocking sorted into 5 equal groups',
             transform=ax3.transAxes, va='top', fontsize=10)
    ax3.legend(loc='lower right', fontsize=9)
    fig.suptitle('Machine learning: trained on the blocking-heatwave catalogue, tested on summers it never saw',
                 x=0.01, ha='left', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / 'fig2_training.png', bbox_inches='tight')
    plt.close(fig)


def fig_occurrence(one, pair):
    """Figure 3: which information tells whether a blocking brings a heatwave?"""
    rows = one.copy()
    rows.loc['chance level\n(length, place, season)'] = [pair[('chance level only', m)] for m in CLASSIFIERS]
    rows.loc['all blocking properties'] = [pair[('all block properties', m)] for m in CLASSIFIERS]
    rows = rows.loc[rows.max(axis=1).sort_values().index]            # most telling at the top
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    y = np.arange(len(rows))
    for k, (model, (color, label)) in enumerate(MODEL_COLORS.items()):
        pos, v = y + (k - 0.5) * 0.38, rows[model].values
        ax.barh(pos, v - 50, left=50, height=0.36, color=color, alpha=0.75, label=label)
        for p_, vi in zip(pos, v):
            ax.text(max(vi, 50) + 0.4, p_, f'{vi:.0f}%', va='center', fontsize=9)
    ax.axvline(50, color='black', linestyle='--', linewidth=1)
    ax.text(50.3, len(rows) - 0.45, 'guessing', fontsize=9, va='bottom')
    ax.set_yticks(y)
    ax.set_yticklabels(rows.index)
    for lab in ax.get_yticklabels():
        if lab.get_text().startswith(('chance level', 'all blocking')):
            lab.set_fontweight('bold')
    ax.set_xlim(45, max(72, rows.values.max() + 5))
    ax.set_xlabel('Pair test (%): how often the model picks the blocking that brought the heatwave')
    ax.set_title('Which information tells whether a blocking brings a heatwave?', loc='left', fontsize=11)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'fig3_occurrence.png', bbox_inches='tight')
    plt.close(fig)


def fig_severity(r2):
    """Figure 4: can the model tell how severe the heatwave is?"""
    best = lambda tg, info: max(r2[(tg, info, m)] for m in REGRESSORS)
    allv = [best(tg, i) for tg in HW_TARGETS for i in INFORMATION]
    lo, hi = min(-0.35, min(allv) - 0.05), max(0.35, max(allv) + 0.08)
    fig, bx = plt.subplots(figsize=(7.5, 4.4), dpi=150)
    bx.axvspan(lo, 0, color='#f4cccc', alpha=0.6)
    bx.axvline(0, color='black', linewidth=1)
    for k, tg in enumerate(HW_TARGETS):
        a_, b_ = best(tg, 'persistence only'), best(tg, 'all block properties')
        bx.plot([a_, b_], [k, k], color='#999999', linewidth=3, zorder=1)
        bx.scatter([a_], [k], s=170, color=GOLD, edgecolor='black', zorder=3, label='persistence only' if k == 0 else None)
        bx.scatter([b_], [k], s=170, color=PURPLE, edgecolor='black', zorder=3, label='all blocking properties' if k == 0 else None)
    bx.set_yticks(range(3))
    bx.set_yticklabels(['intensity', 'duration', 'area'], fontsize=12)
    bx.set_ylim(2.7, -0.7)
    bx.set_xlim(lo, hi)
    bx.text(lo + 0.01, -0.62, 'worse than guessing\nthe average', fontsize=9, color='#a33', va='top')
    bx.set_xlabel('Share of the differences between heatwaves that is explained ($R^2$)')
    bx.set_title('Can the model tell how severe the heatwave is? Hardly', loc='left', fontsize=11)
    bx.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'fig4_severity.png', bbox_inches='tight')
    plt.close(fig)


# ============================================================================ run
def main():
    OUT.mkdir(exist_ok=True)
    t, table = counting(add_chance_level(load_table()))
    print(f'{len(t)} blocks, {t["heatwave"].sum()} with a heatwave ({t["heatwave"].mean() * 100:.0f}%)\n')
    print('PART 1. Counting: blocking and heatwaves by blocking persistence\n')
    print(table.to_string(), '\n')
    table.to_csv(OUT / 'by_lifetime.csv')

    pair, r2, pred = machine_learning(t)
    print('PART 2a. Does a blocking bring a heatwave? Pair test in % (50 = coin toss)\n')
    infos = {**INFORMATION, **CHANCE_SETS}
    print(pd.DataFrame({m: {i: round(pair[(i, m)]) for i in infos} for m in CLASSIFIERS}).to_string(), '\n')
    y, rng = t['heatwave'].values, np.random.default_rng(0)
    print('Gain from adding persistence to the chance level (pair-test points, 95% range, bootstrap over blocking):')
    for m in CLASSIFIERS:
        g = []
        for _ in range(500):
            i = rng.integers(0, len(y), len(y))
            g.append((roc_auc_score(y[i], pred[('chance + persistence', m)][i]) -
                      roc_auc_score(y[i], pred[('chance level only', m)][i])) * 100)
        print(f'  {m:20s} {np.mean(g):+.1f}  ({np.percentile(g, 2.5):+.1f} to {np.percentile(g, 97.5):+.1f})')
    print()
    one = one_property_at_a_time(t)
    print('PART 2a-bis. Which blocking property ALONE tells whether a heatwave occurs? Pair test in % (50 = guessing)\n')
    print(one.round(0).to_string(), '\n')

    print('PART 2b. If it does, what is the heatwave like? R2 (0 = guessing the average)\n')
    short = {'linear regression': 'linear', 'random forest': 'forest'}
    print(pd.DataFrame({f'{i} / {short[m]}': {HW_TARGETS[tg]: round(r2[(tg, i, m)], 2) for tg in HW_TARGETS}
                        for i in INFORMATION for m in REGRESSORS}).to_string(), '\n')

    fig_chance(t)
    fig_training(t)
    fig_occurrence(one, pair)
    fig_severity(r2)
    print(f'written {OUT / "by_lifetime.csv"} and the figures fig1_persistence, fig2_training, '
          f'fig3_occurrence, fig4_severity in {OUT}/')


if __name__ == '__main__':
    main()