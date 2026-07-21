import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mplcm
import matplotlib.lines as mlines
import pandas as pd
from scipy.stats import pearsonr, spearmanr

#made with claude code

# =============================================================================
# Visual encoding scheme (consistent across all four group plots)
#   Marker shape  → subgroup identity (see legend per plot)
#   Colour        → varied quantity: Λ (A), total mass (B), q (C), spin (D)
#   Marker edge   → negative_modes:  thick black edge (lw=1.5) = mirror modes
#                                     no edge (lw=0.5)          = standard fit
# =============================================================================

_DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sims")
_MARKER_POOL = ['o', 's', '^', 'D', 'v', 'P']
_CMAP        = 'viridis'
_S           = 80


def _load(filename):
    try:
        df = pd.read_csv(os.path.join(_DATA_DIR, filename))
        df['mass_ratio_remnant'] = df['mass_remnant'] / df['total_mass']
        return df
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not load {filename}: {e}")


def _corr_pair(x, y):
    """Pearson r and Spearman rho (with p-values) between x and y."""
    r, r_p = pearsonr(x, y)
    rho, rho_p = spearmanr(x, y)
    return dict(r=r, r_p=r_p, rho=rho, rho_p=rho_p, n=len(x))


def _print_corr_table(rows):
    header = f"{'Group':<7}{'Pair':<30}{'n':>4}{'Pearson r':>11}{'p (r)':>9}{'Spearman rho':>14}{'p (rho)':>10}"
    print(header)
    print('-' * len(header))
    for group, label, c in rows:
        print(f"{group:<7}{label:<30}{c['n']:>4}{c['r']:>11.2f}{c['r_p']:>9.2f}"
              f"{c['rho']:>14.2f}{c['rho_p']:>10.2f}")
    print()


def _compute_b1_correlation():
    """Controlled correlation within subgroup B1 (BLQ, q=1): total mass is the
    only quantity varied, at fixed EOS and mass ratio, so this isn't confounded
    by pooling across different EOS or masses the way a whole-group correlation
    would be. Prints Pearson r and Spearman rho for M vs Mf/M and M vs chi."""
    df = _load('group_B.csv')
    b1 = df[df['subgroup'] == 'B1']

    m_mf = _corr_pair(b1['total_mass'].values, b1['mass_ratio_remnant'].values)
    m_chi = _corr_pair(b1['total_mass'].values, b1['spin_remnant'].values)

    _print_corr_table([
        ('B1', 'M vs Mf/M', m_mf),
        ('B1', 'M vs chi', m_chi),
    ])

    return dict(m_mf=m_mf, m_chi=m_chi)


def _plot_sg(ax, sub, x_vals, c_vals, norm, marker, cmap=_CMAP):
    """Scatter one subgroup: x = remnant spin, y = M_rem/M_tot, colour = varied quantity."""
    ec = ['black' if v == 'yes' else 'none' for v in sub['negative_modes']]
    lw = [1.5    if v == 'yes' else 0.5    for v in sub['negative_modes']]
    return ax.scatter(
        x_vals, sub['mass_ratio_remnant'].values,
        c=c_vals, cmap=cmap, norm=norm,
        marker=marker, s=_S, edgecolors=ec, linewidths=lw, zorder=3
    )


def _legend_handles(df, subgroups):
    handles = []
    for i, sg in enumerate(subgroups):
        fp = df.loc[df['subgroup'] == sg, 'fixed_params'].iloc[0]
        handles.append(mlines.Line2D(
            [], [], marker=_MARKER_POOL[i % len(_MARKER_POOL)],
            color='gray', linestyle='None', markersize=8,
            label=f'{sg}  ({fp})'
        ))
    handles.append(mlines.Line2D(
        [], [], marker='o', color='gray', linestyle='None', markersize=8,
        markeredgecolor='black', markeredgewidth=1.5, label='mirror modes fit'
    ))
    handles.append(mlines.Line2D(
        [], [], marker='o', color='gray', linestyle='None', markersize=8,
        markeredgecolor='none', label='standard fit'
    ))
    return handles


def _finish(fig, ax, df, subgroups, norm, cbar_label, cmap=_CMAP,
            cbar_ticks=None, cbar_ticklabels=None):
    ax.set_xlabel(r'$\chi$', fontsize='large')
    ax.set_ylabel(r'$M_\mathrm{f} / M$', fontsize='large')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(handles=_legend_handles(df, subgroups), fontsize='small')
    sm = mplcm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label(cbar_label, fontsize='large')
    fig.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────

def plot_group_a():
    """Group A: EoS / tidal deformability Λ varied; subgroups at fixed total mass."""
    df = _load('group_A.csv')
    subgroups = sorted(df['subgroup'].unique())
    norm = mcolors.Normalize(vmin=df['Lambda'].min(), vmax=df['Lambda'].max())

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, sg in enumerate(subgroups):
        sub = df[df['subgroup'] == sg]
        _plot_sg(ax, sub, sub['spin_remnant'].values, sub['Lambda'].values,
                 norm, _MARKER_POOL[i % len(_MARKER_POOL)])

    _finish(fig, ax, df, subgroups, norm, r'$\tilde{\Lambda}$')
    plt.show()


def plot_group_b():
    """Group B: total mass varied; subgroups at fixed EoS."""
    df = _load('group_B.csv')
    subgroups = sorted(df['subgroup'].unique())
    norm = mcolors.Normalize(vmin=df['total_mass'].min(), vmax=df['total_mass'].max())

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, sg in enumerate(subgroups):
        sub = df[df['subgroup'] == sg]
        _plot_sg(ax, sub, sub['spin_remnant'].values, sub['total_mass'].values,
                 norm, _MARKER_POOL[i % len(_MARKER_POOL)])

    _finish(fig, ax, df, subgroups, norm, r'Total mass $[M_\odot]$')
    plt.show()


def plot_group_c():
    """Group C: mass ratio q varied; subgroups at fixed EoS."""
    df = _load('group_C.csv')
    subgroups = sorted(df['subgroup'].unique())
    norm = mcolors.Normalize(vmin=df['mass_ratio'].min(), vmax=df['mass_ratio'].max())

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, sg in enumerate(subgroups):
        sub = df[df['subgroup'] == sg]
        _plot_sg(ax, sub, sub['spin_remnant'].values, sub['mass_ratio'].values,
                 norm, _MARKER_POOL[i % len(_MARKER_POOL)])

    _finish(fig, ax, df, subgroups, norm, r'Mass ratio $q$')
    plt.show()



def plot_all():
    """All simulations across all groups on a single mass vs spin scatter."""
    files = ['group_A.csv', 'group_B.csv', 'group_C.csv', 'group_D.csv']
    df = pd.concat([pd.read_csv(os.path.join(_DATA_DIR, f)) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset='code')

    collapse_styles = {
        'prompt':     dict(color='steelblue',  label='Prompt'),
        'delayed':    dict(color='firebrick',   label='Delayed'),
    }

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for ctype, style in collapse_styles.items():
        sub = df[df['collapse_type'] == ctype]
        ax.scatter(sub['spin_remnant'], sub['mass_remnant'],
                   s=_S, zorder=3, **style)

    ax.set_xlabel(r'$\chi$', fontsize='large')
    ax.set_ylabel(r'$M_\mathrm{f}\ [M_\odot]$', fontsize='large')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize='large')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    _compute_b1_correlation()
    plot_group_a()
    plot_group_b()
    plot_group_c()
    plot_all()
