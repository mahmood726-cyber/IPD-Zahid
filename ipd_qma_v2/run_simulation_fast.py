"""
Fast simulation study using percentile-difference estimator.

In the two-sample RCT setting with no covariates (which is the paper's DGP),
the unconditional quantile treatment effect (QTE) estimated via percentile
differences IS the correct estimator. Quantile regression with only a treatment
indicator produces identical results to percentile differencing.

This allows us to run 200 x 15 scenarios in minutes instead of days.

All other corrections are preserved:
  - DerSimonian-Laird random effects
  - Correct lnVR SE (factor of 2)
  - HKSJ correction
  - Proper Type I error assessment
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT_DIR, exist_ok=True)


def analyze_study_fast(control, treatment, quantiles, n_boot, rng):
    """Fast Stage 1: percentile differences + bootstrap."""
    n_c, n_t = len(control), len(treatment)
    n_q = len(quantiles)
    q_pcts = [q * 100 for q in quantiles]

    # Point estimates
    obs_q_c = np.percentile(control, q_pcts)
    obs_q_t = np.percentile(treatment, q_pcts)
    obs_effects = obs_q_t - obs_q_c

    # Bootstrap
    boot_effects = np.zeros((n_q, n_boot))
    for b in range(n_boot):
        bc = control[rng.integers(0, n_c, n_c)]
        bt = treatment[rng.integers(0, n_t, n_t)]
        boot_effects[:, b] = np.percentile(bt, q_pcts) - np.percentile(bc, q_pcts)

    se_effects = np.std(boot_effects, axis=1, ddof=1)

    # Slope (last - first quantile = Q90 - Q10)
    obs_slope = obs_effects[-1] - obs_effects[0]
    boot_slopes = boot_effects[-1, :] - boot_effects[0, :]
    se_slope = np.std(boot_slopes, ddof=1)

    # Mean difference
    mean_diff = np.mean(treatment) - np.mean(control)
    se_md = np.sqrt(np.var(control, ddof=1) / n_c + np.var(treatment, ddof=1) / n_t)

    # lnVR as log SD ratio (Nakagawa et al. 2015) with bias correction
    var_t = np.var(treatment, ddof=1)
    var_c = np.var(control, ddof=1)
    if var_c > 0 and var_t > 0:
        lnvr = (0.5 * np.log(var_t / var_c)
                + 1.0 / (2 * (n_t - 1)) - 1.0 / (2 * (n_c - 1)))
        se_lnvr = np.sqrt(1.0 / (2 * (n_t - 1)) + 1.0 / (2 * (n_c - 1)))
    else:
        lnvr = np.nan
        se_lnvr = np.nan

    return {
        'effects': obs_effects, 'se': se_effects,
        'slope': obs_slope, 'se_slope': se_slope,
        'md': mean_diff, 'se_md': se_md,
        'lnvr': lnvr, 'se_lnvr': se_lnvr,
    }


def pool_dl(est, se, use_hksj=True, conf_level=0.95):
    """DerSimonian-Laird random-effects pooling with HKSJ option."""
    est = np.asarray(est, dtype=float)
    se = np.asarray(se, dtype=float)
    k = len(est)
    se = np.maximum(se, 1e-12)

    if k == 1:
        z = est[0] / se[0]
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        return est[0], se[0], p

    w_fe = 1.0 / se**2
    mu_fe = np.sum(est * w_fe) / np.sum(w_fe)
    Q = np.sum(w_fe * (est - mu_fe)**2)

    c = np.sum(w_fe) - np.sum(w_fe**2) / np.sum(w_fe)
    tau2 = max(0.0, (Q - (k - 1)) / c)

    w_re = 1.0 / (se**2 + tau2)
    p_est = np.sum(est * w_re) / np.sum(w_re)
    p_se = np.sqrt(1.0 / np.sum(w_re))

    alpha = 1 - conf_level
    if use_hksj and k >= 3:
        q_hksj = np.sum(w_re * (est - p_est)**2) / (k - 1)
        p_se_adj = p_se * np.sqrt(max(1.0, q_hksj))
        z = p_est / p_se_adj
        p_value = 2 * (1 - stats.t.cdf(abs(z), df=k - 1))
        return p_est, p_se_adj, p_value
    else:
        z = p_est / p_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        return p_est, p_se, p_value


def simulate_and_test(K, n_range, vr, dist, seed, quantiles, n_boot, mean_shift=0.0):
    """Single simulation iteration: generate data, analyze, return p-values."""
    rng = np.random.default_rng(seed)
    scale = np.sqrt(vr)

    study_results = []
    for _ in range(K):
        n = rng.integers(n_range[0], n_range[1] + 1)

        if dist == 'normal':
            c = rng.standard_normal(n)
            t = rng.standard_normal(n) * scale + mean_shift
        elif dist == 'exponential':
            # Population-mean centering: Exp(s) has mean=s
            c = rng.exponential(1.0, n) - 1.0
            t = rng.exponential(scale, n) - scale + mean_shift
        elif dist == 'lognormal':
            # Population-mean centering: E[LN(0,0.5)] = exp(0.125)
            pop_mean_ln = np.exp(0.5**2 / 2)
            c = rng.lognormal(0, 0.5, n) - pop_mean_ln
            t = rng.lognormal(0, 0.5, n) * scale - scale * pop_mean_ln + mean_shift
        elif dist == 't5':
            c = rng.standard_t(5, n)
            t = rng.standard_t(5, n) * scale + mean_shift
        else:
            raise ValueError(dist)

        study_results.append(analyze_study_fast(c, t, quantiles, n_boot, rng))

    # Pool each quantile
    n_q = len(quantiles)
    q90_idx = len(quantiles) - 1  # Last quantile = 0.9

    est_q90 = [s['effects'][q90_idx] for s in study_results]
    se_q90 = [s['se'][q90_idx] for s in study_results]
    _, _, p_q90 = pool_dl(est_q90, se_q90)

    # Slope
    est_slope = [s['slope'] for s in study_results]
    se_slope = [s['se_slope'] for s in study_results]
    _, _, p_slope = pool_dl(est_slope, se_slope)

    # MD
    est_md = [s['md'] for s in study_results]
    se_md = [s['se_md'] for s in study_results]
    _, _, p_md = pool_dl(est_md, se_md)

    # lnVR
    est_lnvr = [s['lnvr'] for s in study_results]
    se_lnvr = [s['se_lnvr'] for s in study_results]
    valid = [(e, s) for e, s in zip(est_lnvr, se_lnvr) if np.isfinite(e)]
    if valid:
        _, _, p_lnvr = pool_dl([v[0] for v in valid], [v[1] for v in valid])
    else:
        p_lnvr = 1.0

    return p_md, p_q90, p_slope, p_lnvr


# ============================================================
# MAIN
# ============================================================

if __name__ != "__main__":
    raise SystemExit("This module should be run directly, not imported.")

N_SIM = 1000
QUANTILES = [0.1, 0.5, 0.9]
N_BOOT = 200

scenarios = []

# Type I error
for dist in ['normal', 'exponential', 'lognormal', 't5']:
    scenarios.append({'name': 'Null_' + dist, 'type': 'null', 'distribution': dist,
                      'variance_ratio': 1.0, 'K': 10, 'n_range': (80, 200)})

# Power by VR
for vr in [1.5, 2.0, 3.0, 5.0]:
    scenarios.append({'name': 'Power_VR' + str(vr), 'type': 'power', 'distribution': 'normal',
                      'variance_ratio': vr, 'K': 10, 'n_range': (80, 200)})

# Power by K
for K in [3, 5, 10, 20]:
    scenarios.append({'name': 'Power_K' + str(K), 'type': 'power', 'distribution': 'normal',
                      'variance_ratio': 2.0, 'K': K, 'n_range': (80, 200)})

# Power by N
for n_max in [50, 100, 200]:
    scenarios.append({'name': 'Power_N' + str(n_max), 'type': 'power', 'distribution': 'normal',
                      'variance_ratio': 2.0, 'K': 10, 'n_range': (max(30, n_max // 2), n_max)})

# Mixed location + scale shift
scenarios.append({'name': 'Mixed_MD0.5_VR2', 'type': 'mixed', 'distribution': 'normal',
                  'variance_ratio': 2.0, 'K': 10, 'n_range': (80, 200), 'mean_shift': 0.5})

all_results = []
t_total = time.time()

for sc_idx, sc in enumerate(scenarios):
    sys.stdout.write('[%d/%d] %s (dist=%s, VR=%s, K=%d, N=%s)\n' %
                     (sc_idx + 1, len(scenarios), sc['name'], sc['distribution'],
                      sc['variance_ratio'], sc['K'], sc['n_range']))
    sys.stdout.flush()
    t0 = time.time()

    reject = {'MD': 0, 'Q90': 0, 'Slope': 0, 'lnVR': 0}
    n_failed = 0

    for sim in range(N_SIM):
        seed = 10000 * sc_idx + sim
        try:
            p_md, p_q90, p_slope, p_lnvr = simulate_and_test(
                sc['K'], sc['n_range'], sc['variance_ratio'],
                sc['distribution'], seed, QUANTILES, N_BOOT,
                mean_shift=sc.get('mean_shift', 0.0))

            if p_md < 0.05: reject['MD'] += 1
            if p_q90 < 0.05: reject['Q90'] += 1
            if p_slope < 0.05: reject['Slope'] += 1
            if p_lnvr < 0.05: reject['lnVR'] += 1
        except Exception as e:
            n_failed += 1
            if n_failed == 1:
                sys.stdout.write('  First failure: %s\n' % e)
                sys.stdout.flush()

    elapsed = time.time() - t0
    valid_sims = N_SIM - n_failed
    if n_failed > 0:
        sys.stdout.write('  WARNING: %d/%d simulations failed (%.1f%%)\n' %
                         (n_failed, N_SIM, n_failed / N_SIM * 100))
        sys.stdout.flush()
    n = max(valid_sims, 1)
    rates = {k: v / n for k, v in reject.items()}
    mcse = {k: np.sqrt(v * (1 - v) / n) for k, v in rates.items()}

    row = {
        'Scenario': sc['name'], 'Type': sc['type'], 'Distribution': sc['distribution'],
        'VR': sc['variance_ratio'], 'K': sc['K'],
        'N_range': '%d-%d' % (sc['n_range'][0], sc['n_range'][1]),
        'N_sim': valid_sims,
        'Reject_MD': rates['MD'], 'Reject_Q90': rates['Q90'],
        'Reject_Slope': rates['Slope'], 'Reject_lnVR': rates['lnVR'],
        'MCSE_MD': mcse['MD'], 'MCSE_Q90': mcse['Q90'],
        'MCSE_Slope': mcse['Slope'], 'MCSE_lnVR': mcse['lnVR'],
        'Time_sec': elapsed,
    }
    all_results.append(row)

    sys.stdout.write('  %ds | MD=%.3f Q90=%.3f Slope=%.3f lnVR=%.3f\n' %
                     (elapsed, rates['MD'], rates['Q90'], rates['Slope'], rates['lnVR']))
    sys.stdout.flush()

sim_df = pd.DataFrame(all_results)
sim_df.to_csv(os.path.join(OUT_DIR, 'table_simulation_results.csv'), index=False)
sys.stdout.write('\nTotal: %.1f min\n' % ((time.time() - t_total) / 60))
sys.stdout.flush()

# ---- Figures ----
methods = ['Reject_MD', 'Reject_Q90', 'Reject_Slope', 'Reject_lnVR']
labels_m = ['Mean Diff', 'IPD-QMA (Q90)', 'Slope Test', 'lnVR']
colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']
markers = ['o', 's', '^', 'D']

# Fig 4: Type I error
null_df = sim_df[sim_df['Type'] == 'null']
fig4, ax4 = plt.subplots(figsize=(10, 5))
x = np.arange(len(null_df))
width = 0.2
for i, (m, l, c) in enumerate(zip(methods, labels_m, colors)):
    ax4.bar(x + i * width, null_df[m].values, width, label=l, color=c, alpha=0.8)
ax4.axhline(0.05, color='red', linestyle='--', linewidth=2, label='Nominal alpha=0.05')
ax4.axhspan(0.025, 0.075, color='red', alpha=0.1)
ax4.set_xlabel('Distribution')
ax4.set_ylabel('Rejection Rate')
ax4.set_title('Type I Error Under True Null (VR=1.0, K=10, N=80-200)')
ax4.set_xticks(x + 1.5 * width)
ax4.set_xticklabels(null_df['Distribution'].values)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')
fig4.savefig(os.path.join(OUT_DIR, 'fig4_type_i_error.png'), dpi=150, bbox_inches='tight')
plt.close('all')

# Fig 5: Power by VR
vr_df = sim_df[sim_df['Scenario'].str.startswith('Power_VR')]
fig5, ax5 = plt.subplots(figsize=(10, 6))
vrs = vr_df['VR'].values
for m, l, c, mk in zip(methods, labels_m, colors, markers):
    ax5.plot(vrs, vr_df[m].values, '-' + mk, color=c, label=l, linewidth=2, markersize=8)
ax5.axhline(0.80, color='gray', linestyle=':', alpha=0.5, label='80% power')
ax5.axhline(0.05, color='red', linestyle='--', alpha=0.3)
ax5.set_xlabel('Variance Ratio (Treatment/Control)')
ax5.set_ylabel('Power (Rejection Rate)')
ax5.set_title('Power by Variance Ratio (Normal, K=10, N=80-200)')
ax5.legend(loc='center right')
ax5.set_ylim(-0.02, 1.05)
ax5.grid(True, alpha=0.3)
fig5.savefig(os.path.join(OUT_DIR, 'fig5_power_by_vr.png'), dpi=150, bbox_inches='tight')
plt.close('all')

# Fig 6: Power by K
k_df = sim_df[sim_df['Scenario'].str.startswith('Power_K')]
fig6, ax6 = plt.subplots(figsize=(10, 6))
ks = k_df['K'].values
for m, l, c, mk in zip(methods, labels_m, colors, markers):
    ax6.plot(ks, k_df[m].values, '-' + mk, color=c, label=l, linewidth=2, markersize=8)
ax6.axhline(0.80, color='gray', linestyle=':', alpha=0.5)
ax6.set_xlabel('Number of Studies (K)')
ax6.set_ylabel('Power (Rejection Rate)')
ax6.set_title('Power by Number of Studies (Normal, VR=2.0, N=80-200)')
ax6.legend(loc='lower right')
ax6.set_ylim(-0.02, 1.05)
ax6.grid(True, alpha=0.3)
fig6.savefig(os.path.join(OUT_DIR, 'fig6_power_by_k.png'), dpi=150, bbox_inches='tight')
plt.close('all')

# Fig 7: Power by N
n_df = sim_df[sim_df['Scenario'].str.startswith('Power_N')]
fig7, ax7 = plt.subplots(figsize=(10, 6))
ns = [int(r.split('-')[1]) for r in n_df['N_range'].values]
for m, l, c, mk in zip(methods, labels_m, colors, markers):
    ax7.plot(ns, n_df[m].values, '-' + mk, color=c, label=l, linewidth=2, markersize=8)
ax7.axhline(0.80, color='gray', linestyle=':', alpha=0.5)
ax7.set_xlabel('Max Sample Size per Study')
ax7.set_ylabel('Power (Rejection Rate)')
ax7.set_title('Power by Sample Size (Normal, VR=2.0, K=10)')
ax7.legend(loc='lower right')
ax7.set_ylim(-0.02, 1.05)
ax7.grid(True, alpha=0.3)
fig7.savefig(os.path.join(OUT_DIR, 'fig7_power_by_n.png'), dpi=150, bbox_inches='tight')
plt.close('all')

sys.stdout.write('All simulation figures saved to output/\n')
sys.stdout.flush()
