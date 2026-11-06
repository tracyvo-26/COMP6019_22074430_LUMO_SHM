
"""
Checks whether each damaged state still differs from Healthy sessions recorded at about the same time, to see if the signal is damage or only the recording period.

Needs: modal_feature_screen.py and ssi_modal_frequencies.csv.
"""
from __future__ import annotations
import datetime
import os
import sys
import warnings
 
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
 
import modal_feature_screen as mfs
 
warnings.filterwarnings("ignore")
 
W_DAYS = next((int(a) for a in sys.argv[1:] if a.isdigit()), 30)
FEATURES = ["f_B1x", "f_B1y", "f_B2x", "f_B2y"]
MIN_GROUP = 5
OUT_CSV = os.path.join(mfs.HERE, "time_matched_check.csv")
 
 
def session_days(sessions):
    """Return the recording day and time stamp of each session from its file name."""
    t = []
    for s in sessions:
        m = mfs._RE.search(str(s))
        if not m:
            raise SystemExit(f"session name {s} is not SHMTS_YYYYMMDDHHMM")
        t.append(datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                   int(m.group(4)), int(m.group(5))))
    t = pd.Series(t)
    return ((t - t.min()).dt.total_seconds() / 86400.0).to_numpy(), t
 
 
def make_fit(x, T, mask):
    """Fit a temperature polynomial on the masked rows and return a function that evaluates it."""
    mu, sd = T[mask].mean(), T[mask].std() + 1e-9
    c = np.polyfit((T[mask] - mu) / sd, x[mask], mfs.COMP_DEGREE)
    return lambda t: np.polyval(c, (t - mu) / sd)
 
 
def residuals(x, T, healthy, weeks):
    """Residuals of one feature after removing the Healthy temperature trend, with Healthy sessions fitted without their own week."""
    res = np.full(len(x), np.nan)
    ok_h = healthy & np.isfinite(x)
    if ok_h.sum() < mfs.MIN_HEALTHY_FIT:
        return res
    f_all = make_fit(x, T, ok_h)
    for i in np.where(np.isfinite(x))[0]:
        if healthy[i]:
            mask = ok_h & (weeks != weeks[i])
            if mask.sum() < mfs.MIN_HEALTHY_FIT:
                continue
            res[i] = x[i] - make_fit(x, T, mask)(T[i])
        else:
            res[i] = x[i] - f_all(T[i])
    return res
 
 
def compare(d, h):
    """Damaged residuals d against Healthy residuals h: difference of medians, Cliff's delta, Mann-Whitney p."""
    d, h = d[np.isfinite(d)], h[np.isfinite(h)]
    if len(d) < MIN_GROUP or len(h) < MIN_GROUP:
        return dict(n_damage=len(d), n_healthy=len(h), diff_median=np.nan, cliffs_delta=np.nan, p=np.nan)
    u = mannwhitneyu(d, h, alternative="two-sided")
    return dict(n_damage=len(d), n_healthy=len(h), diff_median=float(np.median(d) - np.median(h)),
                cliffs_delta=float(2 * u.statistic / (len(d) * len(h)) - 1), p=float(u.pvalue))
 
 
def main():
    """Compare each damaged state with time-matched Healthy sessions and save the table."""
    df, base = mfs.load_table()
    T = df["temperature_degC"].to_numpy(float)
    label = df["damage_label"].to_numpy()
    healthy = label == "Healthy"
    days, stamps = session_days(df["session"])
    weeks = mfs.week_groups(df["session"])
    classes = [c for c in sorted(set(label)) if c != "Healthy"]
    month = stamps.dt.strftime("%Y-%m").to_numpy()
    pd.set_option("display.width", 220, "display.max_columns", 40)
    print(f"\ncompensation: polynomial degree {mfs.COMP_DEGREE}; window +-{W_DAYS} days around each class's recording window\n")
 
    print("=" * 100 + "\n1. TIMELINE\n" + "=" * 100)
    tl = []
    for c in ["Healthy"] + classes:
        i = label == c
        tl.append(dict(cls=c, n=int(i.sum()), first=stamps[i].min().date(), last=stamps[i].max().date(),
                       temp_min=T[i].min(), temp_max=T[i].max()))
    print(pd.DataFrame(tl).round(1).to_string(index=False))
    print("\nsessions per month (rows) and class (columns):")
    print(pd.crosstab(month, label).to_string())
 
    res = {f: residuals(base[f].to_numpy(float), T, healthy, weeks) for f in FEATURES}
    sd_h = {f: np.nanstd(res[f][healthy]) for f in FEATURES}
    med_f = {f: np.nanmedian(base[f].to_numpy(float)) for f in FEATURES}
 
    print("\n" + "=" * 100 + "\n2. DO THE HEALTHY RESIDUALS DRIFT WITH TIME?  (Healthy sessions only)\n" + "=" * 100)
    rows = []
    for f in FEATURES:
        ok = healthy & np.isfinite(res[f])
        rho, p = spearmanr(days[ok], res[f][ok])
        rows.append(dict(feature=f, n=int(ok.sum()), spearman_with_date=rho, p=p, residual_sd_Hz=sd_h[f]))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    mt = pd.DataFrame({"month": month[healthy]}).assign(**{f: res[f][healthy] for f in FEATURES})
    print("\nmedian Healthy residual (Hz) per month  [n in brackets = sessions with f_B2y]:")
    g = mt.groupby("month")
    out = g[FEATURES].median()
    out["n_B2y"] = g["f_B2y"].count()
    print(out.round(4).to_string())
 
    print("\n" + "=" * 100 + f"\n3. EACH DAMAGE CLASS AGAINST HEALTHY SESSIONS RECORDED WITHIN +-{W_DAYS} DAYS\n" + "=" * 100)
    out_rows = []
    for c in classes:
        idx = np.where(label == c)[0]
        t0, t1 = days[idx].min(), days[idx].max()
        near = healthy & (days >= t0 - W_DAYS) & (days <= t1 + W_DAYS)
        out_of_range = np.mean((T[idx] < T[healthy].min()) | (T[idx] > T[healthy].max()))
        for f in FEATURES:
            r_near = compare(res[f][idx], res[f][near])
            r_all = compare(res[f][idx], res[f][healthy])
            row = dict(cls=c, feature=f, window_days=W_DAYS, n_healthy_near_sessions=int(near.sum()),
                       share_damage_outside_Healthy_temp_range=out_of_range,
                       diff_Hz=r_near["diff_median"], diff_percent=100 * r_near["diff_median"] / med_f[f],
                       diff_in_healthy_sd=r_near["diff_median"] / sd_h[f] if sd_h[f] > 0 else np.nan,
                       cliffs_delta=r_near["cliffs_delta"], p=r_near["p"],
                       n_damage=r_near["n_damage"], n_healthy_near=r_near["n_healthy"],
                       diff_Hz_vs_all_healthy=r_all["diff_median"], cliffs_delta_vs_all_healthy=r_all["cliffs_delta"],
                       p_vs_all_healthy=r_all["p"])
            out_rows.append(row)
    res_df = pd.DataFrame(out_rows)
    res_df.to_csv(OUT_CSV, index=False)
    for c in classes:
        sub = res_df[res_df.cls == c]
        n_near = int(sub.n_healthy_near_sessions.iloc[0])
        print(f"\n{c}: {int(sub.n_damage.max())} damaged sessions; {n_near} Healthy sessions within +-{W_DAYS} days of its "
              f"window; {100 * sub.share_damage_outside_Healthy_temp_range.iloc[0]:.0f}% of its sessions lie outside the Healthy "
              f"temperature range")
        if n_near < MIN_GROUP:
            print("   too few time-matched Healthy sessions: the period cannot be separated from the damage for this class")
            continue
        show = ["feature", "n_damage", "n_healthy_near", "diff_Hz", "diff_percent", "diff_in_healthy_sd",
                "cliffs_delta", "p", "cliffs_delta_vs_all_healthy", "p_vs_all_healthy"]
        print(sub[show].round(4).to_string(index=False))
    print(f"\nSaved {OUT_CSV}")
 
 
if __name__ == "__main__":
    main()
 