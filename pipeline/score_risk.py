"""Combine the terrain-hydrology features into a flood susceptibility score per road.

WHY THIS IS NOT A TRAINED CLASSIFIER
------------------------------------
A supervised model needs labels: places where water actually stood, with dates. For
Bengaluru no such public dataset exists. BBMP publishes a static list of flood-prone
spots with no dates; our RTIs and data requests for the complaint logs have not
produced anything usable.

Training a classifier anyway - by inventing negatives, or by treating a static
flood-prone list as if it were event data - produces a model whose reported accuracy
measures nothing. We have seen exactly that failure in an earlier trial repo for this
project, where synthetic features yielded a headline accuracy that was pure artefact.

So the susceptibility score here is a *weighted-overlay index* over physically
meaningful terrain variables, with weights set a priori from the urban-pluvial
literature (the standard AHP approach used when event labels are unavailable). It is
transparent, every term has a physical meaning, and it makes no claim to a fitted
accuracy. `validate.py` checks it against an independent set of documented chronic
flooding locations - as a sanity check, not a performance metric.

If labelled event data ever arrives, `fit_supervised.py` is the slot to drop it into;
the feature matrix is already built.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as C

# ---------------------------------------------------------------------------
# Weights. Set a priori from published AHP studies of urban pluvial flood
# susceptibility; NOT fitted to any Bengaluru outcome data. They must sum to 1.
# Ponding depth and height-above-drainage carry the most weight because urban
# pluvial flooding is, first and last, about local topographic lows.
# ---------------------------------------------------------------------------
WEIGHTS = {
    "hand": 0.25,        # height above nearest drainage  (inverted)
    "sink_depth": 0.22,  # depth of closed depression
    "twi": 0.18,         # topographic wetness index
    "slope": 0.15,       # slope                           (inverted)
    "dist_drain": 0.12,  # distance to mapped drain/lake   (inverted)
    "flowacc": 0.08,     # upslope contributing area       (log)
}

# Physical saturation points for the inverted/decay terms, in metres.
HAND_SCALE_M = 8.0        # at 8 m above drainage, pluvial ponding risk is ~nil
SLOPE_FLAT_DEG = 3.0      # above 3 deg, water runs off rather than stands
DRAIN_NEAR_M = 150.0      # within 150 m of a drain/lake, backflow/overflow matters
SINK_CAP_M = 1.5          # a 1.5 m pond is already impassable; deeper adds nothing

# Rainfall trigger above which we treat a high-susceptibility road as impassable.
IMPASSABLE_RISK = 0.60


def robust_norm(x, lo_pct=2, hi_pct=98):
    """Scale to 0-1 using percentile bounds, so single outliers cannot dominate."""
    x = np.asarray(x, dtype="float64")
    good = np.isfinite(x)
    if good.sum() == 0:
        return np.zeros_like(x)
    lo, hi = np.percentile(x[good], [lo_pct, hi_pct])
    if hi <= lo:
        return np.where(good, 0.5, 0.0)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def susceptibility(df: pd.DataFrame) -> pd.DataFrame:
    """Per-edge static susceptibility, 0-1. Higher = more prone to standing water."""
    out = pd.DataFrame(index=df.index)

    # Each term is oriented so that 1 = worst. Physical saturations first, then a
    # robust percentile stretch so the six terms are comparable before weighting.
    out["hand"] = robust_norm(np.exp(-df["hand"].to_numpy() / HAND_SCALE_M))
    out["sink_depth"] = robust_norm(np.clip(df["sink_depth"].to_numpy(), 0, SINK_CAP_M))
    out["twi"] = robust_norm(df["twi"].to_numpy())
    out["slope"] = robust_norm(1.0 - np.clip(df["slope"].to_numpy() / SLOPE_FLAT_DEG, 0, 1))
    out["dist_drain"] = robust_norm(np.exp(-df["dist_drain"].to_numpy() / DRAIN_NEAR_M))
    out["flowacc"] = robust_norm(np.log1p(np.clip(df["flowacc"].to_numpy(), 0, None)))

    score = sum(out[k] * w for k, w in WEIGHTS.items())
    # Final stretch so the index spans 0-1 across the actual road network.
    return out.assign(susceptibility=robust_norm(score, 1, 99))


def sensitivity(df, comp, n=200, jitter=0.30, seed=0):
    """How stable is the ranking if the a priori weights are wrong?

    Perturbs each weight by up to +-30% and reports the rank correlation with the
    baseline. Weights chosen by judgement deserve to have their fragility stated.
    """
    rng = np.random.default_rng(seed)
    base = df["susceptibility"].to_numpy()
    base_rank = pd.Series(base).rank().to_numpy()
    keys = list(WEIGHTS)
    cors, top_overlap = [], []
    top_n = max(1, len(base) // 20)                      # top 5%
    base_top = set(np.argsort(-base)[:top_n])
    for _ in range(n):
        w = np.array([WEIGHTS[k] for k in keys])
        w = w * (1 + rng.uniform(-jitter, jitter, w.size))
        w = w / w.sum()
        s = sum(comp[k].to_numpy() * wi for k, wi in zip(keys, w))
        cors.append(np.corrcoef(base_rank, pd.Series(s).rank().to_numpy())[0, 1])
        top_overlap.append(len(base_top & set(np.argsort(-s)[:top_n])) / top_n)
    return float(np.mean(cors)), float(np.min(cors)), float(np.mean(top_overlap))


def main():
    df = pd.read_parquet(C.PROC / "edges.parquet")
    print(f"  {len(df):,} edges")

    comp = susceptibility(df)
    df["susceptibility"] = comp["susceptibility"]

    print("\n  component correlations with the final index:")
    for k in WEIGHTS:
        r = np.corrcoef(comp[k], comp["susceptibility"])[0, 1]
        print(f"    {k:<12} w={WEIGHTS[k]:.2f}   r={r:+.3f}")

    mean_r, min_r, overlap = sensitivity(df, comp)
    print(f"\n  weight sensitivity (200 draws, +-30% on each weight):")
    print(f"    rank correlation with baseline : mean {mean_r:.3f}, worst {min_r:.3f}")
    print(f"    top-5% set retained            : {overlap:.1%}")

    q = df["susceptibility"].quantile([0.5, 0.8, 0.9, 0.95, 0.99]).round(3)
    print(f"\n  susceptibility distribution:\n{q.to_string()}")

    # Band the index for display. Quintiles of the road network.
    df["risk_band"] = pd.cut(
        df["susceptibility"],
        bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
        labels=["very low", "low", "moderate", "high", "very high"],
    )
    print(f"\n  band counts:\n{df.risk_band.value_counts().sort_index().to_string()}")

    df.to_parquet(C.PROC / "edges_scored.parquet", index=False)
    meta = {
        "weights": WEIGHTS,
        "scales": {
            "hand_scale_m": HAND_SCALE_M, "slope_flat_deg": SLOPE_FLAT_DEG,
            "drain_near_m": DRAIN_NEAR_M, "sink_cap_m": SINK_CAP_M,
        },
        "impassable_risk": IMPASSABLE_RISK,
        "sensitivity": {"mean_rank_corr": mean_r, "worst_rank_corr": min_r,
                        "top5pct_retained": overlap},
        "n_edges": int(len(df)),
        "note": "Weighted-overlay index with a priori weights. Not a fitted model; "
                "no accuracy is claimed. See module docstring.",
    }
    (C.PROC / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n  wrote edges_scored.parquet and model_meta.json")


if __name__ == "__main__":
    main()
