"""
Build the canonical point set (positives + negatives) that feature
extraction gets run against. This is the handoff contract: whatever
you do locally to pull DEM/OSM/drain data, it operates on THIS file
and returns one row of features per point_id.

Usage:
    python3 src/build_point_set.py --min-dist 300

Output:
    data/processed/points_for_feature_extraction.csv
    columns: point_id, lon, lat, label, name (positives only), min_dist_threshold_m (negatives only)
"""
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-dist", type=int, default=300, choices=[200, 300, 500],
        help="which negative-sampling threshold to use as the primary run "
             "(others stay available in data/processed/ for the sensitivity check)",
    )
    args = parser.parse_args()

    positives = pd.read_csv("data/processed/positives.csv")[["lon", "lat", "label", "name"]]
    negatives = pd.read_csv(f"data/processed/negatives_mindist{args.min_dist}m.csv")[
        ["lon", "lat", "label"]
    ]

    positives["point_id"] = ["pos_" + str(i) for i in range(len(positives))]
    negatives["point_id"] = ["neg_" + str(i) for i in range(len(negatives))]
    negatives["name"] = None

    combined = pd.concat([positives, negatives], ignore_index=True)
    combined = combined[["point_id", "lon", "lat", "label", "name"]]

    out_path = "data/processed/points_for_feature_extraction.csv"
    combined.to_csv(out_path, index=False)
    print(f"wrote {len(combined)} points ({len(positives)} positive, {len(negatives)} negative)")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
