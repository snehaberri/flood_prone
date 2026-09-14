"""Run the whole pipeline in dependency order.

Every stage caches its downloads, so re-running is cheap and an interrupted run
resumes rather than starting over. Pass --from <stage> to skip ahead.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES = [
    ("dem",       "pipeline/fetch_dem.py",       "terrain"),
    ("osm",       "pipeline/fetch_osm.py",       "roads, drains, lakes"),
    ("hydrology", "pipeline/build_hydrology.py", "sinks, slope, flow, TWI, HAND"),
    ("network",   "pipeline/build_network.py",   "routable graph + features"),
    ("rainfall",  "pipeline/fetch_rainfall.py",  "hourly rain -> trigger"),
    ("score",     "pipeline/score_risk.py",      "susceptibility index"),
    ("checkset",  "pipeline/build_checkset.py",  "independent check locations"),
    ("validate",  "pipeline/validate.py",        "sanity checks"),
]


def main():
    start_at = 0
    if "--from" in sys.argv:
        want = sys.argv[sys.argv.index("--from") + 1]
        names = [s[0] for s in STAGES]
        if want not in names:
            raise SystemExit(f"unknown stage {want!r}; choose from {', '.join(names)}")
        start_at = names.index(want)

    t0 = time.time()
    for i, (name, script, desc) in enumerate(STAGES):
        if i < start_at:
            print(f"  skip  {name}")
            continue
        print(f"\n{'=' * 66}\n  [{i + 1}/{len(STAGES)}] {name} - {desc}\n{'=' * 66}")
        t = time.time()
        r = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
        if r.returncode != 0:
            raise SystemExit(f"\n  stage {name!r} failed with exit code {r.returncode}")
        print(f"  ({time.time() - t:.0f}s)")

    print(f"\n{'=' * 66}")
    print(f"  pipeline complete in {(time.time() - t0) / 60:.1f} min")
    print(f"  now run:  python serve.py")


if __name__ == "__main__":
    main()
