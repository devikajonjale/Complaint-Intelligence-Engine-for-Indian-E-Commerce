"""Run full complaint intelligence pipeline sequentially."""

import subprocess
import sys

import warnings
warnings.filterwarnings("ignore")

SCRIPTS = [
    "01_ingest.py",
    "02_preprocess.py",
    "03_embed.py",
    "04_reduce.py",
    "05_cluster.py",
    "06_anomaly.py",
    "07_visualize.py",
    "08_severity_modeling.py",
    "09_router_modeling.py",
    "10_churn_modeling.py",
    "11_response_modeling.py",
    "12_drift_modeling.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"\n=== Running {script} ===")
        result = subprocess.run([sys.executable, script], check=False)
        if result.returncode != 0:
            raise SystemExit(f"Stopped at {script} with code {result.returncode}")
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
