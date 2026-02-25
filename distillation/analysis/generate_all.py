"""Generate all analysis plots. Run from the distillation/ directory."""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "plot_performance.py",
    "plot_cot_lengths.py",
    "plot_cot_vs_performance.py",
]

def main():
    analysis_dir = Path(__file__).parent
    for script in SCRIPTS:
        path = analysis_dir / script
        print(f"\n{'='*60}")
        print(f"Running {script}...")
        print(f"{'='*60}")
        result = subprocess.run([sys.executable, str(path)], cwd=str(analysis_dir.parent))
        if result.returncode != 0:
            print(f"  ERROR: {script} exited with code {result.returncode}")

    print(f"\n{'='*60}")
    print("All plots generated in analysis/plots/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
