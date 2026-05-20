"""Run the WFJSSP-GA demo for three GA feature variants.

Variants:
- no RL, no surrogate
- no RL, with surrogate
- with RL and surrogate

The script imports ``wfjssp_ga_demo`` and reuses its instance selection, seeds,
limits, uncertainty settings, W&B settings and output writer. Results are written
to separate subdirectories and combined into aggregate CSV files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


VARIANTS = [
    # {
    #     "name": "no_rl_no_surrogate",
    #     "enable_rl_mutation_control": False,
    #     "use_surrogate_evaluation": False,
    #     # "do_restart": True,
    # },
    # {
    #     "name": "no_rl_with_surrogate",
    #     "enable_rl_mutation_control": False,
    #     "use_surrogate_evaluation": True,
    # },
    {
        "name": "with_rl_with_surrogate",
        "enable_rl_mutation_control": True,
        "use_surrogate_evaluation": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run wfjssp_ga_demo.py for RL/surrogate comparison variants.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/wfjssp_ga_variant_comparison"),
        help="Base directory for all variant outputs.",
    )
    parser.add_argument(
        "--wandb",
        choices=["demo", "on", "off"],
        default="demo",
        help="Use demo setting, force W&B on, or force W&B off.",
    )
    return parser.parse_args()


def apply_variant(demo_module, variant: dict[str, Any], results_dir: Path) -> None:
    demo = demo_module
    demo.EXPERIMENT_NAME = variant["name"]
    demo.RESULTS_DIR = results_dir
    demo.ENABLE_RL_MUTATION_CONTROL = bool(variant["enable_rl_mutation_control"])
    demo.GA_CONFIG["enable_rl_mutation_control"] = bool(variant["enable_rl_mutation_control"])
    demo.GA_CONFIG["use_surrogate_evaluation"] = bool(variant["use_surrogate_evaluation"])


def read_variant_csv(path: Path, variant: dict[str, Any], csv_name: str) -> pd.DataFrame:
    import pandas as pd

    csv_path = path / csv_name
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df.insert(0, "variant", variant["name"])
    df.insert(1, "variant_rl_enabled", bool(variant["enable_rl_mutation_control"]))
    df.insert(2, "variant_surrogate_enabled", bool(variant["use_surrogate_evaluation"]))
    return df


def write_aggregate_outputs(base_results_dir: Path, variant_outputs: list[tuple[dict[str, Any], Path]]) -> None:
    import pandas as pd

    aggregate_specs = [
        ("run_results.csv", "all_run_results.csv"),
        ("instance_summary.csv", "all_instance_summary.csv"),
        ("ranking_results.csv", "all_ranking_results.csv"),
    ]

    for source_name, target_name in aggregate_specs:
        frames = [
            read_variant_csv(path, variant, source_name)
            for variant, path in variant_outputs
        ]
        frames = [frame for frame in frames if not frame.empty]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(base_results_dir / target_name, index=False)


def main() -> None:
    args = parse_args()
    import wfjssp_ga_demo as demo

    base_results_dir = args.results_dir
    base_results_dir.mkdir(parents=True, exist_ok=True)

    if args.wandb == "on":
        demo.USE_WANDB = True
    elif args.wandb == "off":
        demo.USE_WANDB = False

    variant_outputs = []
    for variant in VARIANTS:
        variant_dir = base_results_dir / variant["name"]
        apply_variant(demo, variant, variant_dir)
        print(
            f"Running {variant['name']} "
            f"(RL={variant['enable_rl_mutation_control']}, "
            f"surrogate={variant['use_surrogate_evaluation']})",
            flush=True,
        )
        demo.main()
        variant_outputs.append((variant, variant_dir))

    write_aggregate_outputs(base_results_dir, variant_outputs)
    print(f"Finished variant comparison. Results: {base_results_dir}", flush=True)


if __name__ == "__main__":
    main()
