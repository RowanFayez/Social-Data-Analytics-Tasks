import argparse
from pathlib import Path

from pipeline import run_task4


def resolve_repo_root(base_dir: Path) -> Path:
    return base_dir.parent.resolve()


def main():
    base_dir = Path(__file__).resolve().parent
    repo_root = resolve_repo_root(base_dir)

    parser = argparse.ArgumentParser(description="Task 4 - Model Evaluation, Optimization, Error Analysis, Deployment")
    parser.add_argument(
        "--task3_run_dir",
        type=str,
        default="",
        help="Path to a Task 3 final_data run directory (e.g., task3/final_data/run_<id>_groq). If empty, auto-detect latest.",
    )
    parser.add_argument(
        "--labeled_dataset",
        type=str,
        default="",
        help="Path to Task 3 labeled_dataset.csv. If empty, uses <task3_run_dir>/labels/labeled_dataset.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="final_data",
        help="Output directory under task4/ (default: final_data)",
    )
    parser.add_argument(
        "--text_variant",
        type=str,
        default="v1_basic",
        help="Which Task3 text variant to optimize on: v1_basic | v2_no_stop | v3_stem (or any existing text column name).",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument(
        "--max_grid",
        type=str,
        choices=["small", "big"],
        default="small",
        help="Hyperparameter grid size (small is faster).",
    )

    args = parser.parse_args()

    outputs = run_task4(
        repo_root=repo_root,
        task3_run_dir=args.task3_run_dir,
        labeled_dataset=args.labeled_dataset,
        output_dir=args.output_dir,
        text_variant=args.text_variant,
        random_seed=args.random_seed,
        test_size=args.test_size,
        max_grid=args.max_grid,
    )

    print("Task 4 completed.")
    for k, v in outputs.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
