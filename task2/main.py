import argparse
from pathlib import Path
from preprocessing.pipeline import run_task2_pipeline


def resolve_input_csv_path(input_csv):
    p = Path(input_csv)
    if p.is_absolute() and p.exists():
        return str(p)

    # 1) relative to current working directory
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)

    # 2) relative to this script directory (task2/)
    base_dir = Path(__file__).resolve().parent
    script_candidate = (base_dir / p).resolve()
    if script_candidate.exists():
        return str(script_candidate)

    # 3) fallback: relative to repo tasks root from task2/
    if str(p).replace("/", "\\").endswith("final_data\\reddit_posts_enriched.csv"):
        fallback_candidates = [
            (base_dir.parent / "final_data" / "reddit_posts_enriched.csv").resolve(),
            (base_dir.parent / "task1" / "final_data" / "reddit_posts_enriched.csv").resolve(),
        ]
        for cand in fallback_candidates:
            if cand.exists():
                return str(cand)

    return str(p)


def resolve_output_dir_path(output_dir):
    p = Path(output_dir)
    if p.is_absolute():
        return str(p)
    base_dir = Path(__file__).resolve().parent
    return str((base_dir / p).resolve())


def main():
    parser = argparse.ArgumentParser(description="Task 2 - Text Preprocessing")
    parser.add_argument(
        "--input_csv",
        type=str,
        default=r"..\task1\final_data\reddit_posts_enriched.csv",
        help="Path to Task 1 enriched Reddit CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"final_data",
        help="Directory for Task 2 outputs",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="Top K unigrams/bigrams/features to export",
    )
    args = parser.parse_args()
    resolved_input = resolve_input_csv_path(args.input_csv)
    resolved_output_dir = resolve_output_dir_path(args.output_dir)

    result = run_task2_pipeline(
        input_csv=resolved_input,
        output_dir=resolved_output_dir,
        top_k=args.top_k,
    )

    print("Task 2 completed.")
    for key, value in result.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
