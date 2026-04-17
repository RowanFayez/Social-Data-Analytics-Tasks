import argparse
import os
from pathlib import Path

from sentiment.pipeline import run_task3


def resolve_path(p: str, base_dir: Path) -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def resolve_input_csv(input_csv: str, base_dir: Path) -> str:
    p = Path(input_csv)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            (Path.cwd() / p).resolve(),
            (base_dir / p).resolve(),
            (base_dir.parent / "task2" / "final_data" / "processed" / "preprocessed_posts.csv").resolve(),
        ])
    for c in candidates:
        if c.exists():
            return str(c)
    return str((base_dir / p).resolve())


def main():
    parser = argparse.ArgumentParser(description="Task 3 Full Sentiment Pipeline")
    parser.add_argument("--input_csv", type=str, default=r"..\task2\final_data\processed\preprocessed_posts.csv")
    parser.add_argument("--output_dir", type=str, default="final_data")
    parser.add_argument("--positive_words", type=str, default=r"words\positive-words.txt")
    parser.add_argument("--negative_words", type=str, default=r"words\negative-words.txt")
    parser.add_argument("--sample_size", type=int, default=200, help="Use 100 for section or 200 for home")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--gemini_api_key", type=str, default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument("--gemini_model", type=str, default="gemini-1.5-flash")
    parser.add_argument("--glove_path", type=str, default=os.getenv("GLOVE_PATH", ""))
    parser.add_argument("--neutral_margin", type=float, default=0.05)
    parser.add_argument("--bow_max_features", type=int, default=1200)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    input_csv = resolve_input_csv(args.input_csv, base_dir)
    output_dir = resolve_path(args.output_dir, base_dir)
    positive_words = resolve_path(args.positive_words, base_dir)
    negative_words = resolve_path(args.negative_words, base_dir)
    glove_path = resolve_path(args.glove_path, base_dir) if args.glove_path else ""

    outputs = run_task3(
        input_csv=input_csv,
        output_dir=output_dir,
        positive_words_path=positive_words,
        negative_words_path=negative_words,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
        gemini_api_key=args.gemini_api_key,
        gemini_model=args.gemini_model,
        glove_path=glove_path,
        neutral_margin=args.neutral_margin,
        bow_max_features=args.bow_max_features,
    )

    print("Task 3 completed.")
    for k, v in outputs.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
