import argparse
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

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
    base_dir = Path(__file__).resolve().parent
    if load_dotenv is not None:
        # Try a few common locations without overriding explicitly-set env vars.
        candidates = [
            base_dir / ".env",
            base_dir.parent / ".env",
            base_dir.parent / "task1" / ".env",
        ]
        for p in candidates:
            if p.exists():
                load_dotenv(dotenv_path=p, override=False)

    parser = argparse.ArgumentParser(description="Task 3 Full Sentiment Pipeline")
    parser.add_argument("--input_csv", type=str, default=r"..\task2\final_data\processed\preprocessed_posts.csv")
    parser.add_argument("--output_dir", type=str, default="final_data")
    parser.add_argument("--positive_words", type=str, default=r"words\positive-words.txt")
    parser.add_argument("--negative_words", type=str, default=r"words\negative-words.txt")
    parser.add_argument("--sample_size", type=int, default=200, help="Use 100 for section or 200 for home")
    parser.add_argument("--random_seed", type=int, default=42)
    default_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not default_provider:
        default_provider = "groq" if os.getenv("GROQ_API_KEY", "").strip() else "gemini"
    parser.add_argument(
        "--llm_provider",
        type=str,
        choices=["groq", "gemini", "none"],
        default=default_provider,
        help="LLM provider for labeling (groq, gemini, or none to disable).",
    )
    parser.add_argument("--gemini_api_key", type=str, default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument("--gemini_model", type=str, default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--groq_api_key", type=str, default=os.getenv("GROQ_API_KEY", ""))
    parser.add_argument("--groq_model", type=str, default=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    parser.add_argument(
        "--strict_llm",
        "--strict_gemini",
        action="store_true",
        dest="strict_llm",
        help="Fail the run if LLM labeling errors instead of falling back to neutral.",
    )
    parser.add_argument("--glove_path", type=str, default=os.getenv("GLOVE_PATH", ""))
    parser.add_argument("--neutral_margin", type=float, default=0.05)
    parser.add_argument("--bow_max_features", type=int, default=1200)
    args = parser.parse_args()

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
        llm_provider=args.llm_provider,
        gemini_api_key=args.gemini_api_key,
        gemini_model=args.gemini_model,
        groq_api_key=args.groq_api_key,
        groq_model=args.groq_model,
        strict_llm=args.strict_llm,
        glove_path=glove_path,
        neutral_margin=args.neutral_margin,
        bow_max_features=args.bow_max_features,
    )

    print("Task 3 completed.")
    for k, v in outputs.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
