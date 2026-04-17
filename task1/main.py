import argparse
import datetime
from utils.storage import save_dataframe
from utils.final_merge import merge_datasets_into_final_data
from analysis.aggregator import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Political Public Opinion Analyzer (task1)")
    parser.add_argument('--topn', type=int, default=10, help='Top trends to query')
    parser.add_argument('--reddit_limit', type=int, default=100, help='Reddit results per term')
    parser.add_argument('--news_per_term', type=int, default=3, help='News articles per term')
    parser.add_argument('--geo', type=str, default='GLOBAL', help='pytrends region')
    args = parser.parse_args()

    print('Running pipeline with', args)
    datasets, trends = run_pipeline(
        top_n=args.topn,
        reddit_limit=args.reddit_limit,
        news_per_term=args.news_per_term,
        geo=args.geo,
    )
    run_id = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")

    saved = {}
    saved['trends'] = save_dataframe(datasets['trends'], f"trends_{run_id}.csv", path='data')
    saved['news'] = save_dataframe(datasets['news'], f"news_{run_id}.csv", path='data')
    saved['reddit_posts_raw'] = save_dataframe(datasets['reddit_posts_raw'], f"reddit_posts_raw_{run_id}.csv", path='data')
    saved['reddit_posts_enriched'] = save_dataframe(datasets['reddit_posts_enriched'], f"reddit_posts_enriched_{run_id}.csv", path='data')
    saved['term_summary'] = save_dataframe(datasets['term_summary'], f"term_summary_{run_id}.csv", path='data')
    saved['subreddit_summary'] = save_dataframe(datasets['subreddit_summary'], f"subreddit_summary_{run_id}.csv", path='data')

    # Backward compatibility with old expected output file.
    legacy_path = save_dataframe(datasets['reddit_posts_enriched'], f"task1_results_{run_id}.csv", path='data')
    saved['legacy_task1_results'] = legacy_path

    merged_paths, merge_summary = merge_datasets_into_final_data(
        datasets=datasets,
        run_id=run_id,
        final_dir='final_data',
    )

    print("Saved output files:")
    for label, path in saved.items():
        print(f"- {label}: {path}")
    print("Merged cumulative files in final_data:")
    for label, path in merged_paths.items():
        stats = merge_summary.get(label, {})
        print(
            f"- {label}: {path} "
            f"(existing={stats.get('existing_rows', 0)}, incoming={stats.get('incoming_rows', 0)}, final={stats.get('final_rows', 0)})"
        )
    total_rows_all_datasets = sum(v.get('final_rows', 0) for v in merge_summary.values())
    total_posts_so_far = merge_summary.get('reddit_posts_enriched', {}).get('final_rows', 0)
    print(f"TOTAL rows across final_data (all datasets): {total_rows_all_datasets}")
    print(f"TOTAL reddit posts in final_data/reddit_posts_enriched.csv: {total_posts_so_far}")
    print(f"Total trend terms collected: {len(trends)}")
    print(f"Enriched Reddit posts: {len(datasets['reddit_posts_enriched'])}")


if __name__ == "__main__":
    main()
