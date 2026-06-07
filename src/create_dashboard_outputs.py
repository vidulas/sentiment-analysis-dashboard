from pathlib import Path
import json
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "sentiment_results.csv"
DASHBOARD_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "dashboard_outputs"

MAIN_COLUMNS = [
    "product_name",
    "product_price",
    "rating",
    "review",
    "summary",
    "full_review_text",
    "sentiment",
    "vader_sentiment",
    "vader_compound",
    "vader_positive",
    "vader_neutral",
    "vader_negative",
    "sentiment_match",
    "review_length",
    "word_count",
]

REVIEW_EXAMPLE_COLUMNS = [
    "product_name",
    "rating",
    "summary",
    "review",
    "vader_compound",
    "vader_sentiment",
]


def load_sentiment_results(file_path):
    return pd.read_csv(file_path)


def create_distribution(df, column_name, output_name):
    distribution = df[column_name].value_counts(dropna=False).reset_index()
    distribution.columns = [column_name, "count"]
    distribution["percentage"] = (distribution["count"] / len(df) * 100).round(2)

    output_path = DASHBOARD_OUTPUT_DIR / output_name
    distribution.to_csv(output_path, index=False)
    return output_path


def create_rating_distribution(df):
    distribution = df["rating"].value_counts().sort_index().reset_index()
    distribution.columns = ["rating", "count"]
    distribution["percentage"] = (distribution["count"] / len(df) * 100).round(2)

    output_path = DASHBOARD_OUTPUT_DIR / "rating_distribution.csv"
    distribution.to_csv(output_path, index=False)
    return output_path


def create_product_sentiment_summary(df):
    product_summary = (
        df.groupby("product_name")
        .agg(
            review_count=("product_name", "size"),
            average_rating=("rating", "mean"),
            average_vader_compound=("vader_compound", "mean"),
            positive_count=("vader_sentiment", lambda x: (x == "positive").sum()),
            neutral_count=("vader_sentiment", lambda x: (x == "neutral").sum()),
            negative_count=("vader_sentiment", lambda x: (x == "negative").sum()),
        )
        .reset_index()
    )

    product_summary = product_summary[product_summary["review_count"] >= 10].copy()
    product_summary["positive_percentage"] = (
        product_summary["positive_count"] / product_summary["review_count"] * 100
    ).round(2)
    product_summary["neutral_percentage"] = (
        product_summary["neutral_count"] / product_summary["review_count"] * 100
    ).round(2)
    product_summary["negative_percentage"] = (
        product_summary["negative_count"] / product_summary["review_count"] * 100
    ).round(2)
    product_summary["average_rating"] = product_summary["average_rating"].round(2)
    product_summary["average_vader_compound"] = product_summary["average_vader_compound"].round(4)

    product_summary = product_summary.sort_values(
        ["review_count", "negative_percentage"], ascending=[False, False]
    )

    output_path = DASHBOARD_OUTPUT_DIR / "product_sentiment_summary.csv"
    product_summary.to_csv(output_path, index=False)
    return output_path


def create_review_example_outputs(df):
    top_positive_path = DASHBOARD_OUTPUT_DIR / "top_positive_reviews.csv"
    top_negative_path = DASHBOARD_OUTPUT_DIR / "top_negative_reviews.csv"

    df.nlargest(25, "vader_compound")[REVIEW_EXAMPLE_COLUMNS].to_csv(top_positive_path, index=False)
    df.nsmallest(25, "vader_compound")[REVIEW_EXAMPLE_COLUMNS].to_csv(top_negative_path, index=False)

    return [top_positive_path, top_negative_path]


def create_dashboard_kpis(df):
    sentiment_percentages = df["vader_sentiment"].value_counts(normalize=True) * 100

    kpis = {
        "total_reviews": int(len(df)),
        "total_products": int(df["product_name"].nunique()),
        "average_rating": round(float(df["rating"].mean()), 2),
        "average_vader_compound": round(float(df["vader_compound"].mean()), 4),
        "positive_percentage": round(float(sentiment_percentages.get("positive", 0)), 2),
        "neutral_percentage": round(float(sentiment_percentages.get("neutral", 0)), 2),
        "negative_percentage": round(float(sentiment_percentages.get("negative", 0)), 2),
        "sentiment_match_percentage": round(float(df["sentiment_match"].mean() * 100), 2),
        "average_word_count": round(float(df["word_count"].mean()), 2),
    }

    output_path = DASHBOARD_OUTPUT_DIR / "dashboard_kpis.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(kpis, file, indent=2)

    return output_path, kpis


def create_dashboard_outputs(df):
    DASHBOARD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_paths = []

    dashboard_main_path = DASHBOARD_OUTPUT_DIR / "dashboard_main.csv"
    df[MAIN_COLUMNS].to_csv(dashboard_main_path, index=False)
    output_paths.append(dashboard_main_path)

    output_paths.append(create_distribution(df, "vader_sentiment", "sentiment_distribution.csv"))
    output_paths.append(create_rating_distribution(df))
    output_paths.append(create_product_sentiment_summary(df))
    output_paths.extend(create_review_example_outputs(df))
    output_paths.append(create_distribution(df, "sentiment_match", "sentiment_match_summary.csv"))

    kpi_path, kpis = create_dashboard_kpis(df)
    output_paths.append(kpi_path)

    return output_paths, kpis


def print_summary_report(df, kpis, output_paths):
    sentiment_distribution = df["vader_sentiment"].value_counts(dropna=False).reset_index()
    sentiment_distribution.columns = ["vader_sentiment", "count"]
    sentiment_distribution["percentage"] = (
        sentiment_distribution["count"] / len(df) * 100
    ).round(2)

    print("Dashboard output summary")
    print()
    print(f"Total reviews: {kpis['total_reviews']:,}")
    print(f"Total products: {kpis['total_products']:,}")
    print(f"Average rating: {kpis['average_rating']:.2f}")
    print()
    print("VADER sentiment distribution:")
    print(sentiment_distribution.to_string(index=False))
    print()
    print(f"Sentiment match percentage: {kpis['sentiment_match_percentage']:.2f}%")
    print(f"Number of output files created: {len(output_paths)}")
    print()
    print(f"Dashboard outputs saved to: {DASHBOARD_OUTPUT_DIR}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df = load_sentiment_results(SENTIMENT_RESULTS_PATH)
    output_paths, kpis = create_dashboard_outputs(df)
    print_summary_report(df, kpis, output_paths)


if __name__ == "__main__":
    main()
