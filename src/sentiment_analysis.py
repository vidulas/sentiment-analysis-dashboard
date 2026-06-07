from pathlib import Path
import sys

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_reviews.csv"
SENTIMENT_RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "sentiment_results.csv"


def load_cleaned_data(file_path):
    return pd.read_csv(file_path)


def label_vader_sentiment(compound_score):
    if compound_score >= 0.05:
        return "positive"
    if compound_score <= -0.05:
        return "negative"
    return "neutral"


def score_reviews(df):
    analyzer = SentimentIntensityAnalyzer()

    scores = df["full_review_text"].fillna("").apply(analyzer.polarity_scores)
    scores_df = pd.DataFrame(scores.tolist())

    df = df.copy()
    df["vader_negative"] = scores_df["neg"]
    df["vader_neutral"] = scores_df["neu"]
    df["vader_positive"] = scores_df["pos"]
    df["vader_compound"] = scores_df["compound"]
    df["vader_sentiment"] = df["vader_compound"].apply(label_vader_sentiment)
    df["sentiment_match"] = df["sentiment"] == df["vader_sentiment"]

    return df


def print_review_examples(df):
    columns = ["product_name", "rating", "sentiment", "vader_sentiment", "vader_compound", "full_review_text"]

    print("Top 5 most positive reviews based on VADER compound score:")
    print(df.nlargest(5, "vader_compound")[columns].to_string(index=False))
    print()

    print("Top 5 most negative reviews based on VADER compound score:")
    print(df.nsmallest(5, "vader_compound")[columns].to_string(index=False))


def print_sentiment_report(df):
    match_percentage = df["sentiment_match"].mean() * 100

    print("Dataset shape:")
    print(df.shape)
    print()

    print("Original sentiment distribution:")
    print(df["sentiment"].value_counts())
    print()

    print("VADER sentiment distribution:")
    print(df["vader_sentiment"].value_counts())
    print()

    print("Sentiment match percentage:")
    print(f"{match_percentage:.2f}%")
    print()

    print_review_examples(df)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df = load_cleaned_data(CLEANED_DATA_PATH)
    results_df = score_reviews(df)

    SENTIMENT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(SENTIMENT_RESULTS_PATH, index=False)

    print_sentiment_report(results_df)
    print()
    print(f"Sentiment results saved to: {SENTIMENT_RESULTS_PATH}")


if __name__ == "__main__":
    main()
