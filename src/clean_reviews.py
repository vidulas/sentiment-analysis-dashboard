from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sentiment.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_reviews.csv"

REQUIRED_COLUMNS = ["review", "summary", "sentiment", "rating", "product_name"]
VALID_SENTIMENTS = {"positive", "negative", "neutral"}


def load_raw_data(file_path):
    """Load the raw CSV with a UTF-8 attempt first and latin1 fallback."""
    try:
        return pd.read_csv(file_path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="latin1", low_memory=False)


def standardize_column_names(df):
    column_mapping = {
        "ProductName": "product_name",
        "ProductPrice": "product_price",
        "Rate": "rating",
        "Review": "review",
        "Summary": "summary",
        "Sentiment": "sentiment",
    }

    df = df.rename(columns=column_mapping)
    df.columns = [
        re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]+", "_", col).strip("_")).lower()
        for col in df.columns
    ]
    return df


def clean_price(price):
    price = str(price)
    price = re.sub(r"[^0-9.]", "", price)
    return pd.to_numeric(price, errors="coerce")


def clean_text(text):
    text = str(text)
    text = re.sub(r"\?{2,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_reviews(df):
    df = standardize_column_names(df)

    missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_required:
        missing = ", ".join(missing_required)
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=REQUIRED_COLUMNS).copy()

    df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
    df = df[df["sentiment"].isin(VALID_SENTIMENTS)].copy()

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[df["rating"].between(1, 5)].copy()

    if "product_price" in df.columns:
        df["product_price"] = df["product_price"].apply(clean_price)

    text_columns = ["product_name", "review", "summary", "sentiment"]
    for col in text_columns:
        df[col] = df[col].apply(clean_text)

    df["full_review_text"] = df["review"] + " " + df["summary"]
    df["full_review_text"] = df["full_review_text"].apply(clean_text)
    df["review_length"] = df["full_review_text"].str.len()
    df["word_count"] = df["full_review_text"].str.split().str.len()

    return df


def print_cleaning_report(original_shape, cleaned_df):
    print("Original shape:")
    print(original_shape)
    print()

    print("Cleaned shape:")
    print(cleaned_df.shape)
    print()

    print("Missing values after cleaning:")
    print(cleaned_df.isnull().sum())
    print()

    print("Sentiment distribution after cleaning:")
    print(cleaned_df["sentiment"].value_counts())
    print()

    print("Rating distribution after cleaning:")
    print(cleaned_df["rating"].value_counts().sort_index())


def main():
    df = load_raw_data(RAW_DATA_PATH)
    original_shape = df.shape

    cleaned_df = clean_reviews(df)

    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)

    print_cleaning_report(original_shape, cleaned_df)
    print()
    print(f"Cleaned dataset saved to: {CLEANED_DATA_PATH}")


if __name__ == "__main__":
    main()
