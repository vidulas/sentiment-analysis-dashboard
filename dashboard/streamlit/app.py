from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "dashboard_outputs"

REQUIRED_FILES = {
    "dashboard_kpis": "dashboard_kpis.json",
    "dashboard_main": "dashboard_main.csv",
    "product_summary": "product_sentiment_summary.csv",
    "sentiment_distribution": "sentiment_distribution.csv",
    "rating_distribution": "rating_distribution.csv",
    "top_positive_reviews": "top_positive_reviews.csv",
    "top_negative_reviews": "top_negative_reviews.csv",
    "sentiment_match_summary": "sentiment_match_summary.csv",
}


st.set_page_config(
    page_title="Customer Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_dashboard_outputs():
    missing_files = [
        filename
        for filename in REQUIRED_FILES.values()
        if not (OUTPUT_DIR / filename).exists()
    ]

    if missing_files:
        return None, missing_files

    with (OUTPUT_DIR / REQUIRED_FILES["dashboard_kpis"]).open("r", encoding="utf-8") as file:
        dashboard_kpis = json.load(file)

    data = {
        "dashboard_kpis": dashboard_kpis,
        "dashboard_main": pd.read_csv(OUTPUT_DIR / REQUIRED_FILES["dashboard_main"]),
        "product_summary": pd.read_csv(OUTPUT_DIR / REQUIRED_FILES["product_summary"]),
        "sentiment_distribution": pd.read_csv(OUTPUT_DIR / REQUIRED_FILES["sentiment_distribution"]),
        "rating_distribution": pd.read_csv(OUTPUT_DIR / REQUIRED_FILES["rating_distribution"]),
        "top_positive_reviews": pd.read_csv(OUTPUT_DIR / REQUIRED_FILES["top_positive_reviews"]),
        "top_negative_reviews": pd.read_csv(OUTPUT_DIR / REQUIRED_FILES["top_negative_reviews"]),
        "sentiment_match_summary": pd.read_csv(OUTPUT_DIR / REQUIRED_FILES["sentiment_match_summary"]),
    }
    return data, []


def plot_bar_chart(df, x_col, y_col, title, x_label, y_label, color):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df[x_col].astype(str), df[y_col], color=color)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return fig


def plot_horizontal_bar_chart(df, x_col, y_col, title, x_label, y_label, color):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df[y_col].astype(str), df[x_col], color=color)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.tight_layout()
    return fig


def format_percentage(value):
    return f"{value:.2f}%"


def filter_reviews(df):
    st.sidebar.header("Filters")

    sentiment_options = sorted(df["vader_sentiment"].dropna().unique().tolist())
    selected_sentiments = st.sidebar.multiselect(
        "VADER sentiment",
        options=sentiment_options,
        default=sentiment_options,
    )

    rating_options = sorted(df["rating"].dropna().unique().tolist())
    selected_ratings = st.sidebar.multiselect(
        "Rating",
        options=rating_options,
        default=rating_options,
    )

    product_search = st.sidebar.text_input("Product name search")

    filtered_df = df[
        df["vader_sentiment"].isin(selected_sentiments)
        & df["rating"].isin(selected_ratings)
    ].copy()

    if product_search.strip():
        filtered_df = filtered_df[
            filtered_df["product_name"].str.contains(product_search.strip(), case=False, na=False)
        ]

    return filtered_df


def show_kpis(kpis):
    first_row = st.columns(4)
    first_row[0].metric("Total Reviews", f"{kpis['total_reviews']:,}")
    first_row[1].metric("Total Products", f"{kpis['total_products']:,}")
    first_row[2].metric("Average Rating", f"{kpis['average_rating']:.2f}")
    first_row[3].metric("Average Word Count", f"{kpis['average_word_count']:.2f}")

    second_row = st.columns(4)
    second_row[0].metric("Positive Reviews", format_percentage(kpis["positive_percentage"]))
    second_row[1].metric("Negative Reviews", format_percentage(kpis["negative_percentage"]))
    second_row[2].metric("Neutral Reviews", format_percentage(kpis["neutral_percentage"]))
    second_row[3].metric("Sentiment Match", format_percentage(kpis["sentiment_match_percentage"]))


def show_executive_overview(data):
    st.header("Executive Overview")
    show_kpis(data["dashboard_kpis"])

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        st.pyplot(
            plot_bar_chart(
                data["sentiment_distribution"],
                "vader_sentiment",
                "count",
                "VADER Sentiment Distribution",
                "Sentiment",
                "Number of Reviews",
                "#2f6f9f",
            )
        )

    with chart_col_2:
        st.pyplot(
            plot_bar_chart(
                data["rating_distribution"],
                "rating",
                "count",
                "Rating Distribution",
                "Rating",
                "Number of Reviews",
                "#3f8f5f",
            )
        )


def show_product_sentiment_analysis(product_summary):
    st.header("Product Sentiment Analysis")

    top_reviewed = product_summary.sort_values("review_count", ascending=False).head(10)
    top_negative = product_summary.sort_values(
        ["negative_percentage", "review_count"], ascending=[False, False]
    ).head(10)

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        st.pyplot(
            plot_horizontal_bar_chart(
                top_reviewed.sort_values("review_count"),
                "review_count",
                "product_name",
                "Top 10 Products by Review Count",
                "Review Count",
                "Product Name",
                "#d27a2c",
            )
        )

    with chart_col_2:
        st.pyplot(
            plot_horizontal_bar_chart(
                top_negative.sort_values("negative_percentage"),
                "negative_percentage",
                "product_name",
                "Top 10 Products by Negative Percentage",
                "Negative Percentage",
                "Product Name",
                "#b3453c",
            )
        )

    st.subheader("Product Sentiment Summary")
    product_table_search = st.text_input("Search product summary table")
    product_table = product_summary.copy()
    if product_table_search.strip():
        product_table = product_table[
            product_table["product_name"].str.contains(
                product_table_search.strip(),
                case=False,
                na=False,
            )
        ]
    st.dataframe(product_table, use_container_width=True)


def show_review_explorer(filtered_df, top_positive_reviews, top_negative_reviews):
    st.header("Review Explorer")
    st.caption(f"Showing {len(filtered_df):,} filtered reviews")

    review_columns = [
        "product_name",
        "rating",
        "summary",
        "vader_sentiment",
        "vader_compound",
        "word_count",
    ]
    st.dataframe(filtered_df[review_columns], use_container_width=True)

    review_col_1, review_col_2 = st.columns(2)
    with review_col_1:
        st.subheader("Top Positive Reviews")
        st.dataframe(top_positive_reviews, use_container_width=True)

    with review_col_2:
        st.subheader("Top Negative Reviews")
        st.dataframe(top_negative_reviews, use_container_width=True)


def main():
    st.title("Customer Sentiment Analysis Dashboard")
    st.write(
        "This dashboard analyzes product review sentiment using VADER NLP sentiment "
        "scoring. It summarizes review volume, sentiment patterns, product-level "
        "sentiment, and example reviews for business insight."
    )

    data, missing_files = load_dashboard_outputs()
    if missing_files:
        st.error("Some dashboard output files are missing.")
        st.write("Expected folder:", str(OUTPUT_DIR))
        st.write("Missing files:")
        st.write(missing_files)
        st.stop()

    filtered_reviews = filter_reviews(data["dashboard_main"])

    show_executive_overview(data)
    show_product_sentiment_analysis(data["product_summary"])
    show_review_explorer(
        filtered_reviews,
        data["top_positive_reviews"],
        data["top_negative_reviews"],
    )


if __name__ == "__main__":
    main()
