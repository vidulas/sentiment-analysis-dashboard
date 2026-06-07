# Customer Sentiment Analysis Dashboard Using NLP and AWS S3

## Project Overview

This portfolio project will build a customer and product sentiment analysis dashboard using product review data. The project will inspect and clean review records, analyze customer sentiment, generate processed datasets, upload outputs to AWS S3, and create a dashboard for business insights.

## Business Problem

Businesses often receive large volumes of product reviews, but it can be difficult to quickly understand customer satisfaction, product-level issues, and sentiment trends. This project aims to turn product review data into practical insights that can help teams identify customer pain points, monitor product performance, and support data-driven decisions.

## Planned Tech Stack

- Python
- pandas
- NumPy
- Matplotlib
- NLTK
- VADER Sentiment
- AWS S3
- boto3
- python-dotenv
- Jupyter Notebook
- Power BI or Streamlit

## Planned Architecture

1. Store the raw dataset in `data/raw/sentiment.csv`.
2. Inspect and clean the review dataset using Python notebooks and scripts.
3. Generate processed sentiment analysis outputs.
4. Upload processed outputs to AWS S3.
5. Connect processed data to a Power BI or Streamlit dashboard.
6. Create business insights and recommendations from the dashboard.

## Development Stages

1. Data inspection
2. Data cleaning
3. Sentiment analysis
4. Processed dataset generation
5. AWS S3 upload
6. Dashboard creation
7. Insights and recommendations

## Stage 2: Data Cleaning

The Stage 2 cleaning script is located at `src/clean_reviews.py`. It loads the raw dataset from `data/raw/sentiment.csv`, standardizes column names, removes rows with missing required review fields, cleans sentiment labels, converts ratings and product prices to numeric values, lightly cleans readable text fields, and creates `full_review_text`, `review_length`, and `word_count`.

Run the script with:

```bash
python src/clean_reviews.py
```

The cleaned dataset is saved to `data/processed/cleaned_reviews.csv`.

## Stage 3: Sentiment Analysis

The Stage 3 sentiment script is located at `src/sentiment_analysis.py`. It loads `data/processed/cleaned_reviews.csv`, applies VADER sentiment scoring to `full_review_text`, creates VADER score columns, converts compound scores into positive, negative, or neutral labels, and compares those labels with the original dataset sentiment column.

Run the script with:

```bash
python src/sentiment_analysis.py
```

The sentiment analysis output is saved to `data/processed/sentiment_results.csv`. The notebook `notebooks/02_sentiment_analysis.ipynb` explains the VADER process step by step with beginner-friendly examples and basic visualizations.

## Stage 4: Dashboard Outputs

The Stage 4 dashboard output script is located at `src/create_dashboard_outputs.py`. It loads `data/processed/sentiment_results.csv` and creates dashboard-ready CSV and JSON files for review-level data, sentiment distribution, rating distribution, product sentiment summaries, top positive and negative review examples, sentiment match summary, and overall KPIs.

Run the script with:

```bash
python src/create_dashboard_outputs.py
```

The dashboard-ready files are saved inside `data/processed/dashboard_outputs/`. The notebook `notebooks/03_dashboard_output_analysis.ipynb` explains the outputs and includes simple visualizations for KPIs, sentiment, ratings, product review counts, and negative review percentages.

## Stage 5: AWS S3 Upload

The dashboard output files from `data/processed/dashboard_outputs/` are uploaded to AWS S3 for storage and later dashboard access. The bucket used for this project is `sentiment-dashboard-vidulas-2026`, and the dashboard files are stored under the S3 prefix `dashboard_outputs/`.

The AWS CLI was used to manually upload and verify the dashboard output files. The script `src/upload_to_s3.py` can automate the upload process using boto3.

Run the script with:

```bash
python src/upload_to_s3.py
```

The script reads the bucket name from the `S3_BUCKET_NAME` environment variable. If that variable is missing, it falls back to `sentiment-dashboard-vidulas-2026`. Use `.env.example` as a template for local environment settings.

Do not commit AWS access keys, secret keys, session tokens, or private credentials.
