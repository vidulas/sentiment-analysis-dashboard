from pathlib import Path
import os
import sys

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "dashboard_outputs"
FALLBACK_BUCKET_NAME = "sentiment-dashboard-vidulas-2026"
DEFAULT_S3_PREFIX = "dashboard_outputs/"


def get_bucket_name():
    load_dotenv(PROJECT_ROOT / ".env")
    return os.getenv("S3_BUCKET_NAME", FALLBACK_BUCKET_NAME)


def get_dashboard_files(local_folder):
    if not local_folder.exists():
        raise FileNotFoundError(f"Local folder not found: {local_folder}")

    if not local_folder.is_dir():
        raise NotADirectoryError(f"Expected a folder but found a file: {local_folder}")

    files = [path for path in local_folder.iterdir() if path.is_file()]
    if not files:
        raise FileNotFoundError(f"No files found to upload in: {local_folder}")

    return sorted(files)


def upload_files_to_s3(local_folder, bucket_name, s3_prefix):
    s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
    files = get_dashboard_files(local_folder)
    uploaded_count = 0

    for file_path in files:
        s3_key = f"{s3_prefix.rstrip('/')}/{file_path.name}"
        s3_client.upload_file(str(file_path), bucket_name, s3_key)
        uploaded_count += 1
        print(f"Uploaded {file_path.name} -> s3://{bucket_name}/{s3_key}")

    return uploaded_count


def print_upload_summary(bucket_name, s3_prefix, uploaded_count):
    print()
    print("Upload summary")
    print(f"Bucket name: {bucket_name}")
    print(f"S3 prefix: {s3_prefix}")
    print(f"Number of files uploaded: {uploaded_count}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    bucket_name = get_bucket_name()

    try:
        uploaded_count = upload_files_to_s3(
            DASHBOARD_OUTPUT_DIR,
            bucket_name,
            DEFAULT_S3_PREFIX,
        )
    except FileNotFoundError as error:
        print(f"Local folder error: {error}")
        sys.exit(1)
    except NotADirectoryError as error:
        print(f"Local path error: {error}")
        sys.exit(1)
    except (NoCredentialsError, PartialCredentialsError) as error:
        print(f"AWS credential error: {error}")
        print("Configure AWS credentials before running this script.")
        sys.exit(1)
    except ClientError as error:
        print(f"AWS client error: {error}")
        sys.exit(1)

    print_upload_summary(bucket_name, DEFAULT_S3_PREFIX, uploaded_count)


if __name__ == "__main__":
    main()
