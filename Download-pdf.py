import boto3

bucket_name = "financial-rag-pdf-bucket"
pdf_file = "sample_financial_report.pdf"

s3 = boto3.client("s3")

s3.download_file(
    bucket_name,
    pdf_file,
    "uploaded.pdf"
)

print("PDF downloaded successfully")
