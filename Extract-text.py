from pypdf import PdfReader

reader = PdfReader("uploaded.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text() + "\n"

print(text[:500])  # Show first 500 characters

with open("pdf_text.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Text extracted successfully")
