from pipeline import run_pipeline
from pdf_extractor import extract_pdf_content

# extracting pdf content

extracted_text, extracted_tables, extracted_image_paths = extract_pdf_content(pdf_path="our_pdf_name.pdf") # put your pdf path

res = run_pipeline(tables=extracted_tables, image_paths=extracted_image_paths, texts=extracted_text, user_id="user_id1")

print(res)