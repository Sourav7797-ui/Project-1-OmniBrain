import fitz  # PyMuPDF
import pandas as pd
import os
from typing import Tuple, List

def extract_pdf_content(
    pdf_path: str, 
    output_img_dir: str = "extracted_images", 
    output_table_dir: str = "extracted_tables",
    min_image_dim: int = 80
) -> Tuple[List[str], List[pd.DataFrame], List[str]]:
    """
    Extracts text, tables, embedded raster images, AND vector diagrams separately.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    doc = fitz.open(pdf_path)
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_table_dir, exist_ok=True)

    extracted_text = []
    extracted_tables = []
    extracted_image_paths = []

    for page_num, page in enumerate(doc, start=1):
        # 1. Text Extraction
        extracted_text.append(page.get_text())

        # 2. Table Extraction
        table_finder = page.find_tables()
        if table_finder.tables:
            for idx, table in enumerate(table_finder.tables, start=1):
                df = table.to_pandas()
                extracted_tables.append(df)
                csv_path = os.path.join(output_table_dir, f"page_{page_num}_table_{idx}.csv")
                df.to_csv(csv_path, index=False)

        # 3. Raster Image Extraction (Bitmaps, JPEGs, PNGs)
        raw_images = page.get_images()
        for img_idx, img in enumerate(raw_images, start=1):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                if pix.width >= min_image_dim and pix.height >= min_image_dim:
                    raw_img_path = os.path.join(output_img_dir, f"page_{page_num}_raster_{img_idx}.png")
                    pix.save(raw_img_path)
                    extracted_image_paths.append(raw_img_path)
                pix = None
            except Exception as e:
                print(f"Skipped raster image {img_idx} on page {page_num}: {e}")

        # 4. Vector Diagram Extraction (Group close vector paths into distinct clusters)
        drawings = page.get_drawings()
        clusters = []

        for d in drawings:
            rect = fitz.Rect(d["rect"])
            # Ignore tiny vector artifacts (bullet points, underlines, rule lines)
            if rect.width < 40 or rect.height < 40:
                continue

            # Check if this drawing belongs to an existing cluster (overlapping/nearby)
            merged = False
            for cluster in clusters:
                # Expand rectangle by 15px margin to group nearby lines/shapes into one diagram
                if cluster.intersects(rect + (-15, -15, 15, 15)):
                    cluster |= rect
                    merged = True
                    break
            if not merged:
                clusters.append(rect)

        # Save each distinct vector cluster as a high-resolution image
        for v_idx, cluster_rect in enumerate(clusters, start=1):
            if cluster_rect.width >= min_image_dim and cluster_rect.height >= min_image_dim:
                # Render region at 2x resolution (DPI ~144) for clear OCR in Ollama
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat, clip=cluster_rect)
                
                vec_img_path = os.path.join(output_img_dir, f"page_{page_num}_vector_{v_idx}.png")
                pix.save(vec_img_path)
                extracted_image_paths.append(vec_img_path)

    doc.close()
    return extracted_text, extracted_tables, extracted_image_paths