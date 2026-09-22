"""
OmniBrain Multimodal PDF Extractor.
Extracts text, structured tables (via PyMuPDF tables / pandas),
embedded raster images, and clustered vector graphics for multimodal RAG.
"""

import os
import fitz  # PyMuPDF
import pandas as pd
from typing import Tuple, List, Dict, Any


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts raw text content across all pages of a PDF document.
    Ensures seamless backward compatibility with ingestion routers.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")

    doc = fitz.open(file_path)
    full_text = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            full_text.append(text)
    doc.close()
    return "\n\n".join(full_text)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits continuous text into overlapping word chunks suitable for embeddings.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap

    return chunks


def extract_pdf_content(
    pdf_path: str,
    output_img_dir: str = "extracted_images",
    output_table_dir: str = "extracted_tables",
    min_image_dim: int = 80
) -> Tuple[List[str], List[pd.DataFrame], List[str]]:
    """
    Extracts text, tables, raster images, and clustered vector diagrams.
    Returns:
        (extracted_text, extracted_tables, extracted_image_paths)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    doc = fitz.open(pdf_path)
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_table_dir, exist_ok=True)

    extracted_text: List[str] = []
    extracted_tables: List[pd.DataFrame] = []
    extracted_image_paths: List[str] = []

    for page_num, page in enumerate(doc, start=1):
        page_rect = page.rect

        # 1. Text Extraction
        page_text = page.get_text()
        extracted_text.append(page_text)

        # 2. Table Extraction (Requires PyMuPDF >= 1.23.0)
        try:
            table_finder = page.find_tables()
            if table_finder and table_finder.tables:
                for idx, table in enumerate(table_finder.tables, start=1):
                    df = table.to_pandas()
                    if not df.empty:
                        extracted_tables.append(df)
                        csv_path = os.path.join(output_table_dir, f"page_{page_num}_table_{idx}.csv")
                        df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Table extraction skipped on page {page_num}: {e}")

        # 3. Raster Image Extraction (Bitmaps, JPEGs, PNGs)
        raw_images = page.get_images()
        for img_idx, img in enumerate(raw_images, start=1):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                # Convert CMYK/non-RGB colorspaces to standard RGB
                if pix.n - pix.alpha > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                if pix.width >= min_image_dim and pix.height >= min_image_dim:
                    raw_img_path = os.path.join(output_img_dir, f"page_{page_num}_raster_{img_idx}.png")
                    pix.save(raw_img_path)
                    extracted_image_paths.append(raw_img_path)
                pix = None
            except Exception as e:
                print(f"Skipped raster image {img_idx} on page {page_num}: {e}")

        # 4. Vector Diagram Extraction (Clustering nearby vectors/drawings)
        try:
            drawings = page.get_drawings()
            clusters: List[fitz.Rect] = []

            for d in drawings:
                rect = fitz.Rect(d["rect"])
                # Filter out tiny vector lines, borders, and artifacts
                if rect.width < 40 or rect.height < 40:
                    continue

                merged = False
                # Expand rectangle by 15px margin to group nearby components into diagrams
                expanded_rect = fitz.Rect(
                    max(0, rect.x0 - 15),
                    max(0, rect.y0 - 15),
                    min(page_rect.width, rect.x1 + 15),
                    min(page_rect.height, rect.y1 + 15)
                )

                for cluster in clusters:
                    if cluster.intersects(expanded_rect):
                        cluster |= rect
                        merged = True
                        break

                if not merged:
                    clusters.append(rect)

            # Render each distinct vector diagram
            for v_idx, cluster_rect in enumerate(clusters, start=1):
                # Clamp within page dimensions to prevent pixmap clip errors
                clamped_rect = cluster_rect & page_rect
                if clamped_rect.width >= min_image_dim and clamped_rect.height >= min_image_dim:
                    mat = fitz.Matrix(2, 2)  # 2x scale for clear OCR and vision models
                    pix = page.get_pixmap(matrix=mat, clip=clamped_rect)
                    vec_img_path = os.path.join(output_img_dir, f"page_{page_num}_vector_{v_idx}.png")
                    pix.save(vec_img_path)
                    extracted_image_paths.append(vec_img_path)
        except Exception as e:
            print(f"Vector diagram extraction failed on page {page_num}: {e}")

    doc.close()
    return extracted_text, extracted_tables, extracted_image_paths


def extract_and_chunk_pdf(
    pdf_path: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    End-to-end multimodal pipeline:
    Extracts text, converts tables to Markdown, and combines them
    into chunk dictionaries ready for embedding and Qdrant storage.
    """
    texts, tables, image_paths = extract_pdf_content(pdf_path)

    # 1. Chunk standard text
    full_text = "\n\n".join(texts)
    text_chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)

    chunks_data: List[Dict[str, Any]] = []

    for idx, tc in enumerate(text_chunks, start=1):
        chunks_data.append({
            "content": tc,
            "type": "text",
            "chunk_index": idx,
            "metadata": {"source": os.path.basename(pdf_path)}
        })

    # 2. Add tables formatted as Markdown chunks
    for t_idx, df in enumerate(tables, start=1):
        try:
            table_md = df.to_markdown(index=False)
            table_content = f"Table {t_idx} from {os.path.basename(pdf_path)}:\n{table_md}"
            chunks_data.append({
                "content": table_content,
                "type": "table",
                "chunk_index": len(chunks_data) + 1,
                "metadata": {"source": os.path.basename(pdf_path), "table_index": t_idx}
            })
        except Exception:
            pass

    return chunks_data