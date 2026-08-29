import os
import uuid
import pandas as pd
from typing import List, Dict, Any, Optional
import ollama
from extractor import summarize_table, upload_to_cloudinary, summarize_image
from chunker import chunk_text
from embedder import embed_and_store_chunks

VISION_MODEL = "minicpm-v"

def process_tables(tables: List[pd.DataFrame], user_id:str, output_dir: str = "extracted_tables") -> List[Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for idx, df in enumerate(tables, start=1):
        asset_id = f"table_{uuid.uuid4().hex[:10]}"
        
        # Save temp CSV
        csv_path = os.path.join(output_dir, f"{asset_id}.csv")
        df.to_csv(csv_path, index=False)

        # 1. Extraction
        summary = summarize_table(df)
        c_meta = upload_to_cloudinary(csv_path, public_id=f"tables/{asset_id}")
        
        # Combine summary & Markdown table
        full_content = f"Table Summary: {summary}\n\nRaw Data:\n{df.to_markdown(index=False)}"

        # 2. Chunking
        chunks = chunk_text(full_content, chunk_size=1000, overlap=150)

        # 3. Embedding & Vector DB Insertion
        embed_and_store_chunks(
            chunks=chunks,
            user_id=user_id,
            parent_asset_id=asset_id,
            asset_type="table",
            cloudinary_public_id=c_meta["public_id"],
            cloudinary_url=c_meta["secure_url"]
        )

        print(f"✅ Indexed Table {idx} | Cloudinary ID: {c_meta['public_id']}")
        results.append({"id": asset_id, "summary": summary, "url": c_meta["secure_url"]})

    return results


def process_images(image_paths: List[str], user_id:str) -> List[Dict[str, Any]]:
    results = []

    for idx, img_path in enumerate(image_paths, start=1):
        if not os.path.exists(img_path):
            continue

        asset_id = f"image_{uuid.uuid4().hex[:10]}"

        # 1. Extraction
        summary = summarize_image(img_path)
        c_meta = upload_to_cloudinary(img_path, public_id=f"images/{asset_id}")

        # 2. Chunking
        chunks = chunk_text(summary, chunk_size=800, overlap=100)

        # 3. Embedding & Vector DB Insertion
        embed_and_store_chunks(
            chunks=chunks,
            user_id=user_id,
            parent_asset_id=asset_id,
            asset_type="image",
            cloudinary_public_id=c_meta["public_id"],
            cloudinary_url=c_meta["secure_url"],
            extra_metadata={"local_path": img_path}
        )

        print(f"✅ Indexed Image {idx} | Cloudinary ID: {c_meta['public_id']}")
        results.append({"id": asset_id, "summary": summary, "url": c_meta["secure_url"]})

    return results


def process_texts(text_blocks: List[str], user_id:str) -> None:
    for idx, text in enumerate(text_blocks, start=1):
        if not text.strip():
            continue

        asset_id = f"text_page_{idx}_{uuid.uuid4().hex[:6]}"
        chunks = chunk_text(text, chunk_size=1000, overlap=150)

        embed_and_store_chunks(
            chunks=chunks,
            user_id=user_id,
            parent_asset_id=asset_id,
            asset_type="text_block",
            extra_metadata={"page_number": idx}
        )


def run_pipeline(
    tables: List[pd.DataFrame],
    image_paths: List[str],
    user_id:str,
    texts: Optional[List[str]] = None
) -> str:
    print("🚀 Starting Cloudinary Sync & Vector DB Ingestion...")

    table_results = process_tables(tables, user_id=user_id)
    image_results = process_images(image_paths, user_id=user_id)

    if texts:
        process_texts(texts, user_id=user_id)

    # Synthesize Executive Summary
    all_summaries = [t["summary"] for t in table_results] + [i["summary"] for i in image_results]
    combined_notes = "\n".join(all_summaries)

    try:
        exec_prompt = f"Synthesize these summaries into an executive summary:\n{combined_notes[:6000]}"
        exec_res = ollama.generate(
            model=VISION_MODEL,
            prompt=exec_prompt,
            options={"num_gpu": 99, "num_ctx": 4096}
        )
        exec_summary = exec_res["response"].strip()
    except Exception:
        exec_summary = "Processed document content into Vector DB."

    return f"# Document Extraction Summaries\n\nProcessed {len(table_results)} tables and {len(image_results)} images.\n\n# Executive Summary\n{exec_summary}"

