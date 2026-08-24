import os
import uuid
import ollama
import pandas as pd
import cloudinary
import cloudinary.uploader
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

VISION_MODEL = "minicpm-v"


def upload_to_cloudinary(file_path: str, public_id: str) -> Dict[str, str]:
    """Uploads file to Cloudinary and returns CDN links."""
    upload_res = cloudinary.uploader.upload(
        file_path,
        public_id=public_id,
        overwrite=True,
        resource_type="auto"
    )
    return {
        "public_id": upload_res.get("public_id"),
        "secure_url": upload_res.get("secure_url")
    }


def summarize_table(df: pd.DataFrame) -> str:
    """Summarizes table data using Vision LLM."""
    table_md = df.to_markdown(index=False)
    prompt = f"Act as a document analyst. Summarize key financial/data points:\n{table_md}"
    try:
        res = ollama.generate(
            model=VISION_MODEL,
            prompt=prompt,
            options={"num_gpu": 99, "temperature": 0.2}
        )
        return res["response"].strip()
    except Exception:
        return f"Table data: {table_md[:200]}"


def summarize_image(img_path: str) -> str:
    """Summarizes image content using Vision LLM."""
    prompt = """
Look at this document visual.
1. Transcribe any readable text or labels.
2. Describe what kind of diagram/chart/logo this is.
3. Summarize its core meaning in detail.
"""
    try:
        res = ollama.generate(
            model=VISION_MODEL,
            prompt=prompt,
            images=[img_path],
            options={"num_gpu": 99, "temperature": 0.2}
        )
        return res["response"].strip()
    except Exception:
        return "Extracted document image."