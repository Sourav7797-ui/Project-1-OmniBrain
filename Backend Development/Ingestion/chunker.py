from typing import List

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """Splits text into overlapping character chunks."""
    if not text or not text.strip():
        return []
        
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks