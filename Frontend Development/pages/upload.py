import streamlit as st
from Utils.api import upload_file, upload_files

st.set_page_config(page_title="Upload Documents | OmniBrain", page_icon="📄", layout="wide")
st.header("📄 Document Ingestion Pipeline")
st.caption("Upload PDFs or image files for OCR, chunking, and vector embedding.")

uploaded_files = st.file_uploader(
    "Select files to upload",
    type=["pdf", "png", "jpg", "jpeg", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"**Selected files:** {len(uploaded_files)}")
    
    if st.button("Trigger Ingestion", type="primary"):
        progress_bar = st.progress(10, text="Packaging files for upload...")
        total_files = len(uploaded_files)
        success_count = 0
        error_messages = []

        # Iterate through files to support the backend's single/multi file endpoint
        for idx, file in enumerate(uploaded_files, start=1):
            pct = int(10 + (idx / total_files) * 85)
            progress_bar.progress(pct, text=f"Processing {file.name} ({idx}/{total_files})...")
            
            file_payload = (file.name, file.getvalue(), file.type or "application/octet-stream")
            
            try:
                # Dispatches to the active /api/v1/upload route
                response = upload_file("/api/v1/upload", file_tuple=file_payload)

                # Accepts standard 200 OK and 202 Accepted (asynchronous processing)
                if response.status_code in (200, 201, 202):
                    success_count += 1
                else:
                    error_messages.append(f"{file.name}: Status {response.status_code} - {response.text}")
            except Exception as err:
                error_messages.append(f"{file.name}: {err}")

        progress_bar.progress(100, text="Ingestion complete!")

        if success_count > 0:
            st.success(f"Successfully processed {success_count}/{total_files} document(s).")

        if error_messages:
            with st.expander("Failed Ingestion Details", expanded=True):
                for err in error_messages:
                    st.error(err)