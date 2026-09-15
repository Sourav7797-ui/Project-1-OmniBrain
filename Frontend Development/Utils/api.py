import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


def _get_headers(is_multipart: bool = False) -> dict:
    """Injects Bearer JWT authentication and default content headers."""
    headers = {}
    token = st.session_state.get("token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if not is_multipart:
        headers["Content-Type"] = "application/json"
    return headers


def _build_url(endpoint: str) -> str:
    """Safely builds backend URL ensuring single slash delimiter."""
    clean_endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    return f"{BACKEND_URL}{clean_endpoint}"


def get(endpoint: str, params: dict = None) -> requests.Response:
    """Dispatches a GET request to the FastAPI backend."""
    url = _build_url(endpoint)
    try:
        return requests.get(url, headers=_get_headers(), params=params, timeout=30)
    except requests.exceptions.RequestException as e:
        # Create an artificial response object to prevent crashing UI callers
        dummy_res = requests.Response()
        dummy_res.status_code = 503
        dummy_res._content = str(e).encode("utf-8")
        return dummy_res


def post(endpoint: str, json_data: dict = None) -> requests.Response:
    """Dispatches a POST request with JSON payload to the FastAPI backend."""
    url = _build_url(endpoint)
    try:
        return requests.post(url, headers=_get_headers(), json=json_data, timeout=60)
    except requests.exceptions.RequestException as e:
        dummy_res = requests.Response()
        dummy_res.status_code = 503
        dummy_res._content = str(e).encode("utf-8")
        return dummy_res


def upload_file(endpoint: str, file_tuple: tuple) -> requests.Response:
    """Sends a single multipart/form-data file for ingestion."""
    url = _build_url(endpoint)
    try:
        return requests.post(
            url,
            headers=_get_headers(is_multipart=True),
            files={"file": file_tuple},
            timeout=120,
        )
    except requests.exceptions.RequestException as e:
        dummy_res = requests.Response()
        dummy_res.status_code = 503
        dummy_res._content = str(e).encode("utf-8")
        return dummy_res


def upload_files(endpoint: str, files: list) -> requests.Response:
    """Sends batch multipart/form-data for document parsing and embedding."""
    url = _build_url(endpoint)
    try:
        return requests.post(
            url,
            headers=_get_headers(is_multipart=True),
            files=files,
            timeout=180,
        )
    except requests.exceptions.RequestException as e:
        dummy_res = requests.Response()
        dummy_res.status_code = 503
        dummy_res._content = str(e).encode("utf-8")
        return dummy_res


def stream_chat(endpoint: str, payload: dict):
    """Consumes Server-Sent Events (SSE) or chunked streams and yields tokens."""
    url = _build_url(endpoint)
    try:
        with requests.post(
            url,
            headers=_get_headers(),
            json=payload,
            stream=True,
            timeout=90,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    if line.startswith("data: "):
                        yield line[6:]
                    else:
                        yield line
    except Exception as exc:
        yield f"Stream connection error: {exc}"