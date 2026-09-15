import streamlit as st
import pandas as pd
from Utils.api import get

st.set_page_config(page_title="Admin Console | OmniBrain", page_icon="⚙️", layout="wide")
st.header("⚙️ Administrative & Telemetry Portal")

# Top-level performance metric cards
st.subheader("System Telemetry & Health")
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

try:
    metrics_res = get("/api/v1/admin/metrics")
    if metrics_res.status_code == 200:
        data = metrics_res.json()
        col_m1.metric("Active Sessions", data.get("active_sessions", 0))
        col_m2.metric("Total Documents", data.get("total_documents", 0))
        col_m3.metric("Vector Store", data.get("vector_store_status", "connected"))
        col_m4.metric("Uptime", f"{round(data.get('uptime_seconds', 0), 1)}s")
    else:
        col_m1.metric("Active Sessions", "N/A")
        col_m2.metric("Total Documents", "N/A")
        col_m3.metric("Vector Store", "Offline")
        col_m4.metric("Uptime", "0s")
except Exception:
    col_m1.metric("Active Sessions", "Offline")
    col_m2.metric("Total Documents", "Offline")
    col_m3.metric("Vector Store", "Offline")
    col_m4.metric("Uptime", "0s")

st.divider()

col_left, col_right = st.columns(2)

# Column 1: Document Index Registry
with col_left:
    st.subheader("Document Index Registry")
    if st.button("Refresh Documents"):
        st.rerun()

    try:
        docs_res = get("/api/v1/admin/docs")
        if docs_res.status_code == 200:
            docs_data = docs_res.json()
            # Normalize plain lists or nested dictionary responses
            if isinstance(docs_data, dict):
                docs_list = docs_data.get("documents", [])
            elif isinstance(docs_data, list):
                docs_list = docs_data
            else:
                docs_list = []

            if docs_list:
                df_docs = pd.DataFrame(docs_list)
                st.dataframe(df_docs, use_container_width=True, hide_index=True)
            else:
                st.info("No documents currently indexed in vector store.")
        elif docs_res.status_code == 403:
            st.warning("Admin authorization required to view indexed documents.")
        else:
            st.error(f"Failed to fetch documents ({docs_res.status_code}): {docs_res.text}")
    except Exception as err:
        st.error(f"Could not connect to document registry: {err}")

# Column 2: User Access Registry
with col_right:
    st.subheader("User Access Registry")
    if st.button("Refresh Users"):
        st.rerun()

    try:
        users_res = get("/api/v1/admin/users")
        if users_res.status_code == 200:
            users_data = users_res.json()
            # Normalize plain lists or nested dictionary responses
            if isinstance(users_data, dict):
                users_list = users_data.get("users", [])
            elif isinstance(users_data, list):
                users_list = users_data
            else:
                users_list = []

            if users_list:
                df_users = pd.DataFrame(users_list)
                # Drop sensitive hashes if returned in payload
                if "hashed_password" in df_users.columns:
                    df_users = df_users.drop(columns=["hashed_password"])
                st.dataframe(df_users, use_container_width=True, hide_index=True)
            else:
                st.info("No registered users found.")
        elif users_res.status_code == 403:
            st.warning("Admin authorization required to inspect user accounts.")
        else:
            st.error(f"Failed to fetch user list ({users_res.status_code}): {users_res.text}")
    except Exception as err:
        st.error(f"Could not connect to user registry: {err}")