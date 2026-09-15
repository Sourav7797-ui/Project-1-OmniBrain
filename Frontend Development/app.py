import streamlit as st

st.set_page_config(
    page_title="OmniBrain Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global Session State Invariant Setup
if "token" not in st.session_state:
    st.session_state["token"] = None
if "user" not in st.session_state:
    st.session_state["user"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def main():
    st.sidebar.title("🧠 OmniBrain")
    
    if st.session_state["token"]:
        username = st.session_state["user"].get("username", "User") if st.session_state["user"] else "User"
        st.sidebar.success(f"Authenticated: **{username}**")
        if st.sidebar.button("Logout"):
            st.session_state["token"] = None
            st.session_state["user"] = None
            st.rerun()
    else:
        st.sidebar.warning("No active session. Please authenticate via Profile.")

    st.title("OmniBrain Multi-Agent Platform")
    st.markdown("""
    Welcome to the OmniBrain Workspace. Navigate across views using the left sidebar:
    
    * **Chat (`Pages/chat.py`)**: Multi-agent conversational stream with agent reasoning steps and multimodal visual charts.
    * **Upload (`Pages/upload.py`)**: Multi-file document intake for PDF/image parsing and vector embedding.
    * **Profile (`Pages/profile.py`)**: Manage JWT session state, user credentials, and external API keys.
    * **Admin (`Pages/admin.py`)**: Monitor Langfuse system telemetry and document index statuses.
    """)

if __name__ == "__main__":
    main()