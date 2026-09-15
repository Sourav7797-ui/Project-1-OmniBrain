import streamlit as st
from Utils.api import post, get

st.set_page_config(page_title="Profile & Settings | OmniBrain", page_icon="👤", layout="wide")
st.header("👤 Profile & Authentication")

tab_login, tab_register, tab_settings = st.tabs(["Authentication", "Register New User", "API Keys & Preferences"])

# Tab 1: Login & Session Management
with tab_login:
    if not st.session_state.get("token"):
        st.subheader("Sign In")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Sign In", type="primary"):
            if not username or not password:
                st.warning("Please provide both username and password.")
            else:
                try:
                    # Dispatches to the unified /api/v1/auth/login route
                    res = post("/api/v1/auth/login", json_data={"username": username, "password": password})
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state["token"] = data.get("access_token")
                        st.session_state["user"] = data.get("user", {"username": username, "role": "user"})
                        st.success("Authentication successful!")
                        st.rerun()
                    else:
                        st.error(f"Login failed ({res.status_code}): {res.text}")
                except Exception as err:
                    st.error(f"Authentication failed: {err}")
    else:
        user = st.session_state.get("user", {})
        username = user.get("username", "Authenticated User")
        role = user.get("role", "user")

        st.success(f"Currently signed in as: **{username}** (Role: `{role}`)")
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.write(f"**User ID:** {user.get('id', 'N/A')}")
            st.write(f"**Email:** {user.get('email', 'Not provided')}")
        with col_info2:
            st.write(f"**Session Active:** Yes")
            st.write(f"**Token Type:** Bearer")

        if st.button("Log Out", type="secondary"):
            st.session_state["token"] = None
            st.session_state["user"] = None
            st.rerun()

# Tab 2: User Registration
with tab_register:
    st.subheader("Create an OmniBrain Account")
    reg_username = st.text_input("New Username", key="reg_username")
    reg_email = st.text_input("Email (optional)", key="reg_email")
    reg_password = st.text_input("New Password", type="password", key="reg_password")
    reg_role = st.selectbox("Requested Role", ["user", "analyst"], index=0)

    if st.button("Register Account", type="primary"):
        if not reg_username or not reg_password:
            st.warning("Username and password are required.")
        else:
            payload = {
                "username": reg_username,
                "password": reg_password,
                "email": reg_email or None,
                "role": reg_role
            }
            reg_res = post("/api/v1/auth/register", json_data=payload)
            if reg_res.status_code in (200, 201):
                st.success("Registration successful! You can now log in using the Authentication tab.")
            else:
                st.error(f"Registration failed ({reg_res.status_code}): {reg_res.text}")

# Tab 3: API Keys and Model Preferences
with tab_settings:
    st.subheader("External Integrations & API Keys")
    if not st.session_state.get("token"):
        st.info("Log in to customize and persist account API keys.")
    else:
        openai_key = st.text_input("Custom LLM API Key (OpenAI / Ollama URL)", type="password")
        langfuse_key = st.text_input("Langfuse Public Tracking Key", type="password")
        
        if st.button("Save Configuration", type="primary"):
            res = post("/api/v1/user/settings", json_data={
                "openai_api_key": openai_key,
                "langfuse_key": langfuse_key
            })
            if res.status_code in (200, 204):
                st.success("API configuration successfully saved.")
            else:
                st.warning(f"Settings saved locally ({res.status_code}).")