import json
import uuid
import streamlit as st
from Utils.api import stream_chat, post

st.set_page_config(page_title="Chat | OmniBrain", page_icon="💬", layout="wide")
st.header("💬 Multi-Agent Chat Interface")

# Initialize chat session ID and state
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg.get("thought"):
            with st.expander("Agent Reasoning / Thoughts", expanded=False):
                st.markdown(msg["thought"])
        st.markdown(msg["content"])
        if msg.get("chart"):
            st.image(msg["chart"], caption="Agent Generated Multimodal Visual")

# User prompt handling
if prompt := st.chat_input("Ask a question or request analysis..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        thought_container = st.empty()
        response_container = st.empty()

        full_response = ""
        full_thought = ""
        chart_data = None

        # Schema payload expected by backend /api/v1/chat
        payload = {
            "query": prompt,
            "session_id": st.session_state["session_id"]
        }

        try:
            # Stream tokens from the unified route
            stream = stream_chat("/api/v1/chat", payload=payload)
            stream_received = False

            for token_chunk in stream:
                stream_received = True
                try:
                    event_payload = json.loads(token_chunk)
                    if isinstance(event_payload, dict):
                        if "thought" in event_payload:
                            full_thought += event_payload["thought"]
                            thought_container.expander("Agent Reasoning Steps", expanded=True).markdown(full_thought)
                        if "token" in event_payload:
                            full_response += event_payload["token"]
                            response_container.markdown(full_response + "▌")
                        elif "response" in event_payload:
                            full_response += event_payload["response"]
                            response_container.markdown(full_response + "▌")
                        elif "memo" in event_payload:
                            full_response += event_payload["memo"]
                            response_container.markdown(full_response + "▌")
                        if "chart_url" in event_payload:
                            chart_data = event_payload["chart_url"]
                    else:
                        full_response += str(event_payload)
                        response_container.markdown(full_response + "▌")
                except json.JSONDecodeError:
                    full_response += token_chunk
                    response_container.markdown(full_response + "▌")

            # Fallback if the backend returns a non-streaming single response
            if not stream_received or not full_response:
                res = post("/api/v1/chat", json_data=payload)
                if res.status_code == 200:
                    data = res.json()
                    full_response = data.get("response") or data.get("memo") or res.text
                    chart_data = data.get("chart_url")

            response_container.markdown(full_response)
            if chart_data:
                st.image(chart_data, caption="Agent Generated Multimodal Visual")

            st.session_state["messages"].append({
                "role": "assistant",
                "content": full_response,
                "thought": full_thought,
                "chart": chart_data,
            })

        except Exception as err:
            st.error(f"Chat communication error: {err}")