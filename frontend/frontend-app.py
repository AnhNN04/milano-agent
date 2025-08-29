import streamlit as st
import requests
from datetime import datetime
from typing import Optional

# API base URL
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Function to create a new session
def create_session(custom_session_id: Optional[str] = None):
    try:
        payload = {}
        if custom_session_id:
            payload["metadata"] = {"custom_session_id": custom_session_id}
        response = requests.post(f"{API_BASE_URL}/agent/session/create", json=payload)
        response.raise_for_status()
        session_data = response.json()
        st.session_state.session_id = session_data["session_id"]
        st.success(f"✅ Created new session: {custom_session_id or st.session_state.session_id}")
        st.session_state.chat_history = []
    except requests.RequestException as e:
        st.error(f"❌ Failed to create session: {str(e)}")

# Function to delete a session
def delete_session(session_id: str):
    try:
        response = requests.delete(f"{API_BASE_URL}/agent/session/{session_id}")
        response.raise_for_status()
        st.session_state.session_id = None
        st.session_state.chat_history = []
        st.success(f"✅ Session {session_id} deleted successfully")
    except requests.RequestException as e:
        st.error(f"❌ Failed to delete session: {str(e)}")

# Function to fetch session history
def fetch_session_history(session_id: str, limit: int = 50, offset: int = 0):
    try:
        response = requests.get(f"{API_BASE_URL}/agent/session/{session_id}/history?limit={limit}&offset={offset}")
        response.raise_for_status()
        history_data = response.json()
        st.session_state.chat_history = [
            {
                "query": item["query"],
                "response": item["response"],
                "timestamp": item["timestamp"],
                "tools_used": item["tools_used"],
                "success": item["success"]
            } for item in history_data["history"]
        ]
        st.success("✅ Chat history refreshed")
    except requests.RequestException as e:
        st.error(f"❌ Failed to fetch session history: {str(e)}")

# Function to query the agent
def query_agent(query: str, session_id: Optional[str]):
    try:
        payload = {"query": query, "session_id": session_id}
        response = requests.post(f"{API_BASE_URL}/agent/query", json=payload)
        response.raise_for_status()
        result = response.json()
        return result
    except requests.RequestException as e:
        st.error(f"❌ Query failed: {str(e)}")
        return None

# Function to get health status
def get_health_status():
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"❌ Failed to fetch health status: {str(e)}")
        return None

# Streamlit app configuration
st.set_page_config(
    page_title="Milano-AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Grok-inspired CSS with white chat text, justified content, and right-aligned Send Query button
st.markdown(
    """
    <style>
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --background: #f8fafc;
        --card-bg: #ffffff;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --success: #10b981;
        --error: #ef4444;
        --info: #3b82f6;
        --button-padding: 6px 8px;
    }
    /* General styling */
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: var(--background);
        color: var(--text-primary);
    }
    .main {
        background-color: var(--card-bg);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        max-width: 800px;
        margin: 24px auto;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border-radius: 8px;
        padding: var(--button-padding);
        font-size: 14px;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
        width: auto;
        min-width: fit-content;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }
    .stButton>button:hover {
        background: var(--primary-dark);
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    /* Button container */
    .button-container {
        display: flex;
        justify-content: space-between;
        margin-top: 8px;
    }
    .button-container .stColumns > div {
        display: flex;
        justify-content: space-between;
        width: 100%;
    }
    .button-container .stColumns > div > div:last-child {
        display: flex;
        justify-content: flex-end;
        margin-left: auto;
    }
    /* Text Input */
    .stTextInput>div>div>input {
        background-color: var(--card-bg);
        border: 2px solid #d1d5db;
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        color: var(--text-primary) !important;
        transition: all 0.3s ease;
        height: 44px;
    }
    .stTextInput>div>div>input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15);
        outline: none;
    }
    .stTextInput::placeholder {
        color: var(--text-secondary) !important;
        opacity: 0.8;
    }
    /* Messages and Alerts */
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.1);
        border: 2px solid var(--success);
        border-radius: 8px;
        padding: 12px;
        font-weight: 500;
        text-align: center;
    }
    .css-1lcbmhc .stSuccess {
        width: 100%;
        box-sizing: border-box;
    }
    .stError {
        background-color: rgba(239, 68, 68, 0.1);
        border: 2px solid var(--error);
        border-radius: 8px;
        padding: 12px;
        font-weight: 500;
        text-align: center;
    }
    .stInfo {
        background-color: rgba(59, 130, 246, 0.1);
        border: 2px solid var(--info);
        border-radius: 8px;
        padding: 12px;
        font-weight: 500;
        text-align: center;
    }
    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 16px;
        padding: 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .user-message {
        background: linear-gradient(145deg, #e0e7ff 0%, #c7d2fe 100%);
        border-left: 4px solid var(--primary);
    }
    .agent-message {
        background: linear-gradient(145deg, #f9fafb 0%, #e5e7eb 100%);
        border-left: 4px solid var(--text-secondary);
    }
    .stChatMessage p {
        margin: 0;
        font-size: 15px;
        line-height: 1.5;
        color: white !important;
    }
    /* Sidebar */
    .css-1lcbmhc {
        background-color: var(--card-bg);
        border-right: 2px solid #e5e7eb;
        padding: 16px;
        box-shadow: 2px 0 10px rgba(0,0,0,0.06);
    }
    .stExpander {
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        background-color: var(--card-bg);
        padding: 12px;
        transition: all 0.3s ease;
    }
    .stExpander:hover {
        background-color: #e0e7ff;
        color: var(--text-primary);
    }
    .stExpander p, .stExpander div {
        color: var(--text-primary) !important;
    }
    /* Active Session */
    .active-session {
        color: #ffffff;
        background-color: var(--primary-dark);
        padding: 8px 12px;
        border-radius: 8px;
        font-weight: 500;
        text-align: center;
        display: inline-block;
        width: 100%;
        margin-bottom: 10px;
    }
    /* Typography */
    h1 {
        font-size: 32px;
        font-weight: 700;
        margin: 16px 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        text-align: center;
    }
    h2 {
        font-size: 24px;
        font-weight: 600;
        margin: 12px 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        text-align: center;
    }
    .stMarkdown {
        line-height: 1.6;
        font-size: 15px;
        }
    .stMarkdown p {
        margin: 8px 0;
    }
    /* Container */
    .stContainer {
        padding: 24px;
        border-radius: 12px;
    }
    /* Divider */
    .stDivider {
        background-color: #e5e7eb;
        height: 2px;
        margin: 16px 0;
    }
    /* Sidebar columns */
    .stColumns > div {
        display: flex;
        gap: 8px;
    }
    .stColumns > div > div {
        flex: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App header
st.title("✨Milano Agent✨")
st.subheader("Milano-I'm an AI-powered assistant for analyzing the Vietnamese stock market.")

# Sidebar for session management and health status
with st.sidebar:
    st.header("Session Management", divider=True)
    custom_session_id = st.text_input(
        "Custom Session ID (optional)",
        placeholder="Enter your preferred session ID",
        key="custom_session_id"
    )
    col1, col2 = st.columns(2, gap="small")
    with col1:
        if st.button("Create Session", key="create_session"):
            create_session(custom_session_id if custom_session_id else None)
    with col2:
        if st.session_state.session_id and st.button("Delete Session", key="delete_session"):
            delete_session(st.session_state.session_id)

    if st.session_state.session_id:
        st.markdown(f'<div class="active-session">Active Session: {st.session_state.session_id}</div>', unsafe_allow_html=True)
        if st.button("Refresh History", key="refresh_history"):
            fetch_session_history(st.session_state.session_id)

    st.header("System Status", divider=True)
    if st.button("Check Health", key="check_health"):
        health_data = get_health_status()
        if health_data:
            status_icon = "✅" if health_data['status'] == 'healthy' else "⚠️" if health_data['status'] == 'degraded' else "❌"
            st.markdown(f"**Status**: {health_data['status'].capitalize()} {status_icon}", unsafe_allow_html=True)
            st.markdown(f"**Version**: {health_data['version']}", unsafe_allow_html=True)
            st.markdown(f"**Response Time**: {health_data['response_time_ms']} ms", unsafe_allow_html=True)
            with st.expander("Service Details", expanded=False):
                for service, details in health_data['services'].items():
                    status_icon = "✅" if details['status'] == 'healthy' else "⚠️" if details['status'] == 'degraded' else "❌"
                    st.markdown(f"- **{service}**: {details['status'].capitalize()} {status_icon}", unsafe_allow_html=True)

# Main chat interface
with st.container():
    # Chat history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            with st.chat_message("user", avatar="🧑‍💼"):
                st.markdown(f"**You**: {message['query']}", unsafe_allow_html=True)
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"**Agent**: {message['response']}", unsafe_allow_html=True)
                if message.get("tools_used"):
                    st.markdown(f"**Tools Used**: {', '.join(message['tools_used'])}", unsafe_allow_html=True)
                # st.markdown(f"**Success**: {'✅' if message['success'] else '❌'}", unsafe_allow_html=True)
    else:
        st.info("No conversation history yet. Start by asking a question!")

    # Input and buttons
    query = st.text_input(
        "Ask about Vietnamese stocks",
        placeholder="E.g., Analyze the stock performance of VNM",
        key="query_input"
    )
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1], gap="small")
    with col1:
        if st.button("Clear History", key="clear_history"):
            st.session_state.chat_history = []
            st.success("Conversation history cleared!")
    with col2:
        if st.button("Send Query", key="send_query"):
            if query:
                if not st.session_state.session_id:
                    st.warning("Please create a session first!")
                else:
                    with st.spinner("Analyzing your query..."):
                        result = query_agent(query, st.session_state.session_id)
                        if result:
                            st.session_state.chat_history.append({
                                "query": query,
                                "response": result["answer"],
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "tools_used": result["tools_used"],
                                "success": result["success"]
                            })
                            st.rerun()
            else:
                st.warning("Please enter a query!")
    st.markdown('</div>', unsafe_allow_html=True)