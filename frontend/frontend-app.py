import json
import os
import time
from datetime import datetime
from typing import Optional

import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

# --- Streamlit App Layout ---
st.set_page_config(
    page_title="Milano-AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API base URL
# API_BASE_URL = "http://backend:8000"
API_BASE_URL = os.getenv("API_BASE_URL")

# Initialize session state for all variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "documents" not in st.session_state:
    st.session_state.documents = []
if "session_info" not in st.session_state:
    st.session_state.session_info = None
if "notification" not in st.session_state:
    st.session_state.notification = {
        "type": None,
        "message": None,
        "timestamp": None,
    }
if "processing" not in st.session_state:
    st.session_state.processing = False
if "session_name_input_key" not in st.session_state:
    st.session_state.session_name_input_key = 0
if "uploaded_s3_keys" not in st.session_state:
    st.session_state.uploaded_s3_keys = ""
if "query_input_value" not in st.session_state:
    st.session_state.query_input_value = ""

# --- Custom CSS for enhanced minimalist design ---
st.markdown(
    """
    <style>
    :root {
        --text-primary: #000000;
        --text-secondary: #666666;
        --background-main: #f5ebe0;
        --background-sidebar: #e3d5ca;
        --border-color: #403d39;
        --accent-color: #d5bdaf;
        --gray-toggle: #B0B0B0;
        --gray-default: #D3D3D3;
    }
    html, body, .stApp {
        background-color: var(--background-main);
        color: var(--text-primary);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        display: flex;
        flex-direction: column;
        height: 100vh;
        margin: 0;
        overflow: hidden;
    }
    .main {
        padding: 0;
        max-width: 900px;
        margin: auto;
        flex: 1;
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    .stSidebar {
        background-color: var(--background-sidebar);
        color: var(--text-primary);
        border-right: 1px solid var(--border-color);
        padding: 10px;
    }
    .stSidebar .st-emotion-cache-1na625b,
    .stSidebar .st-emotion-cache-163hr9x,
    .stSidebar .st-emotion-cache-16ajbmm {
        background-color: var(--background-sidebar);
        color: var(--text-primary);
    }
    .stSidebar [data-testid="collapsedControl"] {
        background-color: var(--accent-color) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 5px !important;
        transition: all 0.3s ease !important;
    }
    .stSidebar [data-testid="collapsedControl"]:hover {
        background-color: #A0A0A0 !important;
    }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #F5F5F5;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 12px;
        color: var(--text-primary);
        caret-color: var(--text-primary);
        transition: all 0.3s ease;
    }

    label, .stTextInput label, .stTextArea label {
        color: var(--text-secondary) !important;
    }
    ::placeholder {
        color: var(--text-secondary) !important;
    }

    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: var(--accent-color);
        box-shadow: 0 0 5px rgba(169, 214, 169, 0.3);
    }
    .stButton > button {
        background-color: var(--background-main);
        color: var(--text-primary);
        border: 1px solid var(--text-primary);
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: var(--accent-color);
        color: #FFFFFF;
        border-color: var(--accent-color);
    }

    /* Styling cho nút Gửi truy vấn */
    .chat-input-container .stButton button {
        background-color: #000000 !important;
        border: none !important;
        border-radius: 50% !important;
        width: 45px;
        height: 45px;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 !important;
        font-size: 20px;
        margin-left: 10px;
        transition: all 0.2s ease;
    }
    .chat-input-container .stButton button:hover,
    .chat-input-container .stButton button:focus {
        background-color: #333333 !important;
        color: #FFFFFF !important;
        box-shadow: none !important;
        transform: scale(1.05);
    }
    .chat-input-container .stButton button p {
        color: #FFFFFF !important;
    }

    h1, h2, h3, h4 {
        color: var(--text-primary);
    }
    h1 {
        background: linear-gradient(90deg, #000000 0%, var(--accent-color) 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .stMarkdown, .stText {
        color: var(--text-primary);
    }

    .stExpander {
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
        margin-bottom: 10px;
        background-color: var(--background-sidebar) !important;
    }
    .stExpanderDetails {
        background-color: var(--background-main) !important;
        border-top: 1px solid var(--border-color);
        margin-left: 0 !important;
        padding: 15px !important;
        border-radius: 0 0 8px 8px;
    }
    .stExpanderHeader {
        background-color: var(--background-sidebar) !important;
        border-radius: 8px !important;
    }
    .stExpanderHeader .st-emotion-cache-1c1y31j {
        background-color: transparent !important;
        color: var(--text-primary) !important;
        font-weight: 500;
        padding: 8px;
    }
    .stChatMessage {
        border: 1px solid var(--border-color);
        border-radius: 12px;
        margin-bottom: 20px;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
    }
    .stChatMessage:hover {
        transform: translateY(-2px);
    }
    .st-emotion-cache-1c7y31j > div > p {
        color: var(--text-primary) !important;
        font-size: 16px;
    }
    .initial-input-container {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 100%;
        max-width: 600px;
        text-align: center;
    }
    .chat-history-container {
        flex: 1;
        max-height: calc(100vh - 180px); /* Điều chỉnh để phù hợp với tiêu đề và input */
        overflow-y: auto;
        padding: 20px;
        margin-bottom: 10px;
        min-height: 100px;
    }
    .chat-input-container {
        position: sticky;
        bottom: 0;
        background-color: var(--background-main);
        padding: 15px;
        display: flex;
        align-items: center;
        border-top: 1px solid var(--border-color);
        z-index: 10;
    }

    /* Custom Toast Notification */
    .toast-container {
        position: fixed;
        top: 10%;
        right: 20px;
        z-index: 1000;
    }
    .toast {
        background-color: var(--background-main);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        gap: 10px;
        color: var(--text-primary);
        font-weight: 500;
        opacity: 0;
        animation: fadein 0.5s forwards, fadeout 0.5s 4.5s forwards;
    }
    @keyframes fadein {
        from { top: calc(10% - 20px); opacity: 0; }
        to { top: 10%; opacity: 1; }
    }
    @keyframes fadeout {
        from { top: 10%; opacity: 1; }
        to { top: calc(10% - 20px); opacity: 0; }
    }
    .toast-success {
        border-left: 5px solid green;
    }
    .toast-error {
        border-left: 5px solid red;
    }
    .toast-warning {
        border-left: 5px solid orange;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# --- Utility Functions ---
def set_notification(type: str, message: str):
    """Sets a notification message in the session state."""
    st.session_state.notification = {
        "type": type,
        "message": message,
        "timestamp": time.time(),
    }


def validate_or_create_session(
    session_id: Optional[str] = None, metadata: Optional[dict] = None
):
    """Validates an existing session or creates a new one."""
    try:
        if session_id:
            response = requests.get(
                f"{API_BASE_URL}/agent/session/{session_id}"
            )
            if response.status_code == 200:
                st.session_state.session_id = session_id
                set_notification(
                    "success", f"Đã kết nối phiên có sẵn: {session_id}"
                )
                fetch_session_history(session_id)
                return True
            else:
                payload = {
                    "session_id": session_id,
                    "metadata": metadata or {},
                }
                response = requests.post(
                    f"{API_BASE_URL}/agent/session/create", json=payload
                )
                response.raise_for_status()
                st.session_state.session_id = session_id
                set_notification("success", f"Đã tạo phiên mới: {session_id}")
                st.session_state.chat_history = []
                return True
        else:
            response = requests.post(
                f"{API_BASE_URL}/agent/session/create",
                json={"metadata": metadata or {}},
            )
            response.raise_for_status()
            session_data = response.json()
            st.session_state.session_id = session_data["session_id"]
            set_notification(
                "success",
                f"Đã tạo phiên mới với ID: {st.session_state.session_id}",
            )
            st.session_state.chat_history = []
            return True
    except requests.RequestException as e:
        set_notification("error", f"Lỗi: {str(e)}")
        st.session_state.session_id = None
        return False


def delete_session(session_id: str):
    """Deletes the active session."""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/agent/session/{session_id}"
        )
        response.raise_for_status()
        st.session_state.session_id = None
        st.session_state.chat_history = []
        st.session_state.session_info = None
        set_notification(
            "success", f"Phiên {session_id} đã được xóa thành công"
        )
    except requests.RequestException as e:
        set_notification("error", f"Không thể xóa phiên: {str(e)}")


def fetch_session_history(session_id: str):
    """Fetches the conversation history for a given session and updates session state."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/agent/session/{session_id}/history"
        )
        response.raise_for_status()

        history_data = response.json()
        st.session_state.chat_history = history_data.get("history", [])
        st.session_state.session_info = {
            "session_id": history_data.get("session_id"),
            "pagination": history_data.get("pagination"),
        }
        set_notification("success", "Lịch sử trò chuyện đã được tải lại")
    except requests.RequestException as e:
        set_notification(
            "error", f"Không thể tải lịch sử trò chuyện: {str(e)}"
        )


def query_agent(query: str, session_id: Optional[str]):
    """Sends a query to the agent and gets a response."""
    try:
        payload = {"query": query, "session_id": session_id}
        response = requests.post(f"{API_BASE_URL}/agent/query", json=payload)
        response.raise_for_status()
        result = response.json()
        return result
    except requests.RequestException as e:
        set_notification("error", f"Truy vấn thất bại: {str(e)}")
        return None


def get_health_status():
    """Checks the health of the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/agent/health")
        response.raise_for_status()
        health_data = response.json()
        set_notification("success", "Kiểm tra sức khỏe thành công")
        return health_data
    except requests.RequestException as e:
        set_notification("error", f"Kiểm tra sức khỏe thất bại: {str(e)}")
        return None


def process_documents(s3_keys: list, force_reprocess: bool):
    """Calls the API to process documents from S3."""
    try:
        payload = {"s3_keys": s3_keys, "force_reprocess": force_reprocess}
        response = requests.post(
            f"{API_BASE_URL}/agent/load-documents", json=payload
        )
        response.raise_for_status()
        result = response.json()
        set_notification(
            "success",
            f"Đã gửi yêu cầu xử lý tài liệu: {result['processed_documents']}",
        )
        return result
    except requests.RequestException as e:
        set_notification(
            "error", f"Gửi yêu cầu xử lý tài liệu thất bại: {str(e)}"
        )


def list_documents(prefix: Optional[str] = None):
    """Calls the API to get a list of processed documents with an optional prefix."""
    try:
        params = {}
        if prefix:
            params["prefix"] = prefix
        response = requests.get(
            f"{API_BASE_URL}/agent/documents", params=params
        )
        response.raise_for_status()
        st.session_state.documents = response.json()
        set_notification("success", "Đã tải danh sách tài liệu thành công")
    except requests.RequestException as e:
        set_notification(
            "error", f"Không thể tải danh sách tài liệu: {str(e)}"
        )


# Sidebar
with st.sidebar:
    st.markdown(
        "<h3 style='margin-bottom: 15px;'>Quản lý Phiên</h3>",
        unsafe_allow_html=True,
    )
    with st.expander("Tạo Phiên", expanded=False):
        st.markdown(
            "<p style='color: var(--text-primary);'>Bạn có thể nhập một tên riêng hoặc để trống để tạo phiên ngẫu nhiên.</p>",
            unsafe_allow_html=True,
        )
        session_id_input = st.text_input(
            "Tên phiên (tùy chọn)", placeholder="Ví dụ: MySession"
        )
        metadata_input = st.text_area(
            "Metadata (JSON, tùy chọn)", placeholder='{"key": "value"}'
        )
        if st.button("Tạo", key="create_session_btn"):
            metadata = None
            if metadata_input:
                try:
                    metadata = json.loads(metadata_input)
                except json.JSONDecodeError:
                    set_notification(
                        "error",
                        "Metadata không hợp lệ. Vui lòng nhập JSON đúng.",
                    )
                    metadata = None
            if metadata is not None:
                validate_or_create_session(
                    session_id_input if session_id_input else None, metadata
                )
                st.rerun()

    if st.session_state.session_id:
        st.markdown(
            f'<div style="font-size: 16px; font-weight: 500; margin-bottom: 15px; color: var(--text-primary);">Phiên hoạt động: {st.session_state.session_id}</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Tác vụ Phiên", expanded=False):
            st.button(
                "Tải lại lịch sử",
                on_click=fetch_session_history,
                args=(st.session_state.session_id,),
                key="refresh_history_btn",
            )
            st.button(
                "Xóa phiên hiện tại",
                on_click=delete_session,
                args=(st.session_state.session_id,),
                key="delete_session_btn",
            )
            st.button(
                "Xóa lịch sử trò chuyện",
                on_click=lambda: st.session_state.update(chat_history=[]),
                key="clear_history_btn",
            )
            if st.session_state.session_info:
                st.write("**Thông tin Phiên**:")
                st.json(st.session_state.session_info)

    st.markdown(
        "<h3 style='margin-bottom: 15px;'>Quản lý Tài liệu</h3>",
        unsafe_allow_html=True,
    )
    with st.expander("Tải lên Tài liệu", expanded=False):
        st.markdown(
            "<p style='color: var(--text-primary);'><b>Nhập các khóa S3, phân cách bằng dấu phẩy.</b></p>",
            unsafe_allow_html=True,
        )
        uploaded_s3_keys = st.text_area(
            "Khóa S3",
            value=st.session_state.uploaded_s3_keys,
            key="s3_keys_input",
        )
        force_reprocess = st.checkbox("Buộc xử lý lại", value=False)
        if st.button("Xử lý Tài liệu"):
            s3_keys_list = [
                k.strip() for k in uploaded_s3_keys.split(",") if k.strip()
            ]
            if s3_keys_list:
                processed_result = process_documents(
                    s3_keys_list, force_reprocess
                )
                st.write(
                    f"Tài liệu: {', '.join([s3key.split('/')[1] for s3key in s3_keys_list])}"
                )
                st.write(f"Tổng: {processed_result['processed_documents']}")
                st.write(
                    f"Thời gian: {round(processed_result['processing_time'],2)}s"
                )
            else:
                set_notification("error", "Vui lòng nhập ít nhất một khóa S3.")

        st.markdown(
            "<p style='color: var(--text-primary);'><b>Lọc và tải danh sách tài liệu đã xử lý.</b></p>",
            unsafe_allow_html=True,
        )
        document_prefix = st.text_input(
            "Lọc theo tiền tố prefix",
            placeholder="Ví dụ: báo-cáo-tài-chính, tài-liệu-rag,...",
        )
        if st.button("Tải danh sách Tài liệu"):
            list_documents(prefix=document_prefix)

        if st.session_state.documents:
            st.write(f"S3-Prefix: {document_prefix}:")
            for doc in st.session_state.documents["documents"]:
                st.write(
                    f"- Tên: {doc['key'].split('/')[1]} - Kích thước: {doc['size']} - Loại: {doc['content_type']}"
                )
            st.write(f"Tổng: {st.session_state.documents['total_count']}")

    st.markdown(
        "<h3 style='margin-bottom: 15px;'>Trạng thái Hệ thống</h3>",
        unsafe_allow_html=True,
    )
    with st.expander("Kiểm tra Sức khỏe", expanded=False):
        if st.button("Kiểm tra", key="check_health_btn"):
            health_data = get_health_status()
            if health_data:
                status_text = (
                    "Healthy"
                    if health_data["status"] == "healthy"
                    else "Unhealthy"
                )
                st.markdown(
                    f"<p style='color: var(--text-primary);'><b>Trạng thái</b>: {status_text}</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<p style='color: var(--text-primary);'><b>Phiên bản</b>: {health_data['version']}</p>",
                    unsafe_allow_html=True,
                )
                if "uptime_seconds" in health_data:
                    st.markdown(
                        f"<p style='color: var(--text-primary);'><b>Thời gian hoạt động</b>: {health_data['uptime_seconds']:.2f}s</p>",
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f"<p style='color: var(--text-primary);'><b>Chi tiết dịch vụ:</b>: {health_data['version']}</p>",
                    unsafe_allow_html=True,
                )
                st.json(health_data["services"])

# Main chat interface
st.title("✨ Milano AI Agent ✨")
st.markdown(
    "<h4 style='text-align: center; color: var(--text-secondary);'>Trợ lý AI giúp phân tích thị trường chứng khoán Việt Nam.</h4>",
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

# Custom toast notification
if st.session_state.notification["message"] and (
    time.time() - st.session_state.notification["timestamp"] < 5
):
    toast_class = f"toast toast-{st.session_state.notification['type']}"
    st.markdown(
        f"""
        <div class="toast-container">
            <div class="{toast_class}">
                {st.session_state.notification['message']}
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

# Check session state and display appropriate UI
if not st.session_state.session_id:
    with st.container():
        st.markdown(
            "<p style='text-align: center; color: var(--text-secondary);'>Nhập tên phiên để tiếp tục hoặc để trống để bắt đầu một phiên mới.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='initial-input-container'>", unsafe_allow_html=True
        )
        initial_session_name = st.text_input(
            "Tên phiên",
            placeholder="Nhập tên phiên của bạn (tùy chọn)",
            label_visibility="collapsed",
        )
        if st.button("Bắt đầu trò chuyện"):
            validate_or_create_session(
                initial_session_name if initial_session_name else None
            )
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
else:
    # Main container with flex layout
    with st.container():
        st.markdown('<div class="main">', unsafe_allow_html=True)

        # Chat history container
        chat_container = st.container()
        with chat_container:
            st.markdown(
                '<div class="chat-history-container">', unsafe_allow_html=True
            )
            for message in st.session_state.chat_history:
                if message["type"] == "human":
                    with st.chat_message("user", avatar="🧑‍💼"):
                        st.markdown(message["content"])
                elif message["type"] == "ai":
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(message["content"])
                        if message.get("tools_used"):
                            st.markdown(
                                f"**Công cụ đã dùng**: {', '.join(message['tools_used'])}"
                            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Auto-scroll to the latest message
        components.html(
            """
            <script>
                function scrollToBottom() {
                    const chatContainer = document.querySelector('.chat-history-container');
                    if (chatContainer) {
                        console.log('Scrolling to bottom. Container height:', chatContainer.scrollHeight);
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    } else {
                        console.log('Chat container not found');
                    }
                }

                // Chạy sau khi DOM được tải
                document.addEventListener('DOMContentLoaded', scrollToBottom);

                // Chạy lại sau một khoảng thời gian ngắn để xử lý render động của Streamlit
                setTimeout(scrollToBottom, 100);
                setTimeout(scrollToBottom, 500);

                // Theo dõi thay đổi DOM
                const chatContainer = document.querySelector('.chat-history-container');
                if (chatContainer) {
                    const observer = new MutationObserver(scrollToBottom);
                    observer.observe(chatContainer, { childList: true, subtree: true });
                }
            </script>
        """,
            height=0,
        )

        # User input at the bottom
        with st.container():
            st.markdown(
                '<div class="chat-input-container">', unsafe_allow_html=True
            )
            col1, col2 = st.columns([10, 1])
            with col1:
                query_input = st.text_input(
                    "Đặt câu hỏi của bạn",
                    placeholder="Ví dụ: Phân tích hiệu suất cổ phiếu của VNM trong quý này",
                    key="query_input_widget",
                    value=st.session_state.query_input_value,
                    label_visibility="collapsed",
                )
            with col2:
                if st.button("➤", key="send_query", help="Gửi truy vấn"):
                    if query_input:
                        st.session_state.processing = True
                        st.session_state.chat_history.append(
                            {
                                "type": "human",
                                "content": query_input,
                                "timestamp": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                            }
                        )

                        result = query_agent(
                            query_input, st.session_state.session_id
                        )

                        if result:
                            st.session_state.chat_history.append(
                                {
                                    "type": "ai",
                                    "content": result["answer"],
                                    "timestamp": datetime.now().strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    ),
                                    "tools_used": result.get("tools_used", []),
                                    "success": result["success"],
                                }
                            )

                        st.session_state.query_input_value = ""
                        st.session_state.processing = False
                        st.rerun()
                    else:
                        set_notification("warning", "Vui lòng nhập câu hỏi!")
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
