import html
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

from config import CONFIDENCE_THRESHOLD

API_BASE = "http://localhost:8000"
PAGE_SIZE = 20

st.set_page_config(page_title="Receipt Review", layout="wide", page_icon="🧾")

# --- Custom CSS ---
st.markdown(
    """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    [data-testid="stSidebar"] {
        background: #f8f9fa;
        border-right: 1px solid #e8ecf0;
    }
    [data-testid="stSidebar"] h1 {
        font-size: 1.7rem !important;
        font-weight: 700 !important;
        color: #1a1a2e !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCaptionContainer {
        font-size: 1.05rem !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        font-size: 1.1rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        font-size: 1.05rem !important;
    }

    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.15s !important;
    }

    .status-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .badge-review { background: #fff3e0; color: #e65100; }
    .badge-approved { background: #e8f5e9; color: #2e7d32; }
    .badge-verified { background: #e3f2fd; color: #1565c0; }
    .badge-pending { background: #f5f5f5; color: #757575; }

    .conf-bar {
        height: 5px;
        border-radius: 3px;
        background: #e8ecf0;
        margin-top: 4px;
    }
    .conf-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
    .conf-high { background: #4caf50; }
    .conf-mid { background: #ff9800; }
    .conf-low { background: #f44336; }

    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

STATUS_BADGE = {
    "review": "badge-review",
    "approved": "badge-approved",
    "verified": "badge-verified",
    "pending": "badge-pending",
}


def _fmt_date(ts):
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y · %I:%M %p")
    except Exception:
        return ts


def _api_get(path, **kwargs):
    try:
        resp = requests.get(f"{API_BASE}{path}", timeout=kwargs.pop("timeout", 10), **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def _api_put(path, json_data):
    try:
        resp = requests.put(f"{API_BASE}{path}", json=json_data, timeout=10)
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
        return False


def _api_upload(file):
    try:
        resp = requests.post(
            f"{API_BASE}/documents/upload",
            files={"file": (file.name, file.getvalue(), file.type)},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"Upload failed: {e}")
        return None


@st.cache_data(ttl=10, show_spinner=False)
def _fetch_documents(status_filter: str, offset: int):
    params = {"limit": PAGE_SIZE, "offset": offset}
    if status_filter != "All":
        params["status"] = status_filter
    return _api_get("/documents", params=params)


@st.cache_data(ttl=10, show_spinner=False)
def _fetch_statistics():
    return _api_get("/statistics")


def _invalidate_cache():
    _fetch_documents.clear()
    _fetch_statistics.clear()


def _load_more():
    st.session_state._load_count += PAGE_SIZE


@st.fragment
def _render_doc(doc, auto_open=False):
    extracted = doc.get("extracted_data") or {}
    conf = doc.get("overall_confidence") or 0.0
    status = doc.get("status", "pending")
    badge_class = STATUS_BADGE.get(status, "")

    if conf >= CONFIDENCE_THRESHOLD:
        conf_class = "conf-high"
    elif conf >= CONFIDENCE_THRESHOLD * 0.8:
        conf_class = "conf-mid"
    else:
        conf_class = "conf-low"
    conf_pct = int(conf * 100)

    with st.container(border=True):
        st.markdown(
            f"""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem">
            <div>
                <span style="font-weight:700;font-size:1rem;color:#1a1a2e">#{doc['id']}</span>
                <span style="color:#aaa;margin:0 8px">·</span>
                <span style="color:#555;font-size:0.9rem">{html.escape(doc['filename'])}</span>
            </div>
            <div style="display:flex;align-items:center;gap:10px">
                <span class="status-badge {badge_class}">{status}</span>
                <span style="font-size:0.78rem;color:#999">{html.escape(_fmt_date(doc.get('created_at', '')))}</span>
            </div>
        </div>
        <div style="margin-bottom:0.75rem">
            <span style="font-size:0.78rem;color:#888">Confidence</span>
            <span style="font-weight:600;font-size:0.85rem;color:#333;margin-left:6px">{conf_pct}%</span>
            <div class="conf-bar"><div class="conf-fill {conf_class}" style="width:{conf_pct}%"></div></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        with st.expander("Review details", expanded=auto_open):
            col_img, col_data = st.columns([1, 2])
            with col_img:
                stored_name = Path(doc.get("upload_path", "")).name
                if stored_name:
                    st.image(
                        f"{API_BASE}/uploads/{stored_name}",
                        width="stretch",
                    )

            with col_data:
                with st.form(key=f"form_{doc['id']}"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        edited_total = st.number_input(
                            "Total",
                            value=extracted.get("total") or 0.0,
                            format="%.2f",
                            key=f"total_{doc['id']}",
                        )
                    with c2:
                        edited_subtotal = st.number_input(
                            "Subtotal",
                            value=extracted.get("subtotal") or 0.0,
                            format="%.2f",
                            key=f"subtotal_{doc['id']}",
                        )
                    with c3:
                        edited_tax = st.number_input(
                            "Tax",
                            value=extracted.get("tax") or 0.0,
                            format="%.2f",
                            key=f"tax_{doc['id']}",
                        )

                    items_data = extracted.get("items", [])
                    edited_items = st.data_editor(
                        items_data,
                        column_config={
                            "name": st.column_config.TextColumn("Name", width="medium"),
                            "price": st.column_config.NumberColumn("Price", format="%.2f"),
                        },
                        num_rows="dynamic",
                        key=f"items_{doc['id']}",
                        width="stretch",
                    )

                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        submitted = st.form_submit_button(
                            "✓ Submit Corrections",
                            width="stretch",
                            type="primary",
                        )
                    with col_btn2:
                        approved = st.form_submit_button(
                            "Approve as-is",
                            width="stretch",
                        )

                    if submitted:
                        corrected = {
                            "total": edited_total,
                            "subtotal": edited_subtotal,
                            "tax": edited_tax,
                            "items": edited_items,
                        }
                        if _api_put(f"/documents/{doc['id']}/verify", {"corrected_data": corrected}):
                            st.toast("Corrections submitted!", icon="✅")
                            _invalidate_cache()
                            st.rerun()

                    if approved:
                        if _api_put(f"/documents/{doc['id']}/verify", {"corrected_data": extracted}):
                            st.toast("Approved as-is!", icon="✅")
                            _invalidate_cache()
                            st.rerun()


# --- Sidebar ---
with st.sidebar:
    st.title("🧾 Receipt Review")

    st.caption("Upload")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "tiff", "bmp"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        if st.button("Upload Receipt", width="stretch"):
            with st.spinner("Processing receipt..."):
                result = _api_upload(uploaded_file)
            if result:
                st.toast("Receipt uploaded!", icon="✅")
                st.session_state.status_filter = result["status"]
                st.session_state.auto_open_doc_id = result["id"]
                _invalidate_cache()
                st.rerun()

    st.divider()

    st.caption("Filter by status")
    status_filter = st.radio(
        "Status",
        ["All", "review", "approved", "verified"],
        index=1,
        horizontal=True,
        label_visibility="collapsed",
        key="status_filter",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Refresh", width="stretch"):
            _invalidate_cache()
            st.rerun()
    with col_b:
        st.checkbox("📊 Stats", value=True, key="show_stats")

    if st.session_state.show_stats:
        st.divider()
        stats = _fetch_statistics()
        if stats is None:
            st.error("Backend not running.\n\n`uvicorn main:app`")
        else:
            total = stats["total_documents"]
            by_status = stats.get("by_status", {})
            avg = stats.get("average_confidence")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Total", total)
            with c2:
                st.metric("Avg Conf", f"{avg:.1%}" if avg is not None else "—")

            for s in ["review", "approved", "verified"]:
                count = by_status.get(s, 0)
                badge = STATUS_BADGE.get(s, "")
                st.markdown(
                    f'<span class="status-badge {badge}">{s}</span>&nbsp;&nbsp;{count}',
                    unsafe_allow_html=True,
                )

# --- Main area ---
status_label = status_filter if status_filter != "All" else "All"

if st.session_state.get("_page_filter") != status_filter:
    st.session_state._page_filter = status_filter
    st.session_state._load_count = PAGE_SIZE

with st.spinner("Loading..."):
    docs = None
    has_more = False
    load_count = st.session_state._load_count
    for offset in range(0, load_count, PAGE_SIZE):
        batch = _fetch_documents(status_filter, offset)
        if batch is None:
            docs = None
            break
        if docs is None:
            docs = []
        docs.extend(batch)
        has_more = len(batch) == PAGE_SIZE
        if len(batch) < PAGE_SIZE:
            break

if docs is None:
    st.error("Backend not running. Start with: `uvicorn main:app`")
elif not docs:
    st.title(f"Documents · {status_label}")
    st.markdown(
        '<div class="empty-state">'
        '<p style="font-size:3rem;margin:0 0 0.5rem">📭</p>'
        '<p style="font-size:1.1rem;font-weight:500;color:#555">No documents found</p>'
        '<p style="color:#999;font-size:0.9rem">Upload a receipt in the sidebar to get started</p>'
        "</div>",
        unsafe_allow_html=True,
    )
else:
    st.title(f"Documents · {status_label} ({len(docs)})")

    auto_open_id = st.session_state.pop("auto_open_doc_id", None)
    for doc in docs:
        _render_doc(doc, auto_open=doc["id"] == auto_open_id)

    if has_more:
        st.button(
            "⬇ Load more",
            on_click=_load_more,
            width="stretch",
        )
