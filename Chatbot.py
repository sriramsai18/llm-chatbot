import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

st.set_page_config(
    page_title="TurboChat AI",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Orbitron:wght@700;900&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117 !important;
    color: #c9d1d9 !important;
    font-family: 'Rajdhani', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d !important;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
footer, #MainMenu { visibility: hidden; }

/* ── Header ── */
.chat-header {
    text-align: center;
    padding: 20px 0 10px;
    border-bottom: 1px solid #21262d;
    margin-bottom: 20px;
}
.chat-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 900;
    color: #fff;
    letter-spacing: 2px;
}
.chat-title span { color: #58a6ff; }
.chat-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #8b949e;
    letter-spacing: 3px;
    margin-top: 4px;
}

/* ── Chat bubbles ── */
.chat-wrap {
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 10px 0;
}
.msg-user {
    display: flex;
    justify-content: flex-end;
    animation: slideInRight 0.3s ease both;
}
.msg-ai {
    display: flex;
    justify-content: flex-start;
    animation: slideInLeft 0.3s ease both;
}
.bubble-user {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: #fff;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 78%;
    font-size: 0.97rem;
    line-height: 1.6;
    box-shadow: 0 4px 15px rgba(31,111,235,0.25);
}
.bubble-ai {
    background: #161b22;
    color: #c9d1d9;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 4px;
    max-width: 78%;
    font-size: 0.97rem;
    line-height: 1.7;
    border: 1px solid #30363d;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    position: relative;
}
.bubble-ai::before {
    content: '⚡';
    position: absolute;
    top: -10px; left: 12px;
    font-size: 0.7rem;
    background: #0d1117;
    padding: 0 4px;
    border-radius: 50%;
}
.msg-meta {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    color: #8b949e;
    margin-top: 4px;
    padding: 0 6px;
}

/* ── Typing animation ── */
.typing-indicator {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 14px 18px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 18px 18px 18px 4px;
    width: fit-content;
    animation: slideInLeft 0.3s ease both;
}
.typing-dot {
    width: 7px; height: 7px;
    background: #58a6ff;
    border-radius: 50%;
    animation: bounce 1.2s ease-in-out infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%,60%,100% { transform: translateY(0); opacity:0.4; }
    30%          { transform: translateY(-8px); opacity:1; }
}

/* ── Stats bar ── */
.stats-bar {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    padding: 8px 12px;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    margin-bottom: 12px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    color: #8b949e;
}
.stat-item { display: flex; align-items: center; gap: 5px; }
.stat-val  { color: #58a6ff; font-weight: 600; }

/* ── Input area ── */
.stTextInput > div > div > input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.97rem !important;
    padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 12px rgba(88,166,255,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #21262d !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    background: #30363d !important;
    border-color: #58a6ff !important;
    color: #58a6ff !important;
}

/* ── Send button ── */
.send-btn > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 0 16px rgba(31,111,235,0.3) !important;
}
.send-btn > button:hover {
    box-shadow: 0 0 28px rgba(31,111,235,0.55) !important;
    transform: translateY(-2px) !important;
    color: #fff !important;
}

/* ── Sidebar inputs ── */
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] input {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
}

/* ── Welcome card ── */
.welcome-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 28px;
    text-align: center;
    margin: 20px 0;
    animation: fadeInUp 0.6s ease both;
}
.welcome-icon { font-size: 2.5rem; margin-bottom: 10px; }
.welcome-title {
    font-family: 'Orbitron', monospace;
    font-size: 1rem; color: #fff; margin-bottom: 8px;
}
.welcome-sub { font-size: 0.9rem; color: #8b949e; line-height: 1.6; }
.tip-chip {
    display: inline-block;
    background: rgba(88,166,255,0.1);
    border: 1px solid rgba(88,166,255,0.3);
    color: #58a6ff;
    padding: 4px 12px; border-radius: 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem; margin: 4px 3px;
}

/* ── Animations ── */
@keyframes slideInRight {
    from { opacity:0; transform: translateX(20px); }
    to   { opacity:1; transform: translateX(0); }
}
@keyframes slideInLeft {
    from { opacity:0; transform: translateX(-20px); }
    to   { opacity:1; transform: translateX(0); }
}
@keyframes fadeInUp {
    from { opacity:0; transform: translateY(16px); }
    to   { opacity:1; transform: translateY(0); }
}

/* scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# ─── MODEL LOAD ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_name = "TinyLlama/TinyLlama-1.1B-chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model

# ─── SESSION STATE ────────────────────────────────────────────────────────────
if "messages"     not in st.session_state: st.session_state.messages     = []
if "total_tokens" not in st.session_state: st.session_state.total_tokens = 0
if "total_time"   not in st.session_state: st.session_state.total_time   = 0.0
if "msg_count"    not in st.session_state: st.session_state.msg_count    = 0

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 20px;">
        <div style="font-family:'Orbitron',monospace;font-size:1rem;color:#fff;letter-spacing:2px;">⚡ TURBO<span style="color:#58a6ff;">CHAT</span></div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#8b949e;letter-spacing:3px;margin-top:4px;">// TinyLlama 1.1B</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**🎭 SYSTEM PROMPT**")
    system_prompt = st.text_area(
        "",
        value="You are TurboChat, a helpful, smart and concise AI assistant. Give clear, accurate answers.",
        height=100,
        label_visibility="collapsed",
        key="sys_prompt"
    )

    st.markdown("---")
    st.markdown("**⚙️ GENERATION SETTINGS**")

    max_tokens = st.slider("Max Tokens", 50, 300, 120, 10)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.6, 0.1,
                            help="Higher = more creative, Lower = more focused")
    top_p = st.slider("Top-P", 0.1, 1.0, 0.9, 0.05,
                      help="Nucleus sampling threshold")

    st.markdown("---")
    st.markdown("**📊 SESSION STATS**")
    st.markdown(f"""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#8b949e;line-height:2;">
        💬 Messages &nbsp;&nbsp;<span style="color:#58a6ff;">{st.session_state.msg_count}</span><br>
        🔤 Tokens &nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#58a6ff;">{st.session_state.total_tokens}</span><br>
        ⏱️ Total time &nbsp;<span style="color:#58a6ff;">{st.session_state.total_time:.1f}s</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ CLEAR CHAT", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.total_tokens = 0
        st.session_state.total_time   = 0.0
        st.session_state.msg_count    = 0
        st.rerun()

    st.markdown("""
    <div style="text-align:center;margin-top:16px;font-family:'Share Tech Mono',monospace;font-size:0.62rem;color:#484f58;">
        Built by <a href="https://github.com/sriramsai18" style="color:#58a6ff;text-decoration:none;">Sriram Sai</a>
    </div>
    """, unsafe_allow_html=True)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
    <div class="chat-title">TURBO<span>CHAT</span> ⚡</div>
    <div class="chat-sub">// POWERED BY TINYLLAMA 1.1B · LOCAL LLM</div>
</div>
""", unsafe_allow_html=True)

# ─── CHAT HISTORY ─────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-icon">⚡</div>
        <div class="welcome-title">TURBOCHAT IS READY</div>
        <div class="welcome-sub">
            Ask me anything — code, concepts, advice, or just chat.<br>
            Powered by TinyLlama 1.1B running locally.
        </div>
        <br>
        <span class="tip-chip">💡 Try: Explain neural networks</span>
        <span class="tip-chip">💡 Try: Write a Python function</span>
        <span class="tip-chip">💡 Try: What is RAG?</span>
    </div>
    """, unsafe_allow_html=True)
else:
    chat_html = '<div class="chat-wrap">'
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_html += f"""
            <div class="msg-user">
                <div>
                    <div class="bubble-user">{msg["content"]}</div>
                    <div class="msg-meta" style="text-align:right;">YOU · {msg.get("time","")}</div>
                </div>
            </div>"""
        else:
            chat_html += f"""
            <div class="msg-ai">
                <div>
                    <div class="bubble-ai">{msg["content"]}</div>
                    <div class="msg-meta">⚡ TURBOCHAT · {msg.get("time","")} · {msg.get("tokens","")} tokens · {msg.get("elapsed","")}s</div>
                </div>
            </div>"""
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

# ─── INPUT ROW ────────────────────────────────────────────────────────────────
st.write("")
col_input, col_btn = st.columns((5, 1))

with col_input:
    user_input = st.text_input(
        "",
        placeholder="// type your message and press Enter or click Send...",
        label_visibility="collapsed",
        key="user_input"
    )

with col_btn:
    st.markdown('<div class="send-btn">', unsafe_allow_html=True)
    send = st.button("▶ SEND", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── GENERATE ─────────────────────────────────────────────────────────────────
if (send or user_input) and user_input.strip():

    # save user message
    ts = time.strftime("%H:%M")
    st.session_state.messages.append({
        "role": "user",
        "content": user_input.strip(),
        "time": ts
    })
    st.session_state.msg_count += 1

    # show typing indicator
    typing_slot = st.empty()
    typing_slot.markdown("""
    <div class="msg-ai">
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # generate response
    try:
        tokenizer, model = load_model()

        prompt = f"""<|system|>
{system_prompt}
<|user|>
{user_input.strip()}
<|assistant|>
"""
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")

        start_time = time.time()

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        elapsed      = round(time.time() - start_time, 1)
        decoded      = tokenizer.decode(output[0], skip_special_tokens=True)
        response     = decoded.split("<|assistant|>")[-1].strip()
        token_count  = output.shape[-1] - input_ids.shape[-1]

        # update stats
        st.session_state.total_tokens += token_count
        st.session_state.total_time   += elapsed
        st.session_state.msg_count    += 1

        # save AI message
        st.session_state.messages.append({
            "role":    "assistant",
            "content": response,
            "time":    time.strftime("%H:%M"),
            "tokens":  token_count,
            "elapsed": elapsed
        })

    except Exception as e:
        st.session_state.messages.append({
            "role":    "assistant",
            "content": f"⚠️ Error: {str(e)}",
            "time":    time.strftime("%H:%M"),
            "tokens":  0,
            "elapsed": 0
        })

    typing_slot.empty()
    st.rerun()
