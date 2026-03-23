"""
GenAI QA Requirement Generator
================================
Single-file Streamlit app — upload ONLY this file + requirements.txt to GitHub.
No folders, no submodules, no IDE needed.

HOW TO DEPLOY ON STREAMLIT CLOUD:
──────────────────────────────────
1. Create a NEW repository on github.com (click + → New repository)
2. Upload this file  →  name it exactly:  app.py
3. Upload requirements.txt as well
4. Go to https://share.streamlit.io → New App
5. Select your repo · branch: main · Main file path: app.py
6. Click "Advanced settings" → Secrets → paste:
       GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"
7. Click Deploy ✅

GET A FREE GROQ API KEY (no credit card):
──────────────────────────────────────────
1. Go to https://console.groq.com
2. Sign up (free, instant)
3. API Keys → Create API Key → copy it
"""

# ── Standard library ─────────────────────────────────────────────────────────
import sys
import os
import re
import json
import time
import requests
from datetime import datetime, timezone
from typing import List

# ── Third-party ───────────────────────────────────────────────────────────────
import streamlit as st
from bs4 import BeautifulSoup
from groq import Groq

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — SCRAPER
#  Extracts headings, buttons, inputs, prices from any web page
# ══════════════════════════════════════════════════════════════════════════════

_NOISE = {
    "home", "menu", "toggle", "close", "open", "skip", "back to top",
    "cookie", "privacy policy", "terms", "accept all", "deny",
    "×", "✕", "»", "«", "...", "more", "less",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def scrape_single_page(url: str) -> dict:
    """Scrape one URL and return a structured dict of UI elements."""
    result = {
        "url": url, "title": "", "headings": [],
        "buttons": [], "inputs": [], "prices": [], "error": None,
    }
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        result["title"] = soup.title.string.strip() if soup.title else url

        seen, headings = set(), []
        for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
            t = tag.get_text(strip=True)
            if t and len(t) < 120 and t not in seen:
                seen.add(t); headings.append(t)
        result["headings"] = headings[:20]

        seen, buttons = set(), []
        for el in soup.find_all(["button", "a", "input"]):
            if el.name == "input" and el.get("type") in ("submit", "button", "reset"):
                t = el.get("value") or el.get("aria-label") or ""
            else:
                t = el.get_text(strip=True) or el.get("aria-label") or ""
            t = t.strip()
            if t and 2 < len(t) < 50 and t.lower() not in _NOISE and t not in seen:
                seen.add(t); buttons.append(t)
        result["buttons"] = buttons[:30]

        seen, inputs = set(), []
        for inp in soup.find_all(["input", "select", "textarea"]):
            if inp.get("type") in ("hidden", "submit", "button", "reset", "image"):
                continue
            label = (
                inp.get("placeholder") or inp.get("name") or
                inp.get("id") or inp.get("aria-label") or inp.get("type") or ""
            ).strip()
            if label and label not in seen:
                seen.add(label); inputs.append(label)
        result["inputs"] = inputs[:20]

        seen, prices = set(), []
        for t in soup.stripped_strings:
            if any(s in t for s in ("$", "£", "€", "₹")):
                t = t.strip()
                if len(t) < 25 and t not in seen:
                    seen.add(t); prices.append(t)
        result["prices"] = prices[:10]

    except requests.exceptions.Timeout:
        result["error"] = f"Timed out — {url} may be slow or unreachable"
    except requests.exceptions.ConnectionError:
        result["error"] = f"Cannot connect to {url}"
    except requests.exceptions.HTTPError as e:
        result["error"] = f"HTTP {e.response.status_code} from {url}"
    except Exception as e:
        result["error"] = f"Error: {e}"
    return result


def scrape_website(pages: list) -> str:
    """Convert list of page dicts into formatted prompt text."""
    sections = []
    for p in pages:
        if p.get("error"):
            sections.append(f"--- Page: {p['url']} ---\nERROR: {p['error']}\n")
            continue
        lines = [
            f"--- Page: {p['url']} ---",
            f"Title: {p.get('title', 'N/A')}",
            "",
            "Headings: " + (", ".join(p.get("headings", [])) or "None found"),
            "",
            "Buttons / CTAs: " + (", ".join(p.get("buttons", [])) or "None found"),
            "",
            "Form Inputs: " + (", ".join(p.get("inputs", [])) or "None found"),
        ]
        if p.get("prices"):
            lines.append("Prices: " + ", ".join(p["prices"]))
        lines.append("")
        sections.append("\n".join(lines))
    return "\n".join(sections)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — LLM  (Groq API — free open-source models)
# ══════════════════════════════════════════════════════════════════════════════

AVAILABLE_MODELS = {
    "llama-3.3-70b-versatile": {
        "label":    "LLaMA 3.3 70B  ★ Best Quality",
        "size":     "70B",
        "type":     "Meta LLaMA",
        "speed":    "Fast",
        "context":  "128K tokens",
        "best_for": "Complex, multi-page QA docs",
    },
    "llama-3.1-8b-instant": {
        "label":    "LLaMA 3.1 8B  ⚡ Fastest",
        "size":     "8B",
        "type":     "Meta LLaMA",
        "speed":    "Instant",
        "context":  "128K tokens",
        "best_for": "Quick drafts, simple pages",
    },
    "mixtral-8x7b-32768": {
        "label":    "Mixtral 8x7B  ⚖ Balanced",
        "size":     "8×7B MoE",
        "type":     "Mistral AI",
        "speed":    "Fast",
        "context":  "32K tokens",
        "best_for": "Balanced speed and quality",
    },
    "gemma2-9b-it": {
        "label":    "Gemma 2 9B  🪶 Lightweight",
        "size":     "9B",
        "type":     "Google",
        "speed":    "Very Fast",
        "context":  "8K tokens",
        "best_for": "Short pages, quick tests",
    },
}

_SYSTEM_PROMPT = """You are a Senior QA Engineer with 10+ years of experience.
Your ONLY task is to produce structured QA documentation from website UI content.

RULES — zero tolerance:
1. Output ONLY structured markdown. No preamble, no commentary.
2. Functional requirements MUST start with: "System shall"
3. User stories MUST follow: "As a [role], I want [action] so that [benefit]"
4. Acceptance criteria MUST use strict Given / When / Then format
5. Edge cases MUST start with: "System should handle"
6. NEVER invent UI elements not in the provided content
7. NEVER write questions — only testable statements
8. Use EXACT button/input names from the scraped content"""


def generate_qa_requirements(prompt: str, api_key: str, model: str) -> dict:
    """Call Groq API. Returns dict: {success, content, tokens, error}."""
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
            top_p=0.9,
        )
        content = response.choices[0].message.content
        tokens  = response.usage.total_tokens if response.usage else 0
        return {"success": True, "content": content, "tokens": tokens, "error": None}

    except Exception as e:
        msg = str(e)
        if "invalid_api_key" in msg.lower() or "authentication" in msg.lower() or "401" in msg:
            msg = "Invalid API key. Check at console.groq.com → API Keys."
        elif "rate_limit" in msg.lower() or "429" in msg:
            msg = "Rate limit hit. Wait ~60 seconds and try again (free tier limit)."
        elif "model_not_found" in msg.lower() or "404" in msg:
            msg = f"Model '{model}' unavailable. Select a different model."
        elif "connection" in msg.lower():
            msg = "Connection error. Check network settings."
        return {"success": False, "content": "", "tokens": 0, "error": msg}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — PROMPT BUILDER  (COSTAR framework)
# ══════════════════════════════════════════════════════════════════════════════

_SECTION_RULES = {
    "Functional Requirements": (
        "## Functional Requirements\n"
        "- Generate at least {n} per page\n"
        "- Each MUST start with: \"System shall\"\n"
        "- Focus on user actions and system responses, not visual display\n"
        "- Format: bulleted list"
    ),
    "User Stories": (
        "## User Stories\n"
        "- Generate at least {n} per page\n"
        "- Format: \"As a [role], I want [action] so that [benefit]\"\n"
        "- Base only on UI elements provided"
    ),
    "Acceptance Criteria": (
        "## Acceptance Criteria\n"
        "- Generate {n} scenarios per page\n"
        "- Strict Given / When / Then — one clause per line\n"
        "- Reference EXACT button/input names from the UI\n"
        "- Each scenario independently testable"
    ),
    "Edge Cases": (
        "## Edge Cases\n"
        "- Generate at least {n} per page\n"
        "- Each MUST start with: \"System should handle\"\n"
        "- Cover: empty inputs, invalid formats, boundary values, timeouts\n"
        "- Statements only — never questions"
    ),
    "Test Cases": (
        "## Test Cases\n"
        "- Generate at least {n} per page\n"
        "- Format: | Test ID | Description | Steps | Expected Result |\n"
        "- Cover happy path AND negative scenarios"
    ),
}


def build_prompt(
    page_content: str,
    sections: List[str],
    max_scenarios: int = 4,
    strict_mode: bool = True,
) -> str:
    section_blocks = "\n\n".join(
        _SECTION_RULES[s].format(n=max_scenarios)
        for s in sections if s in _SECTION_RULES
    )
    section_headers = "\n".join(f"## {s}" for s in sections)
    strict = ""
    if strict_mode:
        strict = (
            "\n---\n"
            "## STRICT MODE\n"
            "NEVER: invent UI elements · write questions · add commentary outside markdown\n"
            "ALWAYS: use exact UI names · start FRs with 'System shall' · use Given/When/Then\n"
        )
    return f"""## CONTEXT
You are a Senior QA Engineer generating production-ready QA documentation.

## OBJECTIVE
Analyse every page in the Website Content below and generate QA documentation.
Cover EVERY page — do not skip any.

## SECTIONS TO GENERATE
{section_blocks}

## STYLE & TONE
Technical, precise, zero ambiguity. Every statement must be independently testable.

## AUDIENCE
QA Engineers writing automated tests · Manual testers · Product Managers
{strict}
## REQUIRED OUTPUT FORMAT (for each page)

# Feature: <Page Name>

{section_headers}

---

## WEBSITE CONTENT

{page_content}

---
Output ONLY markdown. Cover ALL pages above.
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — EXPORTER  (.md / .json / .txt)
# ══════════════════════════════════════════════════════════════════════════════

def export_markdown(content: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        "# QA Requirements Documentation\n\n"
        f"> **Generated:** {ts}  \n"
        "> **Engine:** GenAI QA Generator · Groq API  \n\n"
        "---\n\n"
        + content
        + "\n\n---\n*Auto-generated. Review before use in production.*"
    )


def export_json(content: str, urls: List[str]) -> str:
    features = _parse_to_features(content)
    doc = {
        "metadata": {
            "generated_at":  datetime.now(timezone.utc).isoformat(),
            "generator":     "GenAI QA Requirement Generator",
            "source_urls":   urls,
            "feature_count": len(features),
        },
        "features": features,
    }
    return json.dumps(doc, indent=2, ensure_ascii=False)


def _parse_to_features(content: str) -> list:
    """Parse COSTAR markdown output into structured feature dicts."""
    features, feat, sect, scen = [], None, None, None
    _MAP = {
        "functional": "functional_requirements",
        "user stor":  "user_stories",
        "acceptance": "acceptance_criteria",
        "edge":       "edge_cases",
        "test case":  "test_cases",
    }

    def flush():
        nonlocal scen
        if feat and sect == "acceptance_criteria" and scen and scen.get("steps"):
            feat["acceptance_criteria"].append(dict(scen))
        scen = None

    for line in content.splitlines():
        s = line.strip()
        if not s:
            continue
        if re.match(r"^#\s", s):
            flush()
            if feat:
                features.append(feat)
            feat = {k: [] for k in ["name", "functional_requirements", "user_stories",
                                     "acceptance_criteria", "edge_cases", "test_cases"]}
            feat["name"] = re.sub(r"^#+\s*(Feature:)?\s*", "", s).strip()
            sect = None
        elif re.match(r"^##\s", s) and feat:
            flush()
            h = s.lstrip("#").strip().lower()
            sect = next((v for k, v in _MAP.items() if k in h), None)
        elif re.match(r"^###\s", s) and feat and sect == "acceptance_criteria":
            flush()
            scen = {"title": s.lstrip("#").strip(), "steps": []}
        elif re.match(r"^(given|when|then|and)\s", s, re.I) and feat and sect == "acceptance_criteria":
            if scen is None:
                scen = {"title": "Scenario", "steps": []}
            scen["steps"].append(s)
        elif feat and sect and sect != "acceptance_criteria":
            t = re.sub(r"^[-*•]\s+", "", s)
            t = re.sub(r"^\d+[.)]\s+", "", t).strip()
            if t and t not in feat[sect]:
                feat[sect].append(t)

    flush()
    if feat:
        features.append(feat)
    return features


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="GenAI QA Generator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

:root {
  --bg:     #0b0f1a;
  --surf:   #111827;
  --border: #1e2d42;
  --green:  #00e5a0;
  --blue:   #38bdf8;
  --amber:  #fbbf24;
  --text:   #e2e8f0;
  --muted:  #6b7280;
  --mono:   'JetBrains Mono', monospace;
  --sans:   'Syne', sans-serif;
  --r:      10px;
}

html, body, [class*="css"] { font-family: var(--sans); background: var(--bg); color: var(--text); }
[data-testid="stSidebar"]  { background: var(--surf) !important; border-right: 1px solid var(--border); }
h1,h2,h3,h4                { font-family: var(--sans); font-weight: 800; }

[data-testid="stButton"] > button {
  background: linear-gradient(135deg, var(--green), var(--blue)) !important;
  color: #050a12 !important; font-family: var(--sans) !important; font-weight: 700 !important;
  border: none !important; border-radius: var(--r) !important;
  padding: .55rem 1.4rem !important; letter-spacing: .02em !important;
  transition: transform .15s, box-shadow .15s !important;
}
[data-testid="stButton"] > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 20px rgba(0,229,160,.25) !important;
}

code, pre {
  font-family: var(--mono) !important; font-size: 12.5px !important;
  background: #080d14 !important; border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

.card {
  background: var(--surf); border: 1px solid var(--border);
  border-radius: var(--r); padding: 1.1rem 1.3rem;
  margin-bottom: .8rem; position: relative;
}
.card::before {
  content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%;
  background: linear-gradient(180deg, var(--green), var(--blue));
  border-radius: 4px 0 0 4px;
}

.badge {
  display: inline-block; padding: 2px 9px; border-radius: 20px;
  font-size: 10.5px; font-weight: 700; font-family: var(--mono);
  letter-spacing: .05em; text-transform: uppercase;
}
.bg { background: rgba(0,229,160,.12);  color: var(--green); border: 1px solid rgba(0,229,160,.3); }
.bb { background: rgba(56,189,248,.12); color: var(--blue);  border: 1px solid rgba(56,189,248,.3); }
.ba { background: rgba(251,191,36,.12); color: var(--amber); border: 1px solid rgba(251,191,36,.3); }

.mrow { display: flex; gap: .8rem; flex-wrap: wrap; margin-bottom: 1.3rem; }
.mc   { background: var(--surf); border: 1px solid var(--border); border-radius: var(--r);
        padding: .85rem 1rem; flex: 1; min-width: 100px; text-align: center; }
.mv   { font-size: 1.85rem; font-weight: 800;
        background: linear-gradient(135deg, var(--green), var(--blue));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.1; }
.ml   { font-size: 9.5px; color: var(--muted); text-transform: uppercase;
        letter-spacing: .08em; font-family: var(--mono); margin-top: 3px; }

.hr   { border: none; border-top: 1px solid var(--border); margin: 1.3rem 0; }
.logo { font-size: 1.4rem; font-weight: 800;
        background: linear-gradient(135deg, var(--green), var(--blue));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.logo-sub { font-size: 9.5px; color: var(--muted); font-family: var(--mono); letter-spacing: .07em; }
</style>
""", unsafe_allow_html=True)


# ── Helper: render output + downloads ────────────────────────────────────────
def show_output(content: str, urls: list, model: str, key: str) -> None:
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("### 📄 Generated QA Documentation")
    if model:
        lbl = AVAILABLE_MODELS.get(model, {}).get("label", model)
        st.markdown(
            f'<span class="badge bg">✓ Generated</span> '
            f'<span class="badge bb">🤖 {lbl}</span> '
            f'<span class="badge ba">📄 {len(urls)} page(s)</span>',
            unsafe_allow_html=True,
        )
        st.write("")
    with st.expander("📋 Rendered View", expanded=True):
        st.markdown(content)
    with st.expander("🔤 Raw Markdown"):
        st.code(content, language="markdown")
    st.markdown("#### 💾 Download")
    c1, c2, c3, _ = st.columns([1, 1, 1, 2])
    with c1:
        st.download_button("⬇ .md",   export_markdown(content).encode(),    "qa_requirements.md",   "text/markdown",    key=f"{key}_md")
    with c2:
        st.download_button("⬇ .json", export_json(content, urls).encode(),  "qa_requirements.json", "application/json", key=f"{key}_json")
    with c3:
        st.download_button("⬇ .txt",  content.encode(),                     "qa_requirements.txt",  "text/plain",       key=f"{key}_txt")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:.3rem;">'
        '<span style="font-size:2rem">🧪</span>'
        '<div><div class="logo">QA GenAI</div>'
        '<div class="logo-sub">REQUIREMENT GENERATOR</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    st.markdown("### 🔑 Groq API Key")
    _default = ""
    try:
        _default = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass

    api_key = st.text_input(
        "key", value=_default, type="password",
        placeholder="gsk_...", label_visibility="collapsed",
        help="Free at console.groq.com — no credit card needed.",
    )
    if not api_key:
        st.markdown(
            '<div class="card" style="font-size:12px;color:#6b7280;'
            'font-family:\'JetBrains Mono\',monospace;line-height:2;">'
            '🆓 <strong style="color:#00e5a0;">Get free key</strong><br>'
            '→ console.groq.com<br>→ Sign up (instant, free)<br>'
            '→ API Keys → Create Key<br>→ Paste above ↑</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("### 🤖 Model")
    model_key = st.selectbox(
        "model", list(AVAILABLE_MODELS.keys()),
        format_func=lambda k: AVAILABLE_MODELS[k]["label"],
        label_visibility="collapsed",
    )
    m = AVAILABLE_MODELS[model_key]
    st.markdown(
        f'<div class="card" style="font-size:11px;color:#6b7280;'
        f'font-family:\'JetBrains Mono\',monospace;line-height:1.9;">'
        f'<span class="badge bg">{m["size"]}</span> '
        f'<span class="badge bb">{m["type"]}</span><br><br>'
        f'⚡ {m["speed"]} &nbsp;·&nbsp; 🧠 {m["context"]}<br>'
        f'📝 {m["best_for"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("### 🧩 Sections")
    sections = st.multiselect(
        "s", label_visibility="collapsed",
        options=["Functional Requirements", "User Stories",
                 "Acceptance Criteria", "Edge Cases", "Test Cases"],
        default=["Functional Requirements", "User Stories",
                 "Acceptance Criteria", "Edge Cases"],
    )
    n_scenarios = st.slider("Scenarios per page", 2, 8, 4)
    strict      = st.toggle("Strict COSTAR mode", value=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;color:#6b7280;'
        'font-family:\'JetBrains Mono\',monospace;line-height:2;">'
        '🐍 Python · 🎈 Streamlit<br>'
        '🤖 Groq SDK · 🧠 LLaMA 3<br>'
        '🌐 BeautifulSoup4<br>'
        '📦 100% Open Source</div>',
        unsafe_allow_html=True,
    )


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="font-size:2.2rem;font-weight:800;margin-bottom:.1rem;">'
    'GenAI QA Requirement Generator</h1>'
    '<p style="color:#6b7280;font-family:\'JetBrains Mono\',monospace;font-size:12px;margin-top:0;">'
    'Scrape any website → structured QA docs in seconds &nbsp;|&nbsp;'
    '<span class="badge bg">FREE</span> '
    '<span class="badge bb">OPEN SOURCE</span> '
    '<span class="badge ba">NO COST</span></p>',
    unsafe_allow_html=True,
)
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🌐  URL Scraper", "📝  Paste Content", "📖  How It Works"])


# ── Tab 1: URL Scraper ────────────────────────────────────────────────────────
with tab1:
    col_url, col_btns = st.columns([3, 1])

    with col_url:
        urls_input = st.text_area(
            "🔗 URLs — one per line",
            value=st.session_state.get("demo_urls", ""),
            placeholder=(
                "https://your-app.com/\n"
                "https://your-app.com/login\n"
                "https://your-app.com/register\n"
                "https://your-app.com/cart"
            ),
            height=130,
        )

    with col_btns:
        st.markdown("**Demo:**")
        if st.button("📋 Load demo", key="load_demo"):
            st.session_state["demo_urls"] = (
                "https://sauce-demo.myshopify.com/\n"
                "https://sauce-demo.myshopify.com/account/login\n"
                "https://sauce-demo.myshopify.com/account/register\n"
                "https://sauce-demo.myshopify.com/cart"
            )
            st.rerun()
        if st.button("🧹 Clear", key="clear_all"):
            for k in ["demo_urls", "sc_out", "sc_urls", "sc_model"]:
                st.session_state.pop(k, None)
            st.rerun()

    if not api_key:
        st.warning("⚠️ Add your Groq API key in the sidebar  (free at console.groq.com).")
    if not sections:
        st.warning("⚠️ Select at least one output section in the sidebar.")

    can_run = bool(api_key and urls_input.strip() and sections)

    b1, b2, _ = st.columns([1, 1, 4])
    with b1:
        do_scrape = st.button("🔍 Scrape & Preview", disabled=not urls_input.strip(), key="btn_scrape")
    with b2:
        do_gen = st.button("⚡ Generate QA Docs", disabled=not can_run, key="btn_gen", type="primary")

    if do_scrape and urls_input.strip():
        url_list = [u.strip() for u in urls_input.strip().splitlines() if u.strip()]
        pages, bar = [], st.progress(0, text="Starting…")
        for i, u in enumerate(url_list):
            pages.append(scrape_single_page(u))
            bar.progress((i + 1) / len(url_list), text=f"Scraped: {u[:60]}")
            time.sleep(0.1)
        bar.empty()

        ok  = [p for p in pages if not p.get("error")]
        err = [p for p in pages if p.get("error")]
        for p in err:
            st.error(f"⚠️ {p['url']}: {p['error']}")
        if ok:
            st.success(f"✅ {len(ok)} page(s) scraped.")
            nh = sum(len(p.get("headings", [])) for p in ok)
            nb = sum(len(p.get("buttons",  [])) for p in ok)
            ni = sum(len(p.get("inputs",   [])) for p in ok)
            st.markdown(
                f'<div class="mrow">'
                f'<div class="mc"><div class="mv">{len(ok)}</div><div class="ml">Pages</div></div>'
                f'<div class="mc"><div class="mv">{nh}</div><div class="ml">Headings</div></div>'
                f'<div class="mc"><div class="mv">{nb}</div><div class="ml">Buttons</div></div>'
                f'<div class="mc"><div class="mv">{ni}</div><div class="ml">Inputs</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            for p in pages:
                icon = "⚠️" if p.get("error") else "📄"
                with st.expander(f"{icon} {p['url']}", expanded=False):
                    if p.get("error"):
                        st.error(p["error"])
                    else:
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown("**Headings**")
                            st.write(p.get("headings") or ["—"])
                        with c2:
                            st.markdown("**Buttons / CTAs**")
                            st.write(p.get("buttons", [])[:15] or ["—"])
                        with c3:
                            st.markdown("**Form Inputs**")
                            st.write(p.get("inputs") or ["—"])

    if do_gen and can_run:
        url_list = [u.strip() for u in urls_input.strip().splitlines() if u.strip()]
        with st.spinner("🌐 Scraping pages…"):
            pages = [scrape_single_page(u) for u in url_list]
        content_txt = scrape_website(pages)
        prompt = build_prompt(content_txt, sections, n_scenarios, strict)
        with st.spinner(f"🤖 Generating with {AVAILABLE_MODELS[model_key]['label']}…"):
            result = generate_qa_requirements(prompt, api_key, model_key)
        if result["success"]:
            st.session_state.update({
                "sc_out": result["content"],
                "sc_urls": url_list,
                "sc_model": model_key,
            })
            st.success(f"✅ Done — {result['tokens']:,} tokens used.")
            show_output(result["content"], url_list, model_key, key="sc")
        else:
            st.error(f"❌ {result['error']}")
            st.info("💡 Check your API key · wait 60 s if rate-limited · try LLaMA 3.1 8B.")

    elif "sc_out" in st.session_state and not do_gen:
        show_output(
            st.session_state["sc_out"],
            st.session_state.get("sc_urls", []),
            st.session_state.get("sc_model", ""),
            key="sc_c",
        )


# ── Tab 2: Paste Content ──────────────────────────────────────────────────────
with tab2:
    st.markdown(
        "Paste Figma specs, design notes, HTML snippets, or plain text — "
        "generate QA docs **without scraping**."
    )
    pasted = st.text_area(
        "📋 Paste UI content", height=220,
        placeholder=(
            "Page: Login\n"
            "Elements: Email input, Password input, Login button, Forgot password link\n\n"
            "Page: Dashboard\n"
            "Elements: Welcome banner, Logout button, Search bar, Activity table"
        ),
        key="paste_area",
    )
    if not api_key:
        st.warning("⚠️ Add your Groq API key in the sidebar.")

    paste_ok = bool(api_key and pasted.strip() and sections)
    if st.button("⚡ Generate QA Docs", disabled=not paste_ok, key="btn_paste", type="primary"):
        prompt = build_prompt(pasted, sections, n_scenarios, strict)
        with st.spinner(f"🤖 Generating with {AVAILABLE_MODELS[model_key]['label']}…"):
            result = generate_qa_requirements(prompt, api_key, model_key)
        if result["success"]:
            st.session_state["pt_out"] = result["content"]
            st.success(f"✅ Done — {result['tokens']:,} tokens used.")
            show_output(result["content"], ["Pasted Content"], model_key, key="pt")
        else:
            st.error(f"❌ {result['error']}")
    elif "pt_out" in st.session_state:
        show_output(st.session_state["pt_out"], ["Pasted Content"], "", key="pt_c")


# ── Tab 3: How It Works ───────────────────────────────────────────────────────
with tab3:
    l, r = st.columns(2)

    with l:
        st.markdown("""
### 🏗️ Architecture
```
URLs or pasted text
       │
       ▼
 Scraper  ←  BeautifulSoup4
 headings / buttons / inputs
       │
       ▼
 Prompt Builder  ←  COSTAR framework
 FR · US · AC · EC · TC
       │
       ▼
 Groq API  ←  free open-source LLMs
 LLaMA 3 / Mixtral / Gemma 2
       │
       ▼
 Exporter
 .md  /  .json  /  .txt
       │
       ▼
 Streamlit Cloud UI
```

### ⚙️ Pipeline
1. **Scrape** — extract UI elements from each URL
2. **Prompt** — COSTAR wraps content with QA rules
3. **Generate** — Groq routes to open-source model
4. **Render** — structured markdown in the app
5. **Export** — download `.md`, `.json`, or `.txt`
""")

    with r:
        st.markdown("""
### 🤖 Free Models on Groq

| Model | Params | Context |
|-------|--------|---------|
| LLaMA 3.3 70B | 70B | 128K |
| LLaMA 3.1 8B | 8B | 128K |
| Mixtral 8x7B | 56B MoE | 32K |
| Gemma 2 9B | 9B | 8K |

All **free** — no credit card required.

### 🎯 COSTAR Framework

| | Applied As |
|-|-----------|
| **C**ontext | Senior QA Engineer |
| **O**bjective | QA docs only |
| **S**tyle | Given / When / Then |
| **T**one | Technical, precise |
| **A**udience | QA teams & devs |
| **R**esponse | Strict markdown |

### 🚀 Deploy Steps (GitHub Web UI)

```
1. github.com → New repository
2. Upload app.py  (this file)
3. Upload requirements.txt
4. share.streamlit.io → New App
5. Repo · branch: main · file: app.py
6. Advanced settings → Secrets:
   GROQ_API_KEY = "gsk_..."
7. Deploy ✅
```

### 📦 requirements.txt
```
streamlit>=1.32.0
groq>=0.9.0
beautifulsoup4>=4.12.0
requests>=2.28.0
```
""")
