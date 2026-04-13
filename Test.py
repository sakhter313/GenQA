import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urljoin, urlparse

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Test Case Generator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }

    .stApp {
        background: #0a0e1a;
        color: #e2e8f0;
    }

    /* Header */
    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(99,102,241,0.15) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #818cf8, #c084fc, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        flex: 1;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #818cf8;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* Section labels */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #6366f1;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Test case card */
    .test-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-left: 3px solid #6366f1;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s;
    }
    .test-card:hover {
        border-left-color: #c084fc;
    }
    .test-card.negative { border-left-color: #f87171; }
    .test-card.edge     { border-left-color: #fbbf24; }
    .test-card.positive { border-left-color: #34d399; }

    .tc-id {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #64748b;
        margin-bottom: 0.3rem;
    }
    .tc-title {
        font-size: 1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
    }
    .tc-detail {
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.6;
    }
    .tc-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }
    .badge-positive { background: rgba(52,211,153,0.15); color: #34d399; }
    .badge-negative { background: rgba(248,113,113,0.15); color: #f87171; }
    .badge-edge     { background: rgba(251,191,36,0.15);  color: #fbbf24; }
    .badge-security { background: rgba(192,132,252,0.15); color: #c084fc; }
    .badge-performance { background: rgba(56,189,248,0.15); color: #38bdf8; }

    /* Area tag */
    .area-tag {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        color: #818cf8;
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 6px;
        padding: 0.3rem 0.7rem;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Scraped info box */
    .scraped-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1.2rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #64748b;
        max-height: 200px;
        overflow-y: auto;
    }
    .scraped-box .highlight { color: #38bdf8; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        transition: opacity 0.2s;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stTextArea > div > div > textarea {
        background: #111827 !important;
        border: 1px solid #1e293b !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #1e293b;
    }

    .status-ok  { color: #34d399; font-weight: 700; }
    .status-err { color: #f87171; font-weight: 700; }

    hr { border-color: #1e293b; }
</style>
""", unsafe_allow_html=True)


# ─── Scraper ─────────────────────────────────────────────────────────────────
def scrape_website(url: str) -> dict:
    """Scrape a website and extract testing-relevant information."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "noscript"]):
            tag.decompose()

        data = {
            "url": url,
            "title": soup.title.string.strip() if soup.title else "Unknown",
            "status_code": resp.status_code,
            "forms": [],
            "inputs": [],
            "buttons": [],
            "links": [],
            "headings": [],
            "images": [],
            "tables": [],
            "page_text": "",
            "areas": []
        }

        # Forms
        for form in soup.find_all("form"):
            form_data = {
                "action": form.get("action", ""),
                "method": form.get("method", "GET").upper(),
                "fields": []
            }
            for inp in form.find_all(["input", "select", "textarea"]):
                form_data["fields"].append({
                    "type": inp.get("type", inp.name),
                    "name": inp.get("name", ""),
                    "placeholder": inp.get("placeholder", ""),
                    "required": inp.has_attr("required")
                })
            data["forms"].append(form_data)

        # Inputs (outside forms)
        for inp in soup.find_all("input"):
            data["inputs"].append({
                "type": inp.get("type", "text"),
                "name": inp.get("name", ""),
                "placeholder": inp.get("placeholder", "")
            })

        # Buttons
        for btn in soup.find_all(["button", "input[type=submit]"]):
            txt = btn.get_text(strip=True)
            if txt:
                data["buttons"].append(txt[:60])

        # Links (sample 15)
        links = []
        base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        for a in soup.find_all("a", href=True)[:15]:
            href = a["href"]
            full = urljoin(base, href)
            links.append({"text": a.get_text(strip=True)[:40], "href": full})
        data["links"] = links

        # Headings
        for h in soup.find_all(["h1", "h2", "h3"]):
            txt = h.get_text(strip=True)
            if txt:
                data["headings"].append(txt[:80])

        # Images
        for img in soup.find_all("img")[:10]:
            data["images"].append({
                "alt": img.get("alt", ""),
                "src": img.get("src", "")[:60]
            })

        # Tables
        for tbl in soup.find_all("table")[:3]:
            headers = [th.get_text(strip=True) for th in tbl.find_all("th")]
            if headers:
                data["tables"].append({"headers": headers})

        # Visible text (trimmed)
        data["page_text"] = " ".join(soup.get_text(separator=" ").split())[:2000]

        # Infer test areas
        text_lower = data["page_text"].lower()
        area_map = {
            "Authentication": ["login", "signup", "register", "password", "logout", "auth"],
            "Search":         ["search", "filter", "query", "find", "sort"],
            "Forms":          ["form", "input", "submit", "field", "validate"],
            "Navigation":     ["menu", "nav", "breadcrumb", "pagination", "link"],
            "API/Data":       ["api", "endpoint", "json", "fetch", "response"],
            "Media":          ["audio", "video", "stream", "player", "upload"],
            "E-commerce":     ["cart", "checkout", "payment", "order", "price"],
            "User Profile":   ["profile", "account", "settings", "preference"],
            "Security":       ["token", "csrf", "xss", "https", "cookie", "session"],
            "Accessibility":  ["aria", "alt", "role", "label", "tabindex"],
        }
        for area, keywords in area_map.items():
            if any(kw in text_lower for kw in keywords):
                data["areas"].append(area)

        return {"success": True, "data": data}

    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. Try a different URL."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Could not connect to the website."}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ─── Groq AI ─────────────────────────────────────────────────────────────────
def generate_test_cases(scraped_data: dict, area: str, tc_count: int, api_key: str) -> dict:
    """Call Groq API to generate structured test cases."""
    prompt = f"""
You are a senior QA engineer. Analyze this scraped website data and generate {tc_count} test cases for the area: "{area}".

WEBSITE INFO:
- URL: {scraped_data['url']}
- Title: {scraped_data['title']}
- Forms found: {json.dumps(scraped_data['forms'][:3], indent=2)}
- Buttons: {scraped_data['buttons'][:10]}
- Headings: {scraped_data['headings'][:8]}
- Page text snippet: {scraped_data['page_text'][:800]}
- Links sample: {[l['text'] for l in scraped_data['links'][:8]]}

Return ONLY a valid JSON array. No markdown, no explanation, no code fences.
Each object must have these exact keys:
- "id": string like "TC-001"
- "title": short test case title
- "type": one of "Positive", "Negative", "Edge", "Security", "Performance"
- "precondition": what must be true before the test
- "steps": array of step strings
- "expected_result": what should happen
- "priority": "High", "Medium", or "Low"
- "area": the test area

Generate a mix of Positive, Negative, Edge, Security types where relevant.
"""

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
            "max_tokens": 3000
        }
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        # Strip markdown fences if present
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        test_cases = json.loads(content)
        return {"success": True, "test_cases": test_cases}

    except json.JSONDecodeError as e:
        return {"success": False, "error": f"AI returned invalid JSON: {e}"}
    except requests.exceptions.HTTPError as e:
        return {"success": False, "error": f"Groq API error: {e.response.status_code} — check your API key."}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ─── Render Test Card ─────────────────────────────────────────────────────────
def render_test_card(tc: dict, index: int):
    tc_type = tc.get("type", "Positive").lower()
    card_class = tc_type if tc_type in ["negative", "edge", "positive"] else "positive"
    badge_class = f"badge-{tc_type}" if tc_type in ["positive","negative","edge","security","performance"] else "badge-positive"

    steps_html = "".join(
        f"<div style='padding: 0.2rem 0; color:#94a3b8;'>→ {s}</div>"
        for s in tc.get("steps", [])
    )

    priority_color = {"High": "#f87171", "Medium": "#fbbf24", "Low": "#34d399"}.get(
        tc.get("priority", "Medium"), "#94a3b8"
    )

    st.markdown(f"""
    <div class="test-card {card_class}">
        <div class="tc-id">{tc.get('id', f'TC-{index+1:03d}')} · {tc.get('area', '')}</div>
        <div class="tc-title">{tc.get('title', 'Untitled')}</div>
        <span class="tc-badge {badge_class}">{tc.get('type', 'Positive')}</span>
        <span style="font-size:0.65rem; color:{priority_color}; font-weight:700;
                     text-transform:uppercase; letter-spacing:0.08em;
                     margin-left:0.5rem;">● {tc.get('priority','Medium')} Priority</span>
        <div class="tc-detail" style="margin-top:0.7rem;">
            <strong style="color:#64748b; font-size:0.75rem;">PRECONDITION</strong><br>
            {tc.get('precondition', 'N/A')}
        </div>
        <div class="tc-detail" style="margin-top:0.7rem;">
            <strong style="color:#64748b; font-size:0.75rem;">STEPS</strong>
            {steps_html}
        </div>
        <div class="tc-detail" style="margin-top:0.7rem;">
            <strong style="color:#64748b; font-size:0.75rem;">EXPECTED RESULT</strong><br>
            {tc.get('expected_result', 'N/A')}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get free key at console.groq.com"
    )

    st.markdown("---")
    st.markdown("### 📋 How It Works")
    st.markdown("""
1. Enter your **Groq API key**
2. Paste any **website URL**
3. Click **Scrape Website**
4. Select a **test area**
5. Generate **AI test cases**
    """)

    st.markdown("---")
    st.markdown("### 🔗 Free Resources")
    st.markdown("""
- [Get Groq API Key](https://console.groq.com)
- [Streamlit Docs](https://docs.streamlit.io)
- [BeautifulSoup Docs](https://beautiful-soup-4.readthedocs.io)
    """)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.7rem;color:#475569;'>Model: llama-3.3-70b-versatile<br>"
        "Scraper: BeautifulSoup4<br>Built with Streamlit</div>",
        unsafe_allow_html=True
    )


# ─── Main UI ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🧪 AI Test Case Generator</div>
    <div class="hero-subtitle">
        scrape → analyze → generate · powered by Groq LLaMA 3.3
    </div>
</div>
""", unsafe_allow_html=True)

# Session state
if "scraped_data" not in st.session_state:
    st.session_state.scraped_data = None
if "test_cases" not in st.session_state:
    st.session_state.test_cases = []
if "selected_area" not in st.session_state:
    st.session_state.selected_area = None

# ─── Step 1: Scrape ──────────────────────────────────────────────────────────
st.markdown("### Step 1 — Scrape Website")
col1, col2 = st.columns([4, 1])

with col1:
    url_input = st.text_input(
        "Website URL",
        placeholder="https://example.com",
        label_visibility="collapsed"
    )
with col2:
    scrape_btn = st.button("🔍 Scrape", use_container_width=True)

if scrape_btn:
    if not url_input:
        st.error("Enter a URL first.")
    else:
        if not url_input.startswith("http"):
            url_input = "https://" + url_input
        with st.spinner("Scraping website..."):
            result = scrape_website(url_input)

        if result["success"]:
            st.session_state.scraped_data = result["data"]
            st.session_state.test_cases = []
            st.success(f"✅ Scraped: **{result['data']['title']}** (HTTP {result['data']['status_code']})")
        else:
            st.error(f"❌ {result['error']}")

# ─── Show Scraped Info ────────────────────────────────────────────────────────
if st.session_state.scraped_data:
    d = st.session_state.scraped_data

    # Metrics
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-value">{len(d['forms'])}</div>
            <div class="metric-label">Forms</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{len(d['inputs'])}</div>
            <div class="metric-label">Inputs</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{len(d['buttons'])}</div>
            <div class="metric-label">Buttons</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{len(d['links'])}</div>
            <div class="metric-label">Links</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{len(d['areas'])}</div>
            <div class="metric-label">Test Areas</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Detected areas
    st.markdown('<div class="section-label">Detected Test Areas</div>', unsafe_allow_html=True)
    if d["areas"]:
        areas_html = "".join(f'<span class="area-tag">{a}</span>' for a in d["areas"])
        st.markdown(areas_html, unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#64748b;font-size:0.85rem;">No specific areas detected — will use General.</span>', unsafe_allow_html=True)
        d["areas"] = ["General"]

    # Scraped preview
    with st.expander("📄 Raw Scraped Data Preview"):
        st.markdown(f"""
        <div class="scraped-box">
            <span class="highlight">Title:</span> {d['title']}<br>
            <span class="highlight">Forms:</span> {len(d['forms'])} found<br>
            <span class="highlight">Buttons:</span> {', '.join(d['buttons'][:8]) or 'None'}<br>
            <span class="highlight">Headings:</span> {' | '.join(d['headings'][:5]) or 'None'}<br>
            <span class="highlight">Text snippet:</span> {d['page_text'][:300]}...
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ─── Step 2: Generate ────────────────────────────────────────────────────
    st.markdown("### Step 2 — Generate Test Cases")

    col_a, col_b, col_c = st.columns([2, 1, 1])

    with col_a:
        area_options = d["areas"] + ["General", "Custom"]
        selected_area = st.selectbox("Test Area", area_options)
        if selected_area == "Custom":
            selected_area = st.text_input("Enter custom area", placeholder="e.g. Audio Player")

    with col_b:
        tc_count = st.selectbox("Number of Test Cases", [5, 10, 15, 20], index=1)

    with col_c:
        st.markdown("<br>", unsafe_allow_html=True)
        gen_btn = st.button("⚡ Generate", use_container_width=True)

    if gen_btn:
        if not groq_api_key:
            st.error("Add your Groq API key in the sidebar.")
        else:
            with st.spinner(f"Generating {tc_count} test cases for **{selected_area}**..."):
                result = generate_test_cases(d, selected_area, tc_count, groq_api_key)

            if result["success"]:
                st.session_state.test_cases = result["test_cases"]
                st.session_state.selected_area = selected_area
                st.success(f"✅ Generated {len(result['test_cases'])} test cases")
            else:
                st.error(f"❌ {result['error']}")

# ─── Step 3: Display Test Cases ───────────────────────────────────────────────
if st.session_state.test_cases:
    tcs = st.session_state.test_cases
    area = st.session_state.selected_area

    st.markdown("---")
    st.markdown(f"### Test Cases — {area}")

    # Summary stats
    type_counts = {}
    priority_counts = {}
    for tc in tcs:
        t = tc.get("type", "Positive")
        p = tc.get("priority", "Medium")
        type_counts[t] = type_counts.get(t, 0) + 1
        priority_counts[p] = priority_counts.get(p, 0) + 1

    stats_html = "".join(
        f'<span class="area-tag">{t}: {c}</span>'
        for t, c in type_counts.items()
    )
    st.markdown(stats_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Filter by type
    all_types = ["All"] + list(type_counts.keys())
    filter_type = st.radio("Filter by Type", all_types, horizontal=True)

    filtered = tcs if filter_type == "All" else [tc for tc in tcs if tc.get("type") == filter_type]

    for i, tc in enumerate(filtered):
        render_test_card(tc, i)

    st.markdown("---")

    # Export
    st.markdown("### 📥 Export")
    col_ex1, col_ex2 = st.columns(2)

    with col_ex1:
        json_str = json.dumps(tcs, indent=2)
        st.download_button(
            "⬇ Download JSON",
            data=json_str,
            file_name=f"test_cases_{area.lower().replace(' ','_')}.json",
            mime="application/json",
            use_container_width=True
        )

    with col_ex2:
        # CSV export
        csv_lines = ["ID,Title,Type,Priority,Precondition,Steps,Expected Result,Area"]
        for tc in tcs:
            steps = " | ".join(tc.get("steps", []))
            row = (
                f'"{tc.get("id","")}",'
                f'"{tc.get("title","")}",'
                f'"{tc.get("type","")}",'
                f'"{tc.get("priority","")}",'
                f'"{tc.get("precondition","")}",'
                f'"{steps}",'
                f'"{tc.get("expected_result","")}",'
                f'"{tc.get("area","")}"'
            )
            csv_lines.append(row)
        csv_str = "\n".join(csv_lines)
        st.download_button(
            "⬇ Download CSV",
            data=csv_str,
            file_name=f"test_cases_{area.lower().replace(' ','_')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ─── Empty State ─────────────────────────────────────────────────────────────
if not st.session_state.scraped_data:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color:#334155;">
        <div style="font-size:4rem; margin-bottom:1rem;">🧪</div>
        <div style="font-size:1.1rem; font-weight:600; color:#475569;">
            Enter a URL above and click Scrape to begin
        </div>
        <div style="font-size:0.85rem; margin-top:0.5rem; color:#334155;">
            Works on any public website — login pages, dashboards, e-commerce, APIs
        </div>
    </div>
    """, unsafe_allow_html=True)
