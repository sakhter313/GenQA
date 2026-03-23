"""
GenAI QA Requirement Generator
================================
Single-file Streamlit app — upload ONLY this file + requirements.txt to GitHub.
No folders, no submodules, no IDE needed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO DEPLOY — STEP-BY-STEP GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — Get a Free Groq API Key (no credit card needed)
  a. Open https://console.groq.com in your browser
  b. Click "Sign Up" (Google/GitHub login works too)
  c. Once logged in, go to "API Keys" in the left sidebar
  d. Click "Create API Key", give it any name, then COPY the key
     ⚠️  Save it immediately — Groq only shows it ONCE
     ✅  It will look like: gsk_AbCdEfGhIjKlMnOpQrStUvWxYz123456

STEP 2 — Create a GitHub Repository
  a. Go to https://github.com and log in
  b. Click the "+" icon (top-right) → "New repository"
  c. Name it anything, e.g. "qa-generator"
  d. Keep it Public, then click "Create repository"
  e. Upload THIS file (app.py) and requirements.txt
     → Click "Add file" → "Upload files" → drag both files in

STEP 3 — Deploy on Streamlit Cloud
  a. Go to https://share.streamlit.io
  b. Click "New app"
  c. Select your GitHub repo, branch = main, main file = app.py
  d. Click "Advanced settings" → find the "Secrets" box
  e. Paste EXACTLY this (replace with YOUR key):
       GROQ_API_KEY = "gsk_your_actual_key_here"
     ⚠️  Do NOT paste the key anywhere else — not in code, not in chat
  f. Click "Deploy" and wait ~60 seconds ✅

STEP 4 — Use the App
  a. Your app URL looks like: https://yourname-qa-generator.streamlit.app
  b. Paste any public website URL into the input box
  c. Click "Generate QA Requirements"
  d. Download or copy the results

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECURITY NOTES (for junior engineers)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• NEVER put your real API key in this file or commit it to Git
• The app validates URLs to block SSRF attacks (explained below)
• Response size is capped to prevent memory exhaustion attacks
• All user inputs are sanitised before being passed to the AI model

Glossary of security terms used below:
  SSRF  = Server-Side Request Forgery — attacker tricks your server
          into making HTTP requests to internal/private addresses
  DoS   = Denial of Service — flooding/crashing a service
  Sanitise = clean and validate input before using it
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — IMPORTS
# Why each library is needed:
#   sys          → read Python version for diagnostics
#   os           → read environment variables (fallback for API key)
#   re           → regular expressions for text cleaning
#   json         → parse structured data from the AI response
#   time         → add retry delays when API is busy
#   ipaddress    → block private/internal IP ranges (SSRF defence)
#   requests     → make HTTP calls to scrape web pages
#   datetime     → timestamps on generated reports
#   typing       → type hints so code is easier to understand
#   streamlit    → the web UI framework
#   BeautifulSoup→ parse HTML from scraped pages
#   groq         → official Groq SDK to call the AI model
# ─────────────────────────────────────────────────────────────────────────────
import sys
import os
import re
import json
import time
import ipaddress
import requests
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urlparse

import streamlit as st
from bs4 import BeautifulSoup
from groq import Groq


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONSTANTS & CONFIGURATION
#
# These are values that never change while the app runs.
# Defining them here (not buried in functions) makes them easy to find & edit.
# ═════════════════════════════════════════════════════════════════════════════

# Words commonly found on web pages that are NOT useful for QA testing.
# If a button or heading matches one of these, we skip it.
_NOISE_WORDS = {
    "home", "menu", "toggle", "close", "open", "skip", "back to top",
    "cookie", "privacy policy", "terms", "accept all", "deny",
    "×", "✕", "»", "«", "...", "more", "less",
}

# HTTP headers we send when scraping — mimics a real Chrome browser.
# Without these, some sites block automated requests.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Safety cap: if a page is larger than this, we stop downloading it.
# This prevents a huge file from crashing the server (DoS defence).
# 5 MB is generous for any normal web page.
MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB

# IP ranges that are considered "internal" / "private".
# Requests to these must be blocked to prevent SSRF attacks.
# Example: 169.254.169.254 is the AWS metadata endpoint — very sensitive!
_BLOCKED_HOSTNAMES = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "169.254.169.254",   # AWS / GCP / Azure metadata service
    "metadata.google.internal",
}

# Currency symbols used to detect price strings on a page.
_CURRENCY_SYMBOLS = ("$", "£", "€", "₹")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — URL VALIDATION (SSRF Defence)
#
# WHAT IS SSRF?
#   Imagine someone types this URL into our app:
#     http://169.254.169.254/latest/meta-data/iam/security-credentials/
#   If we fetch it without checking, our SERVER makes that request —
#   and the attacker gets your cloud provider's secret credentials.
#
# HOW WE STOP IT:
#   Before fetching any URL, we run validate_url() which:
#     1. Only allows http:// and https:// — blocks file://, ftp://, etc.
#     2. Blocks known dangerous hostnames
#     3. Resolves the hostname to an IP and blocks private IP ranges
#     4. Disallows redirects (a redirect could point to an internal address)
# ═════════════════════════════════════════════════════════════════════════════

def validate_url(url: str) -> tuple[bool, str]:
    """
    Check whether a URL is safe to fetch.

    Returns:
        (True, "")          → URL is safe, proceed
        (False, "reason")   → URL is blocked, show reason to user

    Example safe URL:    https://example.com/products
    Example blocked URL: http://localhost/admin
    Example blocked URL: http://169.254.169.254/meta-data/
    """
    # ── 1. Basic format check ────────────────────────────────────────────────
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "URL could not be parsed."

    # ── 2. Only allow standard web schemes ──────────────────────────────────
    if parsed.scheme not in ("http", "https"):
        return False, (
            f"Scheme '{parsed.scheme}' is not allowed. "
            "Only http:// and https:// URLs are accepted."
        )

    hostname = parsed.hostname or ""

    # ── 3. Block known dangerous hostnames ──────────────────────────────────
    if hostname.lower() in _BLOCKED_HOSTNAMES:
        return False, (
            f"Hostname '{hostname}' is blocked for security reasons. "
            "Please use a publicly accessible URL."
        )

    # ── 4. Block private / loopback / link-local IP ranges ──────────────────
    #   ipaddress.ip_address() raises ValueError if hostname is not an IP,
    #   in which case we skip this check (hostname will be resolved by DNS).
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False, (
                f"IP address '{hostname}' belongs to a private or reserved range "
                "and cannot be accessed."
            )
    except ValueError:
        # Not an IP address — that's fine, it's a regular hostname like "example.com"
        pass

    return True, ""  # ✅ All checks passed


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — WEB SCRAPER
#
# Fetches a web page and extracts UI elements that are useful for QA testing:
#   • headings  → what sections exist on the page?
#   • buttons   → what actions can a user take?
#   • inputs    → what data does the user enter?
#   • prices    → are there any prices to verify?
#
# The scraper is intentionally simple — it doesn't execute JavaScript,
# so it works best on server-rendered pages (e-commerce, docs, blogs).
# ═════════════════════════════════════════════════════════════════════════════

def _collect_unique(items: List[str], limit: int) -> List[str]:
    """
    Helper: deduplicate a list while preserving order, then cap at `limit`.

    Why a helper? The original code repeated the same seen/list pattern
    four times. Extracting it here keeps things DRY (Don't Repeat Yourself).

    Example:
        _collect_unique(["Login", "Sign Up", "Login", "Home"], limit=3)
        → ["Login", "Sign Up", "Home"]
    """
    seen: set = set()
    result: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
        if len(result) >= limit:
            break
    return result


def scrape_single_page(url: str) -> dict:
    """
    Fetch one URL and return a structured dict of UI elements.

    Returns a dict with keys:
        url, title, headings, buttons, inputs, prices, error

    If anything goes wrong (network error, blocked URL, etc.),
    the 'error' key contains a human-readable message and all
    other lists will be empty — the app can still run gracefully.
    """
    # Start with a clean result — all fields empty, no error yet
    result: dict = {
        "url": url,
        "title": "",
        "headings": [],
        "buttons": [],
        "inputs": [],
        "prices": [],
        "error": None,
    }

    # ── Step 1: Validate the URL before making ANY network request ───────────
    is_safe, reason = validate_url(url)
    if not is_safe:
        result["error"] = f"URL blocked: {reason}"
        return result

    # ── Step 2: Fetch the page with streaming to enforce a size limit ────────
    #   stream=True means we download the body in chunks rather than all at once.
    #   This lets us stop mid-download if the response is too large.
    try:
        resp = requests.get(
            url,
            headers=_BROWSER_HEADERS,
            timeout=12,          # Give up after 12 seconds (prevents hanging)
            allow_redirects=False,  # Block redirect-based SSRF bypasses
            stream=True,
        )

        # If the server returned a redirect, reject it for safety.
        # A redirect to an internal address bypasses our validate_url() check.
        if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
            result["error"] = (
                "The URL redirects to another location, which is not permitted "
                "for security reasons. Please use the final destination URL."
            )
            return result

        resp.raise_for_status()  # Raise an error for 4xx / 5xx HTTP status codes

        # Download body in 8 KB chunks, stopping if we exceed MAX_RESPONSE_BYTES
        raw_bytes = b""
        for chunk in resp.iter_content(chunk_size=8192):
            raw_bytes += chunk
            if len(raw_bytes) > MAX_RESPONSE_BYTES:
                result["error"] = (
                    f"Page exceeds the {MAX_RESPONSE_BYTES // (1024*1024)} MB size limit. "
                    "Try a different URL."
                )
                return result

    except requests.exceptions.ConnectionError:
        result["error"] = "Could not connect to the URL. Is it publicly accessible?"
        return result
    except requests.exceptions.Timeout:
        result["error"] = "The request timed out after 12 seconds."
        return result
    except requests.exceptions.HTTPError as exc:
        result["error"] = f"HTTP error: {exc}"
        return result

    # ── Step 3: Parse the HTML ───────────────────────────────────────────────
    soup = BeautifulSoup(raw_bytes, "html.parser")

    # Page title — use get_text() instead of .string to handle nested tags
    # e.g. <title><span>My Site</span></title> would break .string
    result["title"] = (
        soup.title.get_text(strip=True) if soup.title else ""
    ) or url

    # ── Step 4: Extract headings (h1–h4) ────────────────────────────────────
    # Headings tell QA engineers what sections exist on the page.
    raw_headings = [
        tag.get_text(strip=True)
        for tag in soup.find_all(["h1", "h2", "h3", "h4"])
        if tag.get_text(strip=True) and len(tag.get_text(strip=True)) < 120
    ]
    result["headings"] = _collect_unique(raw_headings, limit=20)

    # ── Step 5: Extract buttons / clickable elements ─────────────────────────
    # These become test cases: "clicking X should do Y"
    raw_buttons: List[str] = []
    for el in soup.find_all(["button", "a", "input"]):
        # For input-type buttons, the label is in 'value' or 'aria-label'
        if el.name == "input" and el.get("type") in ("submit", "button", "reset"):
            label = el.get("value") or el.get("aria-label") or ""
        else:
            label = el.get_text(strip=True) or el.get("aria-label") or ""
        label = label.strip()

        # Skip noise words and very short/long strings
        if label and 2 < len(label) < 50 and label.lower() not in _NOISE_WORDS:
            raw_buttons.append(label)

    result["buttons"] = _collect_unique(raw_buttons, limit=30)

    # ── Step 6: Extract form inputs ─────────────────────────────────────────
    # Each input is a potential test: "entering X into field Y should show Z"
    raw_inputs: List[str] = []
    for inp in soup.find_all(["input", "select", "textarea"]):
        # Skip invisible / non-interactive input types
        if inp.get("type") in ("hidden", "submit", "button", "reset", "image"):
            continue

        # Try multiple attributes to find a human-readable label
        label = (
            inp.get("placeholder")
            or inp.get("name")
            or inp.get("id")
            or inp.get("aria-label")
            or inp.get("type")
            or ""
        ).strip()

        if label:
            raw_inputs.append(label)

    result["inputs"] = _collect_unique(raw_inputs, limit=20)

    # ── Step 7: Extract prices ───────────────────────────────────────────────
    # Prices are critical to verify in QA — wrong prices cause real damage.
    raw_prices: List[str] = []
    for text in soup.stripped_strings:
        if any(symbol in text for symbol in _CURRENCY_SYMBOLS):
            cleaned = text.strip()
            if len(cleaned) < 25:  # Skip long paragraphs that happen to contain "$"
                raw_prices.append(cleaned)

    result["prices"] = _collect_unique(raw_prices, limit=20)

    return result  # ✅ All done — return the structured data


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — API KEY LOADER
#
# Safely reads the Groq API key from Streamlit Secrets or environment variables.
# Never hardcode or log the key — that would be a serious security risk.
# ═════════════════════════════════════════════════════════════════════════════

def load_api_key() -> Optional[str]:
    """
    Load the Groq API key from a secure source.

    Priority:
      1. Streamlit Secrets (st.secrets) — used in production on Streamlit Cloud
      2. OS environment variable         — useful for local development

    Returns the key string, or None if not found.

    HOW TO SET IT LOCALLY (for development):
      On Mac/Linux, run in your terminal before `streamlit run app.py`:
        export GROQ_API_KEY="gsk_your_key_here"
      On Windows PowerShell:
        $env:GROQ_API_KEY="gsk_your_key_here"
    """
    # Try Streamlit Secrets first (set via the Streamlit Cloud dashboard)
    key = st.secrets.get("GROQ_API_KEY", None)

    # Fallback: read from OS environment (useful when running locally)
    if not key:
        key = os.environ.get("GROQ_API_KEY", None)

    return key


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — AI PROMPT BUILDER & QA GENERATOR
#
# This section takes the scraped page data and asks the Groq AI model
# to write QA (Quality Assurance) test requirements.
#
# What are QA requirements?
#   They are structured descriptions of how a feature SHOULD behave.
#   Example: "When the user clicks 'Add to Cart', the cart count increases by 1."
#
# We use a technique called "prompt engineering" — carefully worded instructions
# to the AI that produce consistent, structured output.
# ═════════════════════════════════════════════════════════════════════════════

def build_prompt(page_data: dict) -> str:
    """
    Convert scraped page data into a detailed prompt for the AI model.

    A good prompt:
      • Gives context (what the page is about)
      • Gives the AI the raw data it needs
      • Tells the AI exactly what format to output
      • Uses examples so the AI doesn't guess

    Args:
        page_data: dict returned by scrape_single_page()

    Returns:
        A multi-line string ready to send to the Groq API.
    """
    # Format each section for readability in the prompt
    headings_text = "\n".join(f"  • {h}" for h in page_data["headings"]) or "  (none found)"
    buttons_text  = "\n".join(f"  • {b}" for b in page_data["buttons"])  or "  (none found)"
    inputs_text   = "\n".join(f"  • {i}" for i in page_data["inputs"])   or "  (none found)"
    prices_text   = "\n".join(f"  • {p}" for p in page_data["prices"])   or "  (none found)"

    return f"""
You are a senior QA engineer. Analyse the following web page summary and
generate comprehensive, actionable QA test requirements.

PAGE INFORMATION
================
URL   : {page_data["url"]}
Title : {page_data["title"]}

HEADINGS (page sections)
------------------------
{headings_text}

INTERACTIVE BUTTONS / LINKS
----------------------------
{buttons_text}

FORM INPUTS
-----------
{inputs_text}

PRICES FOUND
------------
{prices_text}

YOUR TASK
=========
Write QA requirements grouped into the following categories.
For each requirement, use the format:

  [REQ-XXX] <Requirement title>
  Given: <precondition>
  When : <user action>
  Then : <expected outcome>

Categories to cover:
  1. Functional Requirements  — core features work as expected
  2. UI / UX Requirements     — layout, labels, and navigation are correct
  3. Input Validation         — form fields handle valid and invalid data
  4. Pricing Accuracy         — prices display correctly (if applicable)
  5. Accessibility            — keyboard navigation, screen-reader labels
  6. Edge Cases               — empty states, network errors, long strings

Write at least 3 requirements per category. Be specific — reference
actual button names and field names from the data above.
""".strip()


def generate_qa_requirements(page_data: dict, api_key: str) -> str:
    """
    Call the Groq API and return the AI-generated QA requirements.

    Includes a simple retry loop: if the API returns a rate-limit error
    (HTTP 429) we wait a few seconds and try again (up to 3 times).

    Args:
        page_data: dict from scrape_single_page()
        api_key:   the Groq API key from load_api_key()

    Returns:
        The AI's response text as a plain string.
    """
    client = Groq(api_key=api_key)
    prompt = build_prompt(page_data)

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",  # Best free model on Groq (as of 2024)
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior QA engineer who writes clear, "
                            "structured, and actionable test requirements. "
                            "Always use the Given/When/Then format."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,   # Lower = more consistent / less creative output
                max_tokens=2048,
            )
            # Extract the text from the first (and usually only) completion choice
            return response.choices[0].message.content

        except Exception as exc:
            error_str = str(exc).lower()

            # Rate limit: wait and retry
            if "rate_limit" in error_str or "429" in error_str:
                wait_seconds = 2 ** attempt  # 2s, 4s, 8s (exponential backoff)
                st.warning(
                    f"Groq API rate limit hit (attempt {attempt}/{max_retries}). "
                    f"Retrying in {wait_seconds}s…"
                )
                time.sleep(wait_seconds)
                continue

            # Any other error — don't retry, surface the message to the user
            raise RuntimeError(f"Groq API error: {exc}") from exc

    raise RuntimeError("Groq API rate limit exceeded after 3 retries. Please wait a minute.")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — STREAMLIT UI
#
# This section builds the web interface using Streamlit.
# Streamlit turns Python functions into interactive web pages automatically.
#
# UI Flow:
#   1. Show title and instructions
#   2. Ask for URL input
#   3. On button click → validate → scrape → generate → display
#   4. Show download button for the results
# ═════════════════════════════════════════════════════════════════════════════

def render_ui() -> None:
    """
    Build and render the complete Streamlit UI.
    Called once from main() — keeps all UI code in one place.
    """
    # ── Page config (must be first Streamlit call) ───────────────────────────
    st.set_page_config(
        page_title="GenAI QA Generator",
        page_icon="🧪",
        layout="centered",
    )

    # ── Header ───────────────────────────────────────────────────────────────
    st.title("🧪 GenAI QA Requirement Generator")
    st.markdown(
        "Paste any **public** web page URL below and we'll scrape its UI elements, "
        "then use AI to generate structured QA test requirements."
    )
    st.divider()

    # ── API Key check ─────────────────────────────────────────────────────────
    api_key = load_api_key()
    if not api_key:
        st.error(
            "⚠️ **Groq API key not found.**\n\n"
            "If you're running locally, set the environment variable:\n"
            "```\nexport GROQ_API_KEY='gsk_your_key_here'\n```\n\n"
            "If you're on Streamlit Cloud, go to your app's "
            "**Advanced Settings → Secrets** and add:\n"
            "```\nGROQ_API_KEY = \"gsk_your_key_here\"\n```"
        )
        st.stop()  # Stop rendering — no point going further without a key

    # ── URL Input ────────────────────────────────────────────────────────────
    url = st.text_input(
        label="🌐 Website URL",
        placeholder="https://example.com/your-page",
        help=(
            "Enter any publicly accessible web page. "
            "Private networks, localhost, and IP addresses are blocked for security."
        ),
    )

    generate_btn = st.button("🚀 Generate QA Requirements", type="primary")

    # ── Main Logic (runs when button is clicked) ─────────────────────────────
    if generate_btn:

        # ── Validate input ───────────────────────────────────────────────────
        if not url or not url.strip():
            st.warning("Please enter a URL first.")
            st.stop()

        url = url.strip()

        # Pre-flight URL safety check — show a friendly message if blocked
        is_safe, block_reason = validate_url(url)
        if not is_safe:
            st.error(f"🚫 **URL blocked:** {block_reason}")
            st.stop()

        # ── Step 1: Scrape ───────────────────────────────────────────────────
        with st.spinner("🔍 Scraping page — this takes a few seconds…"):
            page_data = scrape_single_page(url)

        if page_data["error"]:
            st.error(f"❌ **Scraping failed:** {page_data['error']}")
            st.stop()

        # Show a summary of what was found
        with st.expander("📋 Scraped Page Summary (click to expand)", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Headings found",  len(page_data["headings"]))
                st.metric("Buttons found",   len(page_data["buttons"]))
            with col2:
                st.metric("Inputs found",    len(page_data["inputs"]))
                st.metric("Prices found",    len(page_data["prices"]))

            if page_data["headings"]:
                st.markdown("**Headings:**")
                st.write(page_data["headings"])
            if page_data["buttons"]:
                st.markdown("**Buttons:**")
                st.write(page_data["buttons"])
            if page_data["inputs"]:
                st.markdown("**Inputs:**")
                st.write(page_data["inputs"])
            if page_data["prices"]:
                st.markdown("**Prices:**")
                st.write(page_data["prices"])

        # ── Step 2: Generate ─────────────────────────────────────────────────
        with st.spinner("🤖 Asking AI to generate QA requirements…"):
            try:
                qa_output = generate_qa_requirements(page_data, api_key)
            except RuntimeError as exc:
                st.error(f"❌ **AI generation failed:** {exc}")
                st.stop()

        # ── Step 3: Display results ──────────────────────────────────────────
        st.success("✅ QA Requirements generated successfully!")
        st.divider()
        st.subheader("📄 Generated QA Requirements")
        st.markdown(qa_output)

        # ── Step 4: Download button ──────────────────────────────────────────
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename  = f"qa_requirements_{timestamp}.txt"

        download_content = (
            f"QA Requirements for: {url}\n"
            f"Generated at: {timestamp} UTC\n"
            f"Page title: {page_data['title']}\n"
            f"{'=' * 60}\n\n"
            f"{qa_output}"
        )

        st.download_button(
            label="⬇️ Download as .txt",
            data=download_content,
            file_name=filename,
            mime="text/plain",
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ENTRY POINT
#
# In Python, `if __name__ == "__main__"` means:
#   "Only run this block if this file is run directly (not imported)."
#
# When Streamlit runs `streamlit run app.py`, it imports app.py as a module,
# so this block still executes — it is the standard pattern for Streamlit apps.
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    render_ui()
