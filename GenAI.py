"""
Advanced GenAI QA Requirement Generator
========================================
Multi-model · Deep Scraper · Application-Area Aware · Streamlit App

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEPLOY GUIDE (one-time setup)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — Get a Free Groq API Key
  a. Visit https://console.groq.com → Sign Up (free, no credit card)
  b. Go to "API Keys" → "Create API Key" → COPY the key immediately
     Looks like: gsk_AbCdEfGhIjKlMnOpQrStUvWxYz123456
     Groq shows it ONLY ONCE — save it now

STEP 2 — GitHub Repository
  a. https://github.com → "+" → "New repository" → name it (e.g. qa-generator)
  b. Upload: this file (app.py) + requirements.txt

STEP 3 — Streamlit Cloud
  a. https://share.streamlit.io → "New app"
  b. Select repo · branch: main · Main file: app.py
  c. "Advanced settings" → Secrets → paste:
       GROQ_API_KEY = "gsk_your_actual_key_here"
  d. Click "Deploy"

STEP 4 — Done!
  Paste any public URL, pick a model + focus area, click Generate.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT'S NEW vs BASIC VERSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 5 Groq models — each gives a different QA style and depth
• 10 application area detectors (e-commerce, auth, search, media, etc.)
• Deep scraper: tables, modals, carousels, alerts, breadcrumbs,
  pagination, social links, JSON-LD structured data, A11y signals
• Side-by-side model comparison mode
• Confidence scoring per detected application area
• Export as .txt or .md
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import json
import time
import ipaddress
import requests
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import streamlit as st
from bs4 import BeautifulSoup
from groq import Groq


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MODEL REGISTRY
#
# Each model has different strengths for QA generation.
# Temperature: lower = more consistent/structured. Higher = more creative.
# We expose all 5 models so users can compare outputs.
# ═════════════════════════════════════════════════════════════════════════════

MODELS = {
    "llama-3.3-70b-versatile": {
        "label": "Llama 3.3 70B — Versatile (Recommended)",
        "description": "Best all-rounder. Deep, structured QA output with broad coverage.",
        "temperature": 0.3,
        "max_tokens": 3000,
        "strength": "Balanced depth & breadth",
        "speed": "Medium",
        "icon": "🦙",
    },
    "llama-3.1-8b-instant": {
        "label": "Llama 3.1 8B — Instant (Fast)",
        "description": "Fastest model. Good for quick drafts or high-level requirement lists.",
        "temperature": 0.4,
        "max_tokens": 2000,
        "strength": "Speed & conciseness",
        "speed": "Very Fast",
        "icon": "⚡",
    },
    "llama-3.3-70b-specdec": {
        "label": "Llama 3.3 70B — SpecDec (Edge Cases)",
        "description": "Speculative decoding variant. Best for granular edge-case discovery.",
        "temperature": 0.2,
        "max_tokens": 3500,
        "strength": "Edge cases & failure modes",
        "speed": "Medium",
        "icon": "🔬",
    },
    "qwen/qwen3-32b": {
        "label": "Qwen3 32B — Multilingual",
        "description": "Strong multilingual reasoning. Great for internationalised apps.",
        "temperature": 0.3,
        "max_tokens": 3000,
        "strength": "Multilingual & i18n testing",
        "speed": "Medium",
        "icon": "🌐",
    },
    "openai/gpt-oss-120b": {
        "label": "GPT-OSS 120B — Deep Reasoning",
        "description": "Largest model. Best for complex enterprise QA and compliance.",
        "temperature": 0.25,
        "max_tokens": 4000,
        "strength": "Complex reasoning & compliance",
        "speed": "Slow",
        "icon": "🧠",
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — APPLICATION AREA DEFINITIONS
#
# Each area = a named domain of web app functionality.
# The scraper uses keywords + CSS selectors to detect whether each area
# is present on the scraped page, and assigns a confidence score.
#
# QA needs are fundamentally different per area:
#   Checkout    → payment, cart state, error recovery
#   Auth        → session management, brute-force, 2FA
#   Search      → result accuracy, filters, empty states
#   Media       → buffering, controls, responsive scaling
#   A11y        → WCAG, keyboard navigation, screen readers
# ═════════════════════════════════════════════════════════════════════════════

APP_AREAS = {
    "ecommerce": {
        "label": "🛒 E-Commerce & Checkout",
        "keywords": [
            "cart", "checkout", "buy", "purchase", "order", "payment",
            "shipping", "add to cart", "quantity", "wishlist", "coupon",
            "discount", "promo", "total", "subtotal", "tax", "invoice",
        ],
        "selectors": [
            "[class*='cart']", "[class*='checkout']", "[class*='product']",
            "[class*='price']", "[class*='basket']", "form[action*='cart']",
        ],
        "qa_focus": (
            "payment flows, cart persistence across sessions, pricing accuracy, "
            "order confirmation emails, coupon/promo validation, stock edge cases, "
            "guest vs authenticated checkout"
        ),
    },
    "authentication": {
        "label": "🔐 Authentication & Security",
        "keywords": [
            "login", "sign in", "sign up", "register", "password", "forgot",
            "reset", "logout", "2fa", "otp", "verify", "session", "token",
            "oauth", "sso", "remember me", "biometric",
        ],
        "selectors": [
            "[type='password']", "form[action*='login']", "form[action*='auth']",
            "[class*='login']", "[class*='auth']", "[class*='signup']",
        ],
        "qa_focus": (
            "login/logout flows, password strength rules, session timeout, "
            "2FA enrollment & fallback, brute-force lockout, OAuth redirect handling, "
            "concurrent session management"
        ),
    },
    "search": {
        "label": "🔍 Search & Filtering",
        "keywords": [
            "search", "filter", "sort", "query", "find", "results",
            "autocomplete", "facet", "refine", "keyword", "category filter",
            "no results", "did you mean",
        ],
        "selectors": [
            "[type='search']", "[class*='search']", "[class*='filter']",
            "[class*='facet']", "[class*='sort']", "input[placeholder*='search' i]",
        ],
        "qa_focus": (
            "search relevance, filter combinations, empty-state messaging, "
            "special characters & SQL injection in search, pagination of results, "
            "autocomplete performance, saved searches"
        ),
    },
    "forms": {
        "label": "📝 Forms & Data Entry",
        "keywords": [
            "submit", "required", "optional", "valid", "invalid", "error",
            "field", "input", "select", "textarea", "upload", "attach",
            "date picker", "phone", "address", "postcode",
        ],
        "selectors": [
            "form", "input:not([type='hidden'])", "select", "textarea",
            "[required]", "[class*='form']", "[class*='field']",
        ],
        "qa_focus": (
            "required-field enforcement, inline validation messages, "
            "submission with missing/invalid data, file upload size & type limits, "
            "date picker edge cases, browser autofill compatibility, "
            "data persistence on back navigation"
        ),
    },
    "navigation": {
        "label": "🧭 Navigation & Routing",
        "keywords": [
            "menu", "nav", "breadcrumb", "tab", "sidebar", "header",
            "footer", "link", "back", "next", "previous", "home",
            "404", "sitemap",
        ],
        "selectors": [
            "nav", "header", "footer", "[role='navigation']", "[class*='nav']",
            "[class*='menu']", "[class*='breadcrumb']", "[class*='sidebar']",
        ],
        "qa_focus": (
            "all links resolve correctly, breadcrumb accuracy, mobile hamburger menu, "
            "keyboard-only navigation, deep-link sharing, browser back/forward, "
            "404 handling, active-state indicators"
        ),
    },
    "media": {
        "label": "🎬 Media & Content",
        "keywords": [
            "video", "audio", "image", "gallery", "carousel", "slider",
            "play", "pause", "stream", "podcast", "player", "thumbnail",
            "caption", "subtitle", "fullscreen",
        ],
        "selectors": [
            "video", "audio", "[class*='carousel']", "[class*='slider']",
            "[class*='gallery']", "[class*='player']",
            "iframe[src*='youtube']", "iframe[src*='vimeo']",
        ],
        "qa_focus": (
            "playback controls (play/pause/seek/volume), loading & buffering states, "
            "autoplay policy, captions/subtitles availability, "
            "responsive scaling across viewports, lazy-loading correctness, "
            "carousel accessibility"
        ),
    },
    "dashboard": {
        "label": "📊 Dashboard & Analytics",
        "keywords": [
            "dashboard", "analytics", "report", "chart", "graph", "metric",
            "kpi", "statistics", "overview", "summary", "widget", "export",
            "date range", "real-time",
        ],
        "selectors": [
            "[class*='dashboard']", "[class*='chart']", "[class*='graph']",
            "[class*='metric']", "[class*='widget']", "canvas", "svg",
        ],
        "qa_focus": (
            "data accuracy vs source-of-truth, chart rendering with zero/null data, "
            "date range filters & timezone handling, CSV/PDF export correctness, "
            "real-time refresh, permission-based widget visibility"
        ),
    },
    "notifications": {
        "label": "🔔 Notifications & Alerts",
        "keywords": [
            "alert", "notification", "toast", "banner", "message", "warning",
            "error", "success", "info", "badge", "unread", "dismiss",
            "push notification",
        ],
        "selectors": [
            "[role='alert']", "[class*='toast']", "[class*='notification']",
            "[class*='alert']", "[class*='banner']", "[class*='badge']",
        ],
        "qa_focus": (
            "trigger conditions for each notification type, manual dismissal, "
            "auto-dismiss timing, stacking of multiple toasts, "
            "screen-reader ARIA live region announcements, "
            "notification persistence across page reloads"
        ),
    },
    "accessibility": {
        "label": "♿ Accessibility & Internationalisation",
        "keywords": [
            "aria", "role", "lang", "alt", "skip", "focus", "keyboard",
            "screen reader", "contrast", "rtl", "locale", "i18n", "translate",
            "wcag", "tabindex",
        ],
        "selectors": [
            "[aria-label]", "[aria-describedby]", "[role]", "[lang]",
            "[alt]", "[tabindex]", "[class*='sr-only']", "[class*='visually-hidden']",
        ],
        "qa_focus": (
            "WCAG 2.2 AA compliance, keyboard trap detection, "
            "colour contrast ratios (4.5:1 normal text, 3:1 large text), "
            "screen-reader label completeness, RTL layout, "
            "locale-specific date/currency formatting"
        ),
    },
    "api_integrations": {
        "label": "🔌 API & Third-Party Integrations",
        "keywords": [
            "api", "webhook", "sdk", "integration", "connect", "sync",
            "import", "export", "stripe", "paypal", "google", "facebook",
            "maps", "recaptcha", "analytics", "twilio",
        ],
        "selectors": [
            "script[src*='stripe']", "script[src*='paypal']",
            "script[src*='google']", "script[src*='facebook']",
            "script[src*='recaptcha']", "[class*='g-recaptcha']",
        ],
        "qa_focus": (
            "API error handling & user-facing messages, third-party downtime fallbacks, "
            "data sync accuracy & conflict resolution, "
            "webhook retry logic, reCAPTCHA failure handling, "
            "SDK version compatibility"
        ),
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SECURITY: URL VALIDATION (SSRF Defence)
# ═════════════════════════════════════════════════════════════════════════════

_BLOCKED_HOSTNAMES = {
    "localhost", "127.0.0.1", "0.0.0.0",
    "169.254.169.254",          # AWS / GCP / Azure metadata service
    "metadata.google.internal",
}

MAX_RESPONSE_BYTES = 8 * 1024 * 1024  # 8 MB

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def validate_url(url: str) -> tuple[bool, str]:
    """Check whether a URL is safe to fetch. Returns (is_safe, reason)."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "URL could not be parsed."

    if parsed.scheme not in ("http", "https"):
        return False, f"Only http/https allowed. Got: '{parsed.scheme}'"

    hostname = (parsed.hostname or "").lower()

    if hostname in _BLOCKED_HOSTNAMES:
        return False, f"Hostname '{hostname}' is blocked for security."

    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False, f"Private/reserved IP '{hostname}' is not allowed."
    except ValueError:
        pass  # Regular domain name — fine

    return True, ""


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DEEP SCRAPER
#
# Extracts far more than the basic version:
#   Core      : title, meta tags, headings, buttons, inputs, prices
#   Structure : tables, breadcrumbs, pagination, tabs, modals, alerts
#   Media     : images (with alt check), videos, iframes
#   Data      : JSON-LD structured data, Open Graph meta tags
#   A11y      : ARIA roles/labels, skip links, lang, missing alts
#   Tech      : third-party scripts (Stripe, Analytics, etc.)
#   Areas     : scored detection for all 10 application domains
# ═════════════════════════════════════════════════════════════════════════════

def _dedup(items: list, limit: int) -> list:
    """Deduplicate a list preserving order, capped at limit."""
    seen, out = set(), []
    for item in items:
        key = str(item).lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
        if len(out) >= limit:
            break
    return out


def _detect_app_areas(soup: BeautifulSoup, page_text: str) -> dict:
    """
    Score each application area by keyword frequency + CSS selector hits.
    Returns: { area_key: { score, evidence, detected, label, qa_focus } }
    A score >= 4 means the area is confidently detected.
    """
    page_lower = page_text.lower()
    detection = {}

    for area_key, cfg in APP_AREAS.items():
        score = 0
        evidence = []

        # Keyword matches in full page text
        for kw in cfg["keywords"]:
            count = page_lower.count(kw.lower())
            if count > 0:
                score += min(count, 3)   # cap at 3 per keyword
                evidence.append(f'keyword "{kw}" x{count}')

        # CSS selector hits in the DOM (stronger signal than keywords)
        for sel in cfg["selectors"]:
            try:
                hits = soup.select(sel)
                if hits:
                    score += len(hits) * 2
                    evidence.append(f'selector [{sel}] = {len(hits)} element(s)')
            except Exception:
                pass

        detection[area_key] = {
            "score": score,
            "evidence": evidence[:5],
            "detected": score >= 4,      # confidence threshold
            "label": cfg["label"],
            "qa_focus": cfg["qa_focus"],
        }

    # Sort by score descending so highest-confidence areas appear first
    return dict(sorted(detection.items(), key=lambda x: x[1]["score"], reverse=True))


def scrape_page(url: str) -> dict:
    """
    Deep-scrape a URL and return a rich structured dict covering all
    application areas, media, accessibility signals, and structured data.
    """
    result = {
        "url": url,
        "meta": {},
        "headings": [],
        "buttons": [],
        "inputs": [],
        "prices": [],
        "tables": [],
        "pagination": False,
        "breadcrumbs": [],
        "tabs": [],
        "modals": [],
        "alerts": [],
        "images": [],
        "videos": False,
        "video_count": 0,
        "iframes": [],
        "third_party": [],
        "json_ld": [],
        "aria": {},
        "app_areas": {},
        "raw_text": "",
        "error": None,
    }

    # ── Validate URL before any network activity ──────────────────────────────
    ok, reason = validate_url(url)
    if not ok:
        result["error"] = f"URL blocked: {reason}"
        return result

    # ── Fetch page with streaming size cap ────────────────────────────────────
    try:
        resp = requests.get(
            url, headers=_BROWSER_HEADERS, timeout=15,
            allow_redirects=False, stream=True,
        )
        if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
            result["error"] = "URL redirects — please use the final destination URL."
            return result
        resp.raise_for_status()

        raw = b""
        for chunk in resp.iter_content(8192):
            raw += chunk
            if len(raw) > MAX_RESPONSE_BYTES:
                result["error"] = "Page exceeds 8 MB size limit."
                return result

    except requests.exceptions.ConnectionError:
        result["error"] = "Could not connect. Is the URL publicly accessible?"
        return result
    except requests.exceptions.Timeout:
        result["error"] = "Request timed out after 15 seconds."
        return result
    except requests.exceptions.HTTPError as e:
        result["error"] = f"HTTP error: {e}"
        return result

    soup = BeautifulSoup(raw, "html.parser")
    page_text = soup.get_text(separator=" ", strip=True)
    result["raw_text"] = page_text[:5000]

    # ── Meta tags ─────────────────────────────────────────────────────────────
    desc_tag = soup.find("meta", attrs={"name": "description"})
    og_title = soup.find("meta", property="og:title")
    og_type  = soup.find("meta", property="og:type")
    canonical = soup.find("link", rel="canonical")

    result["meta"] = {
        "title":       soup.title.get_text(strip=True) if soup.title else url,
        "description": desc_tag.get("content", "") if desc_tag else "",
        "lang":        soup.html.get("lang", "") if soup.html else "",
        "canonical":   canonical.get("href", "") if canonical else "",
        "og_title":    og_title.get("content", "") if og_title else "",
        "og_type":     og_type.get("content", "") if og_type else "",
    }

    # ── Headings ──────────────────────────────────────────────────────────────
    raw_h = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5"]):
        t = tag.get_text(strip=True)
        if t and len(t) < 150:
            raw_h.append(t)
    result["headings"] = _dedup(raw_h, 25)

    # ── Buttons / clickable elements ──────────────────────────────────────────
    _noise = {
        "home", "menu", "toggle", "close", "open", "skip", "x", "cookie",
        "accept all", "deny", "back to top", "more", "less",
    }
    raw_btns = []
    for el in soup.find_all(["button", "a"]):
        t = (el.get_text(strip=True) or el.get("aria-label") or "").strip()
        if t and 2 < len(t) < 60 and t.lower() not in _noise:
            raw_btns.append(t)
    result["buttons"] = _dedup(raw_btns, 40)

    # ── Inputs (enriched: type, required, placeholder) ────────────────────────
    raw_inputs = []
    for inp in soup.find_all(["input", "select", "textarea"]):
        if inp.get("type") in ("hidden", "submit", "button", "reset", "image"):
            continue
        label = (
            inp.get("placeholder") or inp.get("aria-label") or
            inp.get("name") or inp.get("id") or inp.get("type") or ""
        ).strip()
        if label:
            req = " [required]" if inp.has_attr("required") else ""
            raw_inputs.append(f'{label} ({inp.get("type", inp.name)}){req}')
    result["inputs"] = _dedup(raw_inputs, 25)

    # ── Prices (currency symbol detection) ───────────────────────────────────
    raw_prices = []
    for t in soup.stripped_strings:
        if any(s in t for s in ("$", "£", "€", "₹", "¥")) and len(t) < 30:
            raw_prices.append(t.strip())
    result["prices"] = _dedup(raw_prices, 25)

    # ── Tables ────────────────────────────────────────────────────────────────
    for table in soup.find_all("table")[:10]:
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        rows = table.find_all("tr")
        result["tables"].append({
            "headers": headers[:10],
            "row_count": len(rows),
        })

    # ── Pagination ────────────────────────────────────────────────────────────
    pg = soup.select(
        "[class*='pagination'], [class*='pager'], [aria-label*='pagination' i], "
        "a[href*='page='], a[href*='p=']"
    )
    result["pagination"] = len(pg) > 0

    # ── Breadcrumbs ───────────────────────────────────────────────────────────
    crumb_els = soup.select(
        "[class*='breadcrumb'], [aria-label*='breadcrumb' i]"
    )
    crumbs = []
    for el in crumb_els:
        for child in el.find_all(["a", "li", "span"]):
            t = child.get_text(strip=True)
            if t and len(t) < 60:
                crumbs.append(t)
    result["breadcrumbs"] = _dedup(crumbs, 10)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_els = soup.select(
        "[role='tab'], [class*='tab-'], [class*='-tab'], [class*='tabs'] a"
    )
    result["tabs"] = _dedup(
        [el.get_text(strip=True) for el in tab_els if el.get_text(strip=True)], 15
    )

    # ── Modals (by trigger button attributes) ─────────────────────────────────
    modal_triggers = soup.select(
        "[data-toggle='modal'], [data-bs-toggle='modal'], [class*='modal-trigger']"
    )
    result["modals"] = _dedup(
        [el.get_text(strip=True) or el.get("aria-label", "") for el in modal_triggers], 10
    )

    # ── Alerts / banners ──────────────────────────────────────────────────────
    alert_els = soup.select(
        "[role='alert'], [class*='alert'], [class*='toast'], [class*='banner']"
    )
    result["alerts"] = _dedup(
        [el.get_text(strip=True)[:80] for el in alert_els if el.get_text(strip=True)], 10
    )

    # ── Images (check for alt text — important for a11y) ─────────────────────
    for img in soup.find_all("img")[:30]:
        alt = img.get("alt", "").strip()
        src = img.get("src") or img.get("data-src") or ""
        result["images"].append({
            "alt": alt or "(no alt attribute)",
            "has_alt": bool(alt),
            "src_hint": src.split("/")[-1][:40] if src else "",
        })

    # ── Video detection ───────────────────────────────────────────────────────
    videos = soup.find_all(["video", "iframe"])
    vid_count = sum(
        1 for v in videos
        if v.name == "video" or any(
            kw in (v.get("src") or "")
            for kw in ("youtube", "vimeo", "wistia", "loom", "dailymotion")
        )
    )
    result["videos"] = vid_count > 0
    result["video_count"] = vid_count

    # ── iFrames ───────────────────────────────────────────────────────────────
    for iframe in soup.find_all("iframe")[:10]:
        src = iframe.get("src", "")
        if src:
            result["iframes"].append(src[:80])

    # ── Third-party script detection ──────────────────────────────────────────
    _tp_map = {
        "stripe":            "Stripe (payments)",
        "paypal":            "PayPal (payments)",
        "google-analytics":  "Google Analytics",
        "googletagmanager":  "Google Tag Manager",
        "facebook":          "Facebook SDK",
        "recaptcha":         "Google reCAPTCHA",
        "intercom":          "Intercom (live chat)",
        "zendesk":           "Zendesk (support)",
        "hotjar":            "Hotjar (heatmaps)",
        "segment":           "Segment (analytics)",
        "mixpanel":          "Mixpanel (analytics)",
        "hubspot":           "HubSpot (CRM/marketing)",
        "sentry":            "Sentry (error tracking)",
        "maps.googleapis":   "Google Maps",
        "twilio":            "Twilio (messaging)",
        "cloudflare":        "Cloudflare",
    }
    scripts_text = " ".join(
        (s.get("src") or "") for s in soup.find_all("script")
    ).lower()
    result["third_party"] = [
        label for key, label in _tp_map.items() if key in scripts_text
    ]

    # ── JSON-LD structured data ───────────────────────────────────────────────
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "{}")
            if data:
                result["json_ld"].append({
                    "type": data.get("@type", "Unknown"),
                    "name": data.get("name", ""),
                })
        except json.JSONDecodeError:
            pass

    # ── Accessibility signals ─────────────────────────────────────────────────
    result["aria"] = {
        "lang": result["meta"]["lang"],
        "skip_links": len(soup.select("a[href='#main'], a[href='#content'], [class*='skip']")),
        "aria_labels": len(soup.select("[aria-label]")),
        "aria_roles": len(soup.select("[role]")),
        "missing_alts": sum(1 for img in result["images"] if not img["has_alt"]),
        "total_images": len(result["images"]),
        "form_labels": len(soup.select("label[for]")),
        "tabindex_els": len(soup.select("[tabindex]")),
    }

    # ── Application area detection (scored) ───────────────────────────────────
    result["app_areas"] = _detect_app_areas(soup, page_text)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PROMPT BUILDER
#
# Builds a model-aware (system_prompt, user_prompt) pair.
# System prompt changes personality based on the model's strength.
# User prompt includes all scraped data + targeted area QA focus.
# ═════════════════════════════════════════════════════════════════════════════

MODEL_PERSONAS = {
    "llama-3.3-70b-versatile": (
        "You are a senior QA engineer with 10+ years of experience across web, "
        "mobile, and enterprise applications. You write balanced, comprehensive "
        "test requirements covering happy paths, edge cases, and cross-browser concerns."
    ),
    "llama-3.1-8b-instant": (
        "You are a QA analyst focused on rapid test planning. You write concise, "
        "actionable test requirements prioritised by risk. Keep each requirement "
        "short and specific — no padding."
    ),
    "llama-3.3-70b-specdec": (
        "You are a meticulous QA architect specialising in edge-case and failure-mode "
        "discovery. You think adversarially: what will break? What did the developer "
        "forget? Write granular, defensive requirements that expose hidden assumptions."
    ),
    "qwen/qwen3-32b": (
        "You are a QA engineer specialising in internationalised and multilingual "
        "web applications. Pay close attention to locale handling, character encoding, "
        "RTL layouts, date/currency formatting, and i18n compliance throughout."
    ),
    "openai/gpt-oss-120b": (
        "You are a principal QA architect with deep expertise in enterprise software "
        "compliance, WCAG 2.2 accessibility standards, API contract validation, and "
        "formal traceability. Write rigorous, structured requirements with risk ratings."
    ),
}


def build_prompt(page_data: dict, model_key: str, focus_areas: list) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) tailored to the chosen model."""
    system_prompt = MODEL_PERSONAS.get(model_key, MODEL_PERSONAS["llama-3.3-70b-versatile"])

    meta = page_data.get("meta", {})
    areas = page_data.get("app_areas", {})
    aria  = page_data.get("aria", {})

    # Determine which areas to include in the prompt
    if focus_areas:
        # Use explicitly selected areas + any auto-detected ones
        relevant = {k: v for k, v in areas.items() if k in focus_areas or v["detected"]}
    else:
        # Auto-detect only
        relevant = {k: v for k, v in areas.items() if v["detected"]}

    # If nothing detected and nothing selected, include top 3 by score
    if not relevant:
        relevant = dict(list(areas.items())[:3])

    area_lines = "\n".join(
        f"  • {v['label']}: {v['qa_focus']}" for v in relevant.values()
    )

    def fmt(items, limit=15):
        if not items:
            return "  (none detected)"
        return "\n".join(f"  • {i}" for i in items[:limit])

    tables_txt = "\n".join(
        f"  • {t['row_count']} rows | columns: {', '.join(t['headers'][:5]) or 'unnamed'}"
        for t in page_data.get("tables", [])
    ) or "  (none)"

    a11y_txt = (
        f"  • Page lang attribute : {aria.get('lang') or 'NOT SET ⚠️'}\n"
        f"  • Skip links          : {aria.get('skip_links', 0)}\n"
        f"  • Elements with ARIA labels : {aria.get('aria_labels', 0)}\n"
        f"  • Elements with ARIA roles  : {aria.get('aria_roles', 0)}\n"
        f"  • Images missing alt text   : {aria.get('missing_alts', 0)}"
        f" / {aria.get('total_images', 0)}\n"
        f"  • <label for> elements      : {aria.get('form_labels', 0)}\n"
        f"  • Elements with tabindex    : {aria.get('tabindex_els', 0)}"
    )

    third_party = ", ".join(page_data.get("third_party", [])) or "none detected"

    json_ld_txt = "\n".join(
        f"  • {j['type']}: {j['name']}" for j in page_data.get("json_ld", [])
    ) or "  (none)"

    user_prompt = f"""
ANALYSED PAGE
=============
URL       : {page_data['url']}
Title     : {meta.get('title', '')}
Language  : {meta.get('lang') or 'NOT SET'}
OG Type   : {meta.get('og_type') or 'N/A'}
Canonical : {meta.get('canonical') or 'N/A'}

DETECTED APPLICATION AREAS — Write requirements for ALL of these:
=================================================================
{area_lines}

HEADINGS (page structure)
--------------------------
{fmt(page_data.get('headings', []))}

INTERACTIVE BUTTONS / LINKS
----------------------------
{fmt(page_data.get('buttons', []))}

FORM INPUTS  ([required] = mandatory field)
-------------------------------------------
{fmt(page_data.get('inputs', []))}

PRICES DETECTED
---------------
{fmt(page_data.get('prices', []))}

DATA TABLES
-----------
{tables_txt}

NAVIGATION
----------
Tabs        : {', '.join(page_data.get('tabs', [])) or 'none'}
Breadcrumbs : {' > '.join(page_data.get('breadcrumbs', [])) or 'none'}
Pagination  : {'YES' if page_data.get('pagination') else 'NO'}

MODALS / OVERLAYS
-----------------
{fmt(page_data.get('modals', []))}

ALERTS / BANNERS
----------------
{fmt(page_data.get('alerts', []))}

MEDIA
-----
Videos/Embeds : {'YES — ' + str(page_data.get('video_count', 0)) + ' detected' if page_data.get('videos') else 'none'}
iFrames       : {fmt(page_data.get('iframes', []), limit=5)}

THIRD-PARTY INTEGRATIONS
-------------------------
{third_party}

STRUCTURED DATA (JSON-LD)
--------------------------
{json_ld_txt}

ACCESSIBILITY SIGNALS
---------------------
{a11y_txt}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generate QA requirements for EACH detected application area listed above.
Group them clearly with a heading per area.

FORMAT for every requirement:
  [REQ-NNN] <Short descriptive title>
  Given : <precondition / system state>
  When  : <user action or system event>
  Then  : <specific, measurable expected outcome>
  Risk  : <Low | Medium | High>

RULES:
  1. Number requirements sequentially REQ-001, REQ-002 … across all areas
  2. Write a minimum of 4 requirements per detected area
  3. Reference ACTUAL element names from the data above (exact button labels, field names)
  4. Include at least 1 negative/failure-path test per area
  5. For accessibility: reference WCAG 2.2 success criteria (e.g. SC 1.1.1, SC 2.4.3)
  6. For third-party integrations: include a downtime/fallback scenario test
  7. End the response with a SUMMARY TABLE:
     | Area | Req Count | Highest Risk |
""".strip()

    return system_prompt, user_prompt


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — AI GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_qa(
    page_data: dict,
    model_key: str,
    focus_areas: list,
    api_key: str,
) -> str:
    """Call Groq API with retry on rate-limit. Returns QA requirement text."""
    system_prompt, user_prompt = build_prompt(page_data, model_key, focus_areas)
    cfg = MODELS[model_key]
    client = Groq(api_key=api_key)

    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=model_key,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
            )
            return resp.choices[0].message.content

        except Exception as exc:
            err = str(exc).lower()
            if "rate_limit" in err or "429" in err:
                wait = 2 ** attempt
                st.warning(f"Rate limit hit (attempt {attempt}/3). Retrying in {wait}s…")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Groq API error: {exc}") from exc

    raise RuntimeError("Rate limit exceeded after 3 retries. Please wait a moment.")


def load_api_key() -> Optional[str]:
    """Load Groq API key from Streamlit Secrets or environment variable."""
    return st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — STREAMLIT UI
# ═════════════════════════════════════════════════════════════════════════════

def render_scrape_summary(data: dict):
    """Show visual summary of what was scraped."""
    aria  = data.get("aria", {})
    areas = data.get("app_areas", {})

    # Top-level metrics
    st.markdown("#### 📊 Scrape Summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Headings",      len(data.get("headings", [])))
    c2.metric("Buttons",       len(data.get("buttons",  [])))
    c3.metric("Inputs",        len(data.get("inputs",   [])))
    c4.metric("Prices",        len(data.get("prices",   [])))
    c5.metric("3rd Party",     len(data.get("third_party", [])))
    c6.metric("Missing ALTs",  aria.get("missing_alts", 0))

    # Application area confidence bars
    st.markdown("#### 🎯 Detected Application Areas")
    detected = [(k, v) for k, v in areas.items() if v["score"] > 0]
    if detected:
        max_score = max(v["score"] for _, v in detected)
        for area_key, area_val in detected[:8]:
            pct = min(int((area_val["score"] / max(max_score, 1)) * 100), 100)
            badge = "🟢 Detected" if area_val["detected"] else ("🟡 Partial" if pct > 15 else "⚪ Weak")
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(f"**{area_val['label']}**")
                st.progress(pct / 100)
            with col_b:
                st.markdown(f"<br><small>{badge}</small>", unsafe_allow_html=True)
    else:
        st.info("No strong application area signals detected.")

    # Raw detail tabs
    with st.expander("📋 Full Scrape Data (click to expand)", expanded=False):
        t1, t2, t3, t4, t5, t6, t7 = st.tabs(
            ["Buttons", "Inputs", "Prices", "Tables", "A11y", "3rd Party", "Structured Data"]
        )
        with t1: st.write(data.get("buttons", []) or "None")
        with t2: st.write(data.get("inputs",  []) or "None")
        with t3: st.write(data.get("prices",  []) or "None")
        with t4:
            for t in data.get("tables", []):
                st.write(f"**{t['row_count']} rows** | cols: {', '.join(t['headers']) or 'unnamed'}")
            if not data.get("tables"):
                st.write("No tables detected.")
        with t5: st.json(aria)
        with t6: st.write(data.get("third_party", []) or "None detected")
        with t7: st.write(data.get("json_ld", []) or "None found")


def render_ui():
    st.set_page_config(
        page_title="Advanced GenAI QA Generator",
        page_icon="🧪",
        layout="wide",
    )

    st.markdown("""
    <style>
        .model-card {
            background: #f0f4ff;
            border-radius: 8px;
            padding: 10px 14px;
            border-left: 4px solid #3b6cf8;
            margin: 6px 0 12px 0;
            font-size: 0.88rem;
        }
        .stProgress > div > div { border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🧪 Advanced GenAI QA Requirement Generator")
    st.markdown(
        "**Deep-scrapes any public web page · Detects 10 application areas · "
        "5 AI models · Side-by-side comparison · Export as .txt or .md**"
    )
    st.divider()

    # ── API Key check ─────────────────────────────────────────────────────────
    api_key = load_api_key()
    if not api_key:
        st.error(
            "**Groq API key not found.**\n\n"
            "**Streamlit Cloud:** Advanced Settings → Secrets → add:\n"
            "```\nGROQ_API_KEY = \"gsk_your_key_here\"\n```\n\n"
            "**Local development:** `export GROQ_API_KEY='gsk_your_key_here'`"
        )
        st.stop()

    # ── Two-column layout ─────────────────────────────────────────────────────
    cfg_col, out_col = st.columns([1, 2], gap="large")

    with cfg_col:
        st.subheader("⚙️ Configuration")

        # URL input
        url = st.text_input(
            "🌐 Website URL",
            placeholder="https://example.com/page",
            help="Any publicly accessible URL. Private IPs and localhost are blocked.",
        )

        st.markdown("---")

        # Model selector
        st.markdown("**🤖 Primary AI Model**")
        model_key = st.selectbox(
            "primary_model",
            options=list(MODELS.keys()),
            format_func=lambda k: f"{MODELS[k]['icon']} {MODELS[k]['label']}",
            label_visibility="collapsed",
        )
        m = MODELS[model_key]
        st.markdown(
            f"<div class='model-card'>"
            f"<strong>{m['strength']}</strong> · ⚡ {m['speed']}<br>"
            f"{m['description']}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Compare mode
        compare = st.checkbox("🔀 Compare two models side-by-side")
        model_key_2 = None
        if compare:
            st.markdown("**Second Model**")
            model_key_2 = st.selectbox(
                "second_model",
                options=[k for k in MODELS if k != model_key],
                format_func=lambda k: f"{MODELS[k]['icon']} {MODELS[k]['label']}",
                label_visibility="collapsed",
            )
            m2 = MODELS[model_key_2]
            st.markdown(
                f"<div class='model-card'>"
                f"<strong>{m2['strength']}</strong> · ⚡ {m2['speed']}<br>"
                f"{m2['description']}"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Focus area selector
        st.markdown("**🎯 Application Area Focus**")
        st.caption("Optional — leave blank to auto-detect from page content.")
        focus_areas = st.multiselect(
            "focus_areas",
            options=list(APP_AREAS.keys()),
            format_func=lambda k: APP_AREAS[k]["label"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        generate_btn = st.button(
            "🚀 Generate QA Requirements",
            type="primary",
            use_container_width=True,
        )

    # ── Output column ─────────────────────────────────────────────────────────
    with out_col:
        st.subheader("📊 Results")

        if not generate_btn:
            st.info(
                "👈 Fill in the configuration on the left, then click **Generate**.\n\n"
                "**What happens:**\n"
                "1. 🔍 Deep-scrape the page (detects 10 application areas)\n"
                "2. 🎯 Score each area by keyword + selector evidence\n"
                "3. 🤖 Build a model-specific prompt using all scraped data\n"
                "4. 📄 Generate Given/When/Then test requirements per area\n"
                "5. ⬇️ Download as `.txt` or `.md`"
            )
            st.stop()

        # Validate URL
        if not url or not url.strip():
            st.warning("Please enter a URL first.")
            st.stop()

        url = url.strip()
        ok, reason = validate_url(url)
        if not ok:
            st.error(f"🚫 URL blocked: {reason}")
            st.stop()

        # Scrape
        with st.spinner("🔍 Deep-scraping page and detecting application areas…"):
            page_data = scrape_page(url)

        if page_data["error"]:
            st.error(f"❌ Scraping failed: {page_data['error']}")
            st.stop()

        render_scrape_summary(page_data)
        st.divider()

        # Generate
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        if compare and model_key_2:
            st.markdown("### 🔀 Model Comparison")
            col1, col2 = st.columns(2)

            out1 = out2 = ""

            with col1:
                m1_cfg = MODELS[model_key]
                st.markdown(f"**{m1_cfg['icon']} {m1_cfg['label']}**")
                st.caption(f"Strength: {m1_cfg['strength']}")
                with st.spinner("Generating…"):
                    try:
                        out1 = generate_qa(page_data, model_key, focus_areas, api_key)
                        st.success("✅ Done")
                        st.markdown(out1)
                    except RuntimeError as e:
                        st.error(str(e))

            with col2:
                m2_cfg = MODELS[model_key_2]
                st.markdown(f"**{m2_cfg['icon']} {m2_cfg['label']}**")
                st.caption(f"Strength: {m2_cfg['strength']}")
                with st.spinner("Generating…"):
                    try:
                        out2 = generate_qa(page_data, model_key_2, focus_areas, api_key)
                        st.success("✅ Done")
                        st.markdown(out2)
                    except RuntimeError as e:
                        st.error(str(e))

            # Download both
            if out1 or out2:
                combined = (
                    f"QA Comparison — {url}\nGenerated: {ts} UTC\n{'='*60}\n\n"
                    f"=== Model 1: {MODELS[model_key]['label']} ===\n\n{out1}\n\n"
                    f"=== Model 2: {MODELS[model_key_2]['label']} ===\n\n{out2}"
                )
                d1, d2 = st.columns(2)
                with d1:
                    st.download_button(
                        "⬇️ Download comparison (.txt)",
                        data=combined, file_name=f"qa_comparison_{ts}.txt",
                        mime="text/plain", use_container_width=True,
                    )
                with d2:
                    st.download_button(
                        "⬇️ Download comparison (.md)",
                        data=combined, file_name=f"qa_comparison_{ts}.md",
                        mime="text/markdown", use_container_width=True,
                    )

        else:
            # Single model output
            st.markdown(f"### 📄 QA Requirements — {MODELS[model_key]['icon']} {MODELS[model_key]['label']}")
            st.caption(f"Strength: {MODELS[model_key]['strength']} · Speed: {MODELS[model_key]['speed']}")

            with st.spinner(f"🤖 Generating with {MODELS[model_key]['label']}…"):
                try:
                    output = generate_qa(page_data, model_key, focus_areas, api_key)
                except RuntimeError as e:
                    st.error(f"❌ {e}")
                    st.stop()

            st.success("✅ Generation complete")
            st.markdown(output)

            header = (
                f"QA Requirements for: {url}\n"
                f"Model: {MODELS[model_key]['label']}\n"
                f"Generated: {ts} UTC\n{'='*60}\n\n"
            )
            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "⬇️ Download as .txt",
                    data=header + output,
                    file_name=f"qa_requirements_{ts}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with d2:
                st.download_button(
                    "⬇️ Download as .md",
                    data=header + output,
                    file_name=f"qa_requirements_{ts}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    render_ui()
