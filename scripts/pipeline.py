"""
pipeline.py  —  News-to-video content pipeline
================================================
LLM fallback chain (priority order):
  1. Antigravity Manager  (Docker, port 8045)          — API_KEY / API_BASE_URL
  2. GitHub Models        (gpt-4o via Azure endpoint)  — GH_MODELS_TOKEN
  3. Cloudflare Workers AI                             — CF_ACCOUNT_ID / CF_AI_TOKEN
  4. Ollama               (local llama3.2:3b)          — OLLAMA_BASE_URL / OLLAMA_MODEL

Monthly report:
  - Each run appends a JSON line to pipeline_stats_YYYY_MM.jsonl on Drive.
  - Set env GENERATE_MONTHLY_REPORT=true (or run on the 1st of the month)
    to generate/overwrite monthly_report_YYYY_MM.md on Drive.
"""

import os
import sys
import urllib.request
import json
import time
import subprocess
from datetime import datetime, timedelta, timezone
import re
import base64
import io
import random
import pickle
from collections import Counter, defaultdict
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# LLM SOURCE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
# Source 1 — Antigravity Manager (primary)
API_KEY         = os.environ.get("API_KEY", "password")
API_BASE_URL    = os.environ.get("API_BASE_URL", "http://127.0.0.1:8045/v1")

# Source 2 — GitHub Models (cloud fallback)
GH_MODELS_TOKEN    = os.environ.get("GH_MODELS_TOKEN", "")
GH_MODELS_BASE_URL = os.environ.get("GH_MODELS_BASE_URL", "https://models.inference.ai.azure.com")
GH_MODEL           = os.environ.get("GH_MODEL", "gpt-4o")

# Source 3 — Cloudflare Workers AI (multi-account pool)
# Secret format — one JSON array covers all accounts:
#   CLOUDFLARE_ACCOUNTS_JSON = '[{"id":"abc123","token":"tok1"},{"id":"def456","token":"tok2"}]'
# Falls back to legacy single-pair env vars if the JSON secret is absent.
CF_AI_MODEL = os.environ.get("CF_AI_MODEL", "@cf/meta/llama-3.1-8b-instruct")

def _load_cf_accounts() -> list[dict]:
    """
    Load the Cloudflare account pool from env.  Returns a list of
    {"id": "...", "token": "..."} dicts, deduplicated by account id.

    Sources checked (first match wins per account id):
      1. CLOUDFLARE_ACCOUNTS_JSON  — JSON array  [{"id":…,"token":…}, …]
      2. Numbered pairs            — CLOUDFLARE_ACCOUNT_ID_N / CLOUDFLARE_API_TOKEN_N
      3. Legacy single pair        — CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN
      4. Pipeline convention       — CF_ACCOUNT_ID / CF_AI_TOKEN
    """
    raw: list[tuple[str, str]] = []

    json_str = os.environ.get("CLOUDFLARE_ACCOUNTS_JSON", "").strip()
    if json_str:
        try:
            for acc in json.loads(json_str):
                aid   = acc.get("id") or acc.get("account_id") or ""
                token = acc.get("token") or acc.get("api_token") or ""
                if aid and token:
                    raw.append((aid, token))
        except Exception as e:
            print(f"   ⚠️  CLOUDFLARE_ACCOUNTS_JSON parse error: {e}")

    for key in sorted(os.environ):
        if key.startswith("CLOUDFLARE_ACCOUNT_ID_"):
            idx   = key[len("CLOUDFLARE_ACCOUNT_ID_"):]
            token = os.environ.get(f"CLOUDFLARE_API_TOKEN_{idx}", "")
            aid   = os.environ.get(key, "")
            if aid and token:
                raw.append((aid, token))

    for aid, tok in [
        (os.environ.get("CLOUDFLARE_ACCOUNT_ID", ""), os.environ.get("CLOUDFLARE_API_TOKEN", "")),
        (os.environ.get("CF_ACCOUNT_ID", ""),         os.environ.get("CF_AI_TOKEN", "")),
    ]:
        if aid and tok:
            raw.append((aid, tok))

    seen: dict[str, str] = {}
    for aid, tok in raw:
        if aid not in seen:
            seen[aid] = tok
    return [{"id": k, "token": v} for k, v in seen.items()]


def _ping_cf_account(account_id: str, api_token: str) -> bool:
    """Return True if this CF account can serve a 1-token request right now."""
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/ai/run/@cf/meta/llama-3-8b-instruct"
    )
    payload = json.dumps({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type":  "application/json",
    }
    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return bool(body.get("success"))
    except Exception:
        return False


# Runtime state: tracks which CF account index to try next across calls
_cf_accounts: list[dict] = []          # populated lazily on first use
_cf_account_idx: int     = 0           # round-robin cursor
_cf_exhausted:   bool    = False       # set True once all accounts are at limit

def _get_cf_account() -> tuple[str, str] | None:
    """
    Return (account_id, api_token) for the next available Cloudflare account,
    rotating through the pool and skipping accounts that are at their limit.
    Returns None if every account in the pool is exhausted.
    """
    global _cf_accounts, _cf_account_idx, _cf_exhausted

    if _cf_exhausted:
        return None

    if not _cf_accounts:
        _cf_accounts = _load_cf_accounts()
        if not _cf_accounts:
            return None

    n = len(_cf_accounts)
    for _ in range(n):
        acc = _cf_accounts[_cf_account_idx % n]
        _cf_account_idx = (_cf_account_idx + 1) % n
        if not acc.get("_exhausted"):
            return acc["id"], acc["token"]

    _cf_exhausted = True
    return None


def _mark_cf_exhausted(account_id: str):
    """Flag a specific account as at-limit so it's skipped in future calls."""
    global _cf_exhausted
    for acc in _cf_accounts:
        if acc["id"] == account_id:
            acc["_exhausted"] = True
            print(f"   ⚠️  CF account {account_id[:12]}… marked exhausted (limit reached)")
            break
    if all(acc.get("_exhausted") for acc in _cf_accounts):
        _cf_exhausted = True
        print("   ⚠️  All Cloudflare accounts exhausted for this run")

# Source 4 — Ollama (local last resort)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

MAX_RETRIES  = 3
TOPIC_LIMIT  = int(os.environ.get("TOPIC_LIMIT", "30"))

ART_PERCENTAGE             = float(os.environ.get("ART_PERCENTAGE", "0.33"))
ANTHROPOMORPHIC_PERCENTAGE = float(os.environ.get("ANTHROPOMORPHIC_PERCENTAGE", "0.3"))

_ARTISTIC_STYLES_DEFAULT = (
    "Studio Ghibli watercolor illustration style|"
    "Pixar 3D cinematic render style|"
    "Synthwave retro 80s neon poster style|"
    "Ukiyo-e Japanese woodblock print style|"
    "Marvel Comics bold ink and color style|"
    "LEGO brick diorama style|"
    "Impressionist oil painting style (Monet)|"
    "GTA V loading screen poster style|"
    "Cyberpunk 2077 concept art style|"
    "Disney golden-age hand-drawn animation style"
)
POPULAR_ARTISTIC_STYLES = [
    s.strip()
    for s in os.environ.get("POPULAR_ARTISTIC_STYLES", _ARTISTIC_STYLES_DEFAULT).split("|")
    if s.strip()
]
_artistic_style_index = 0

# ──────────────────────────────────────────────────────────────────────────────
# RUN STATS  (accumulated during main(), flushed by save_run_stats())
# ──────────────────────────────────────────────────────────────────────────────
_stats = {
    "llm_calls":       0,
    "llm_source_hits": {},   # {"Antigravity": 12, "GitHub Models": 3, ...}
    "topics_fetched":  0,
    "topics_approved": 0,
    "topics_skipped":  0,
    "images_ok":       0,
    "images_failed":   0,
    "audio_ok":        0,
    "errors":          [],
}

def _stat_hit(source: str):
    _stats["llm_calls"] += 1
    _stats["llm_source_hits"][source] = _stats["llm_source_hits"].get(source, 0) + 1

def _stat_error(msg: str):
    _stats["errors"].append(msg)

# ──────────────────────────────────────────────────────────────────────────────
# MULTI-SOURCE TEXT GENERATION
# ──────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = "You are an expert video script writer producing engaging scripts for YouTube Shorts/TikTok."


def _call_openai_compat(base_url: str, api_key: str, model: str, prompt: str) -> str:
    """Generic OpenAI-compatible /v1/chat/completions call."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=90) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"]


def _call_cloudflare_ai(prompt: str) -> str:
    """
    Cloudflare Workers AI — rotates through the account pool automatically.
    Marks accounts as exhausted on 429 / neuron-limit errors and retries
    with the next account in the pool.
    """
    from urllib.error import HTTPError

    while True:
        pair = _get_cf_account()
        if pair is None:
            raise RuntimeError("All Cloudflare accounts exhausted or none configured")

        account_id, api_token = pair
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
            f"/ai/run/{CF_AI_MODEL}"
        )
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_token}",
        }
        payload = json.dumps({
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        }).encode("utf-8")

        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=90) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            if not result.get("success"):
                errors = result.get("errors", [])
                err_str = str(errors).lower()
                if "neuron" in err_str or "daily" in err_str or "limit" in err_str:
                    _mark_cf_exhausted(account_id)
                    continue   # try next account
                raise RuntimeError(f"CF AI error on {account_id[:12]}…: {errors}")

            print(f"      ✅ [Cloudflare AI] account {account_id[:12]}… OK")
            return result["result"]["response"]

        except HTTPError as e:
            if e.code == 429:
                _mark_cf_exhausted(account_id)
                continue   # try next account
            try:
                body = json.loads(e.read().decode("utf-8"))
                err_str = str(body.get("errors", "")).lower()
                if "neuron" in err_str or "daily" in err_str or "limit" in err_str:
                    _mark_cf_exhausted(account_id)
                    continue
            except Exception:
                pass
            raise RuntimeError(f"CF AI HTTP {e.code} on {account_id[:12]}…: {e.reason}")


def generate_text(prompt: str) -> str:
    """
    Try each LLM source in priority order; return first success.
    Priority: Antigravity (1) -> GitHub Models (2) -> Cloudflare AI (3) -> Ollama (4)
    Stats recorded per source for monthly report.
    """
    sources = [
        ("Antigravity",   lambda: _call_openai_compat(API_BASE_URL, API_KEY, "gemini-3-flash", prompt)),
        ("GitHub Models", lambda: _call_openai_compat(GH_MODELS_BASE_URL, GH_MODELS_TOKEN, GH_MODEL, prompt)
                                  if GH_MODELS_TOKEN
                                  else (_ for _ in ()).throw(ValueError("GH_MODELS_TOKEN not set"))),
        ("Cloudflare AI", lambda: _call_cloudflare_ai(prompt)),
        ("Ollama",        lambda: _call_openai_compat(OLLAMA_BASE_URL, "ollama", OLLAMA_MODEL, prompt)),
    ]

    last_err = None
    for name, fn in sources:
        try:
            print(f"   🤖 [{name}] generating text...")
            text = fn()
            _stat_hit(name)
            print(f"      ✅ [{name}] OK")
            return text
        except Exception as exc:
            print(f"      ⚠️  [{name}] failed: {exc}")
            last_err = exc

    msg = f"All LLM sources exhausted. Last error: {last_err}"
    _stat_error(msg)
    print(f"❌ {msg}")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# GOOGLE DRIVE HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def get_drive_service():
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError:
        print("Google API clients not installed.")
        return None

    FALLBACK_SCOPES = [
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = None
    token_file = None
    for path in ["token.json", os.path.expanduser("~/.api_tools/token.json")]:
        if os.path.exists(path):
            token_file = path
            break

    if token_file:
        try:
            from google.auth.transport.requests import Request
            with open(token_file, "rb") as f:
                content = f.read()

            if content.startswith(b"\x80"):
                print("🔄 Detected Pickle token — converting to JSON...")
                pkl_creds = pickle.loads(content)
                raw_scopes = getattr(pkl_creds, "_scopes", getattr(pkl_creds, "scopes", None))
                granted_scopes = list(raw_scopes) if raw_scopes else FALLBACK_SCOPES
                creds_dict = {
                    "token":         getattr(pkl_creds, "token", None),
                    "refresh_token": getattr(pkl_creds, "_refresh_token",
                                     getattr(pkl_creds, "refresh_token", None)),
                    "token_uri":     getattr(pkl_creds, "_token_uri",
                                     getattr(pkl_creds, "token_uri", "https://oauth2.googleapis.com/token")),
                    "client_id":     getattr(pkl_creds, "_client_id",
                                     getattr(pkl_creds, "client_id", None)),
                    "client_secret": getattr(pkl_creds, "_client_secret",
                                     getattr(pkl_creds, "client_secret", None)),
                    "scopes":        granted_scopes,
                }
                with open(token_file, "w", encoding="utf-8") as f:
                    json.dump(creds_dict, f, indent=2)
                creds = Credentials.from_authorized_user_info(creds_dict, granted_scopes)
            else:
                creds_dict = json.loads(content)
                granted_scopes = creds_dict.get("scopes", FALLBACK_SCOPES)
                if isinstance(granted_scopes, str):
                    granted_scopes = granted_scopes.split()
                creds = Credentials.from_authorized_user_info(creds_dict, granted_scopes)

            if creds and creds.expired and creds.refresh_token:
                print("♻️  Refreshing expired token...")
                creds.refresh(Request())
                with open(token_file, "w", encoding="utf-8") as f:
                    f.write(creds.to_json())
        except Exception as e:
            print(f"❌ Auth error: {e}")

    if not creds or not creds.valid:
        print("⚠️  Google Drive not authenticated.")
        return None

    from googleapiclient.discovery import build
    return build("drive", "v3", credentials=creds)


def get_pacific_time():
    return datetime.now(timezone(timedelta(hours=-7)))


# ──────────────────────────────────────────────────────────────────────────────
# DRIVE: OPINIONS SYNC
# ──────────────────────────────────────────────────────────────────────────────
def sync_opinions_from_drive(service):
    if not service:
        return ""
    default_opinions = "Write your personal opinions here. They will be included in every video script."
    folder_id = "1tnTb4BjVjOARRKaQjmrse4kddddj9ogj"
    print(f"☁️  Syncing opinions from Drive folder: {folder_id}...")
    try:
        from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
        service.files().get(fileId=folder_id).execute()
        items = service.files().list(
            q=f"'{folder_id}' in parents and name='opinions.txt' and trashed=false",
            fields="files(id, name)",
        ).execute().get("files", [])

        if not items:
            print("   -> opinions.txt not found. Creating default template...")
            service.files().create(
                body={"name": "opinions.txt", "parents": [folder_id]},
                media_body=MediaIoBaseUpload(io.BytesIO(default_opinions.encode("utf-8")), mimetype="text/plain"),
                fields="id",
            ).execute()
            return ""
        fh = io.BytesIO()
        dl = MediaIoBaseDownload(fh, service.files().get_media(fileId=items[0]["id"]))
        done = False
        while not done:
            _, done = dl.next_chunk()
        content = fh.getvalue().decode("utf-8")
        if content.strip() and content.strip() != default_opinions:
            print("   ✅ Custom opinions loaded.")
            return content
        print("   ⚠️  Opinions file is empty or default.")
        return ""
    except Exception as e:
        print(f"Failed to sync opinions: {e}")
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# DRIVE: TODAY'S PROCESSED TITLES (dedup guard)
# ──────────────────────────────────────────────────────────────────────────────
def get_todays_processed_titles(service):
    if not service:
        return []
    pt_now = get_pacific_time()
    year   = pt_now.strftime("%Y")
    month  = pt_now.strftime("%m")
    date   = pt_now.strftime("%Y-%m-%d")
    root_id = "1tnTb4BjVjOARRKaQjmrse4kddddj9ogj"
    print(f"🔍 Checking Drive history for PT date: {date}")

    def find_folder(parent_id, name):
        try:
            q = (f"name='{name}' and mimeType='application/vnd.google-apps.folder'"
                 f" and '{parent_id}' in parents and trashed=false")
            files = service.files().list(q=q, fields="files(id,name)").execute().get("files", [])
            return files[0]["id"] if files else None
        except Exception as e:
            print(f"   ⚠️  find_folder '{name}': {e}")
            return None

    year_id  = find_folder(root_id, year)
    if not year_id:  return []
    month_id = find_folder(year_id, month)
    if not month_id: return []
    date_id  = find_folder(month_id, date)
    if not date_id:
        print(f"   ℹ️  No folder for {date} yet.")
        return []
    try:
        folders = service.files().list(
            q=f"'{date_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields="files(name)",
        ).execute().get("files", [])
        titles = []
        for f in folders:
            m = re.search(r"News-\d{4}-\d{2}-(.+)", f["name"])
            titles.append(m.group(1).replace("-", " ") if m else f["name"])
        if titles:
            print(f"   📈 {len(titles)} existing topics today: {', '.join(titles[:3])}...")
        return titles
    except Exception as e:
        print(f"   ⚠️  listing today topics: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# NEWS FETCHING + AI FILTERING
# ──────────────────────────────────────────────────────────────────────────────
def fetch_top_news(limit=30):
    ts      = datetime.now().strftime("%H:%M:%S")
    regions = ["US", "CA"]
    all_headlines: list[str] = []
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    for geo in regions:
        # Google Trends RSS
        try:
            req = urllib.request.Request(f"https://trends.google.com/trending/rss?geo={geo}", headers=ua)
            xml_data = urllib.request.urlopen(req, timeout=10).read().decode("utf-8")
            items = re.findall(r"<item>(.*?)</item>", xml_data, re.DOTALL)
            for item in items:
                title = re.search(r"<title>(.*?)</title>", item).group(1).replace("&amp;", "&")
                snippets = re.findall(r"<ht:news_item_snippet>(.*?)</ht:news_item_snippet>", item, re.DOTALL)
                clean = [re.sub(r"<[^>]+>", "", s).replace("&quot;", '"').replace("&#39;", "'")
                         for s in snippets]
                pic = re.search(r"<ht:picture>(.*?)</ht:picture>", item)
                line = f"({geo}) {title}: " + " / ".join(clean)
                if pic:
                    line += f" [Picture: {pic.group(1)}]"
                all_headlines.append(line)
            print(f"[{ts}] ✅ {geo} Trends: {len(items)} items")
        except Exception as e:
            print(f"[{ts}] ⚠️  {geo} Trends RSS failed: {e}")

        # Google News RSS
        try:
            req = urllib.request.Request(
                f"https://news.google.com/rss/search?q=trending+news+{geo}&hl=en-US&gl={geo}&ceid={geo}:en",
                headers=ua,
            )
            xml_data = urllib.request.urlopen(req, timeout=10).read().decode("utf-8")
            items = re.findall(r"<item>(.*?)</item>", xml_data, re.DOTALL)
            for item in items[:30]:
                title = re.search(r"<title>(.*?)</title>", item).group(1).replace("&amp;", "&")
                pic_url = ""
                desc_m = re.search(r"<description>(.*?)</description>", item)
                if desc_m:
                    img_m = re.search(r'img[^>]+src=["\'](.*?)["\']', desc_m.group(1))
                    if img_m:
                        pic_url = img_m.group(1)
                line = f"({geo}) {title}"
                if pic_url:
                    line += f" [Picture: {pic_url}]"
                all_headlines.append(line)
            print(f"[{ts}] ✅ {geo} News: {len(items[:30])} items")
        except Exception as e:
            print(f"[{ts}] ❌ {geo} News RSS failed: {e}")

    if not all_headlines:
        print(f"[{ts}] ❌ No headlines found.")
        _stat_error("fetch_top_news: no headlines from any region")
        return []

    print(f"[{ts}] 🧠 Grouping {len(all_headlines)} headlines into {limit} topics...")
    try:
        group_prompt = (
            f"Given these raw news headlines from US and CA, group similar stories and identify "
            f"the TOP {limit} most significant distinct trending topics globally. "
            f"For each topic provide Title, 2-sentence description, and if any headline had "
            f"[Picture: URL] include it in 'picture'. "
            f'Format strictly as JSON list: [{{"title":"...","description":"...","picture":"..."}},...]\n\n'
            f"Headlines:\n" + "\n".join(all_headlines[:50])
        )
        raw_json = generate_text(group_prompt).strip()
        if "```" in raw_json:
            raw_json = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_json, re.DOTALL).group(1)
        raw_json = re.sub(r",\s*([\]}])", r"\1", raw_json)
        topics = json.loads(raw_json)
        for t in topics:
            t["link"]    = "https://news.google.com"
            t["pubDate"] = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")
        print(f"[{ts}] ✅ Grouped into {len(topics)} topics.")
        _stats["topics_fetched"] = len(topics)
        return topics
    except Exception as e:
        print(f"[{ts}] ❌ Grouping failed: {e}")
        _stat_error(f"fetch_top_news grouping: {e}")
        fallback = [
            {"title": h.split(": ")[0], "description": h, "link": "", "pubDate": ""}
            for h in all_headlines[:limit]
        ]
        _stats["topics_fetched"] = len(fallback)
        return fallback


def filter_topics_with_ai(new_topics, existing_titles):
    if not new_topics:
        return []
    print(f"🧠 Filtering {len(new_topics)} candidates vs {len(existing_titles)} existing today...")
    final_topics = []
    category_counts: dict[str, int] = {}
    processed_context = "\n".join(f"- {t}" for t in existing_titles) if existing_titles else "None yet."

    for topic in new_topics:
        title = topic["title"]
        desc  = topic["description"]
        prompt = (
            f"You are a content curator. Topics already processed today:\n{processed_context}\n\n"
            f"NEW CANDIDATE — Title: {title}\nDescription: {desc}\n\n"
            f"1. Categorize this topic (Sports/Technology/Politics/Entertainment/Science/etc).\n"
            f"2. Is it similar/redundant to any existing topic? Be aggressive.\n"
            f'Respond ONLY as JSON: {{"category":"...","is_similar":true/false,"reason":"..."}}'
        )
        try:
            resp = generate_text(prompt).strip()
            if "```" in resp:
                resp = re.search(r"```(?:json)?\s*(.*?)\s*```", resp, re.DOTALL).group(1)
            res = json.loads(resp)
            category   = res.get("category", "General")
            is_similar = res.get("is_similar", False)

            if is_similar:
                print(f"   ⏩ Skip '{title}': similar ({res.get('reason')})")
                _stats["topics_skipped"] += 1
                continue
            if category_counts.get(category, 0) >= 3:
                print(f"   ⏩ Skip '{title}': category '{category}' limit (3)")
                _stats["topics_skipped"] += 1
                continue

            category_counts[category] = category_counts.get(category, 0) + 1
            final_topics.append(topic)
            processed_context += f"\n- {title}"
            print(f"   ✅ Approved '{title}' → {category} ({category_counts[category]}/3)")
        except Exception as e:
            print(f"   ⚠️  AI check failed for '{title}': {e}. Including as fallback.")
            final_topics.append(topic)

    _stats["topics_approved"] = len(final_topics)
    return final_topics


# ──────────────────────────────────────────────────────────────────────────────
# IMAGE GENERATION
# ──────────────────────────────────────────────────────────────────────────────
def generate_image(prompt, output_file, reference_image_url=None):
    url = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    if reference_image_url:
        try:
            img_req = urllib.request.Request(reference_image_url, headers={"User-Agent": "Mozilla/5.0"})
            image_data = urllib.request.urlopen(img_req).read()
            b64img = base64.b64encode(image_data).decode("utf-8")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Generate a cinematic 16:9 image in this style: {prompt}. Return as base64."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64img}"}},
                ],
            }]
        except Exception as e:
            print(f"   Reference image fetch failed ({e}), using text-only.")
            messages = [{"role": "user", "content": f"Generate a cinematic 16:9 image for: {prompt}. Return as base64."}]
    else:
        messages = [{"role": "user", "content": f"Generate a cinematic 16:9 image for: {prompt}. Return as base64."}]

    payload = json.dumps({"model": "gemini-3.1-flash-image-16-9", "messages": messages}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    content = result["choices"][0]["message"]["content"]

    b64_matches = re.findall(r"[A-Za-z0-9+/]{100,}", content) or \
                  re.findall(r"base64,([A-Za-z0-9+/=]+)", content)
    if b64_matches:
        b64 = max(b64_matches, key=len)
        pad = len(b64) % 4
        if pad: b64 += "=" * (4 - pad)
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        out = re.sub(r"\.[^.]+$", ".png", output_file)
        img.save(out, "PNG")
        print(f"✅ Image saved → {out}")
        return True

    url_match = re.search(r"https?://[^\s\"']+", content)
    if url_match:
        img_req = urllib.request.Request(url_match.group(0), headers={"User-Agent": "Mozilla/5.0"})
        img = Image.open(io.BytesIO(urllib.request.urlopen(img_req).read()))
        out = re.sub(r"\.[^.]+$", ".png", output_file)
        img.save(out, "PNG")
        return True

    raise RuntimeError("No image data found in response")


def generate_image_with_retry(prompt, output_file, reference_image_url=None, retries=MAX_RETRIES):
    for attempt in range(1, retries + 1):
        try:
            result = generate_image(prompt, output_file, reference_image_url)
            _stats["images_ok"] += 1
            return result
        except Exception as e:
            print(f"   [Attempt {attempt}/{retries}] Image failed: {e}")
            if attempt < retries:
                time.sleep(2)
    print(f"   ⚠️  Image skipped after {retries} attempts.")
    _stats["images_failed"] += 1
    return None


# ──────────────────────────────────────────────────────────────────────────────
# AUDIO GENERATION
# ──────────────────────────────────────────────────────────────────────────────
def generate_audio(text, output_file, voice="alloy"):
    url = f"{API_BASE_URL}/audio/speech"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = json.dumps({"model": "tts-1", "input": text, "voice": voice}).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req) as resp:
            with open(output_file, "wb") as f:
                f.write(resp.read())
        print(f"✅ Audio saved → {output_file}")
        _stats["audio_ok"] += 1
        return True
    except Exception as e:
        print(f"   Audio API failed ({e}). Trying edge-tts fallback...")

    voice_map = {
        "alloy": "en-US-AriaNeural", "echo": "en-US-GuyNeural",
        "fable": "en-GB-SoniaNeural", "onyx": "en-US-ChristopherNeural",
        "nova": "en-US-NatashaNeural", "shimmer": "en-US-JennyNeural",
    }
    edge_voice = voice_map.get(voice, "en-US-AriaNeural")

    try:
        import edge_tts
    except ImportError:
        os.system("pip install edge-tts -q")
        import edge_tts

    import asyncio

    async def _run():
        clean = re.sub(r"[*_`#]", "", text.strip())
        if not clean: return False
        for attempt in range(3):
            try:
                await edge_tts.Communicate(clean, edge_voice).save(output_file)
                if os.path.exists(output_file) and os.path.getsize(output_file) > 1000:
                    return True
            except Exception as exc:
                print(f"   edge-tts attempt {attempt+1}: {exc}")
            await asyncio.sleep(2)
        return False

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as ex:
            ok = ex.submit(lambda: asyncio.run(_run())).result()
    else:
        ok = loop.run_until_complete(_run())

    if ok:
        _stats["audio_ok"] += 1
    return ok


# ──────────────────────────────────────────────────────────────────────────────
# BACKGROUND MUSIC + COMBINE
# ──────────────────────────────────────────────────────────────────────────────
def download_bg_music(style, output_file):
    try:
        import yt_dlp
    except ImportError:
        os.system("pip install yt-dlp -q")
        import yt_dlp

    print(f"🎵 Downloading BG music for: {style}")
    cookie_file     = "cookies.txt"
    browser_session = os.path.expanduser("~/.browser-session")
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        "outtmpl": output_file.rsplit(".", 1)[0],
        "quiet": True, "noplaylist": True, "nocheckcertificate": True,
        "ignoreerrors": True, "logtostderr": False,
    }
    if os.path.exists(cookie_file):
        ydl_opts["cookiefile"] = cookie_file
    elif os.path.exists(browser_session):
        ydl_opts["cookiesfrombrowser"] = ("chrome", browser_session)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(f"ytsearch1:{style} instrumental background music no copyright", download=True)
        return True
    except Exception as e:
        print(f"   BG music failed: {e}")
        return False


def combine_audio(voice_file, bg_file, output_file):
    cmd = [
        "ffmpeg", "-y", "-i", voice_file, "-i", bg_file,
        "-filter_complex", "[1:a]volume=0.08[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2",
        output_file,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Combined audio → {output_file}")
        return True
    except Exception as e:
        print(f"   Audio combine failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# VISUAL STYLE PICKER
# ──────────────────────────────────────────────────────────────────────────────
def clean_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()


def pick_visual_style(title):
    global _artistic_style_index
    if random.random() < ART_PERCENTAGE:
        style = POPULAR_ARTISTIC_STYLES[_artistic_style_index % len(POPULAR_ARTISTIC_STYLES)]
        _artistic_style_index += 1
        print(f"   🎨 Artistic → {style}")
        return True, style, "iconic ARTISTIC visual style"
    categories = ("National Geographic, Cinematic 4k Film Still, Magnum Photography, "
                  "Drone Perspective, Kodak Portra 400, Macro Tech Photography")
    q = (f"Pick the best visual style from: {categories} for the news story: '{title}'. "
         f"Return ONLY the style phrase (under 10 words).")
    style = generate_text(q).strip().strip('"\'- \n') or "Cinematic National Geographic style"
    print(f"   📷 Photographic → {style}")
    return False, style, "professional PHOTOGRAPHIC visual style"


# ──────────────────────────────────────────────────────────────────────────────
# RUN STATS: SAVE + MONTHLY REPORT
# ──────────────────────────────────────────────────────────────────────────────
def _drive_upload_text(service, folder_id: str, filename: str, content: str, mimetype: str):
    """Create or overwrite a text file on Drive. Returns file ID."""
    from googleapiclient.http import MediaIoBaseUpload
    media = MediaIoBaseUpload(io.BytesIO(content.encode("utf-8")), mimetype=mimetype)
    q = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    existing = service.files().list(q=q, fields="files(id)").execute().get("files", [])
    if existing:
        return service.files().update(fileId=existing[0]["id"], media_body=media).execute()["id"]
    return service.files().create(
        body={"name": filename, "parents": [folder_id]},
        media_body=media, fields="id",
    ).execute()["id"]


def _drive_append_jsonl(service, folder_id: str, filename: str, record: dict):
    """Append one JSON line to a JSONL file on Drive (download, append, re-upload)."""
    from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
    new_line = json.dumps(record, ensure_ascii=False) + "\n"
    q = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    files = service.files().list(q=q, fields="files(id)").execute().get("files", [])
    if files:
        fh = io.BytesIO()
        dl = MediaIoBaseDownload(fh, service.files().get_media(fileId=files[0]["id"]))
        done = False
        while not done:
            _, done = dl.next_chunk()
        updated = fh.getvalue().decode("utf-8") + new_line
        service.files().update(
            fileId=files[0]["id"],
            media_body=MediaIoBaseUpload(io.BytesIO(updated.encode("utf-8")), mimetype="application/x-ndjson"),
        ).execute()
    else:
        service.files().create(
            body={"name": filename, "parents": [folder_id]},
            media_body=MediaIoBaseUpload(io.BytesIO(new_line.encode("utf-8")), mimetype="application/x-ndjson"),
            fields="id",
        ).execute()


def save_run_stats(service, base_dir: str, news_items: list):
    """Append this run's stats to the monthly JSONL log on Drive and write locally."""
    pt_now    = get_pacific_time()
    year_str  = pt_now.strftime("%Y")
    month_str = pt_now.strftime("%m")

    source_hits = _stats["llm_source_hits"]
    primary_src = max(source_hits, key=source_hits.get) if source_hits else "none"

    record = {
        "run_ts":           pt_now.strftime("%Y-%m-%d %H:%M"),
        "llm_primary":      primary_src,
        "llm_source_hits":  source_hits,
        "llm_total_calls":  _stats["llm_calls"],
        "topics_fetched":   _stats["topics_fetched"],
        "topics_approved":  _stats["topics_approved"],
        "topics_skipped":   _stats["topics_skipped"],
        "images_ok":        _stats["images_ok"],
        "images_failed":    _stats["images_failed"],
        "audio_ok":         _stats["audio_ok"],
        "errors":           _stats["errors"],
        "titles":           [item["title"] for item in news_items],
    }

    # Local stats file
    stats_path = os.path.join(base_dir, "run_stats.json")
    try:
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        print(f"📊 Run stats → {stats_path}")
    except Exception as e:
        print(f"   ⚠️  Local stats write failed: {e}")

    if not service:
        return

    root_id = "1tnTb4BjVjOARRKaQjmrse4kddddj9ogj"
    log_filename = f"pipeline_stats_{year_str}_{month_str}.jsonl"
    try:
        _drive_append_jsonl(service, root_id, log_filename, record)
        print(f"☁️  Stats appended → {log_filename}")
    except Exception as e:
        print(f"   ⚠️  Drive stats upload failed: {e}")


def generate_monthly_report(service):
    """
    Aggregate the previous month's JSONL stats and upload a Markdown report to Drive.
    Triggered when GENERATE_MONTHLY_REPORT=true OR today is the 1st of the month.
    """
    if not service:
        print("⚠️  No Drive service — skipping monthly report.")
        return

    pt_now = get_pacific_time()
    # Report covers the previous calendar month
    first_this = pt_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_month  = first_this - timedelta(days=1)
    year_str   = last_month.strftime("%Y")
    month_str  = last_month.strftime("%m")
    month_name = last_month.strftime("%B %Y")

    log_filename    = f"pipeline_stats_{year_str}_{month_str}.jsonl"
    report_filename = f"monthly_report_{year_str}_{month_str}.md"
    root_id = "1tnTb4BjVjOARRKaQjmrse4kddddj9ogj"

    print(f"\n📅 Generating monthly report for {month_name}...")

    try:
        from googleapiclient.http import MediaIoBaseDownload
        q = f"name='{log_filename}' and '{root_id}' in parents and trashed=false"
        files = service.files().list(q=q, fields="files(id)").execute().get("files", [])
        if not files:
            print(f"   ℹ️  No stats log found for {month_name}.")
            return
        fh = io.BytesIO()
        dl = MediaIoBaseDownload(fh, service.files().get_media(fileId=files[0]["id"]))
        done = False
        while not done:
            _, done = dl.next_chunk()
        records = [json.loads(line) for line in fh.getvalue().decode("utf-8").splitlines() if line.strip()]
    except Exception as e:
        print(f"   ❌ Failed to load stats log: {e}")
        return

    if not records:
        print("   ℹ️  Stats log is empty.")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────────
    total_runs     = len(records)
    total_fetched  = sum(r.get("topics_fetched",  0) for r in records)
    total_approved = sum(r.get("topics_approved", 0) for r in records)
    total_skipped  = sum(r.get("topics_skipped",  0) for r in records)
    total_images   = sum(r.get("images_ok",       0) for r in records)
    total_img_fail = sum(r.get("images_failed",   0) for r in records)
    total_audio    = sum(r.get("audio_ok",        0) for r in records)
    total_errors   = sum(len(r.get("errors", []))    for r in records)
    total_llm      = sum(r.get("llm_total_calls", 0) for r in records)

    llm_totals: dict[str, int] = {}
    for r in records:
        for src, cnt in r.get("llm_source_hits", {}).items():
            llm_totals[src] = llm_totals.get(src, 0) + cnt
    llm_rows = sorted(llm_totals.items(), key=lambda x: -x[1])

    all_titles: list[str] = []
    for r in records:
        all_titles.extend(r.get("titles", []))
    top_titles = Counter(all_titles).most_common(20)

    # Daily breakdown
    by_date: dict[str, list] = defaultdict(list)
    for r in records:
        by_date[r.get("run_ts", "")[:10]].append(r)

    # ── Build Markdown ────────────────────────────────────────────────────────
    lines = [
        f"# Pipeline Monthly Report — {month_name}",
        "",
        f"*Generated {pt_now.strftime('%Y-%m-%d %H:%M')} PT  |  Source: `{log_filename}`*",
        "",
        "---",
        "",
        "## Run Summary",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Pipeline runs | {total_runs} |",
        f"| LLM calls total | {total_llm} |",
        f"| Topics fetched (raw) | {total_fetched} |",
        f"| Topics approved | {total_approved} |",
        f"| Topics skipped (duplicate / category limit) | {total_skipped} |",
        f"| Images generated | {total_images} |",
        f"| Image failures | {total_img_fail} |",
        f"| Audio files generated | {total_audio} |",
        f"| Errors logged | {total_errors} |",
        "",
        "---",
        "",
        "## LLM Source Usage",
        "",
        "| Source | Calls | Share |",
        "|--------|------:|------:|",
    ]
    for src, cnt in llm_rows:
        share = f"{cnt / total_llm * 100:.1f}%" if total_llm else "—"
        lines.append(f"| {src} | {cnt} | {share} |")

    lines += [
        "",
        "---",
        "",
        "## Top 20 Topics This Month",
        "",
    ]
    for i, (title, count) in enumerate(top_titles, 1):
        freq = f"  *(×{count})*" if count > 1 else ""
        lines.append(f"{i}. {title}{freq}")

    lines += [
        "",
        "---",
        "",
        "## Daily Breakdown",
        "",
        "| Date | Runs | Approved | LLM Calls | Errors |",
        "|------|-----:|---------:|----------:|-------:|",
    ]
    for date_key in sorted(by_date.keys()):
        recs = by_date[date_key]
        lines.append(
            f"| {date_key} "
            f"| {len(recs)} "
            f"| {sum(r.get('topics_approved', 0) for r in recs)} "
            f"| {sum(r.get('llm_total_calls', 0) for r in recs)} "
            f"| {sum(len(r.get('errors', [])) for r in recs)} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Error Log",
        "",
    ]
    error_entries = [
        f"- `{r['run_ts']}` — {err}"
        for r in records
        for err in r.get("errors", [])
    ]
    lines += error_entries if error_entries else ["*No errors recorded this month.*"]

    report_md = "\n".join(lines)

    # ── Upload to Drive ───────────────────────────────────────────────────────
    try:
        _drive_upload_text(service, root_id, report_filename, report_md, "text/markdown")
        print(f"   ✅ Monthly report uploaded → {report_filename}")
        print(f"\n{'='*62}")
        print(f"  MONTHLY REPORT — {month_name}")
        print(f"{'='*62}")
        print(f"  Runs: {total_runs}  Approved: {total_approved}  LLM calls: {total_llm}")
        for src, cnt in llm_rows:
            share = f"{cnt / total_llm * 100:.1f}%" if total_llm else "—"
            print(f"    {src:<20} {cnt:>4} calls  ({share})")
        print(f"  Errors: {total_errors}")
        print(f"{'='*62}\n")
    except Exception as e:
        print(f"   ❌ Report upload failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    service = get_drive_service()
    user_opinions   = sync_opinions_from_drive(service) if service else ""
    existing_titles = get_todays_processed_titles(service) if service else []

    if not service:
        print("⚠️  No Drive service — skipping history check and opinions sync.")

    pt_now = get_pacific_time()

    report_only = "--report-only" in sys.argv or os.environ.get("ONLY_GENERATE_REPORT", "").lower() == "true"
    # Monthly report: on 1st of month OR GENERATE_MONTHLY_REPORT=true OR report_only
    if pt_now.day == 1 or os.environ.get("GENERATE_MONTHLY_REPORT", "").lower() == "true" or report_only:
        generate_monthly_report(service)
        if report_only:
            print("✅ Report generation complete. Exiting due to report-only option.")
            return

    year_str  = pt_now.strftime("%Y")
    month_str = pt_now.strftime("%m")
    date_str  = pt_now.strftime("%Y-%m-%d")

    base_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "news")
    )
    os.makedirs(base_dir, exist_ok=True)

    raw_news   = fetch_top_news(limit=TOPIC_LIMIT)
    news_items = filter_topics_with_ai(raw_news, existing_titles)

    if not news_items:
        print("📭 No new topics after filtering.")
        save_run_stats(service, base_dir, [])
        return

    for idx, item in enumerate(news_items):
        title = item["title"]
        print(f"\n{'='*60}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [{idx+1}/{len(news_items)}] {title}")
        print(f"{'='*60}")

        clean_title    = clean_filename(title)[:50]
        start_hhmm     = pt_now.strftime("%H%M")
        year_month_dir = os.path.join(base_dir, year_str, month_str, date_str)
        os.makedirs(year_month_dir, exist_ok=True)

        project_dir = os.path.join(year_month_dir, f"{start_hhmm}-{clean_title}")
        os.makedirs(project_dir, exist_ok=True)

        opinions_block = ""
        if user_opinions:
            opinions_block = (
                f"\nUSER OPINIONS TO INCLUDE:\n{user_opinions}\n\n"
                f"CRITICAL: Weave these into dialogue naturally. At least one character must "
                f"embody this perspective.\n"
            )

        # ── Script ────────────────────────────────────────────────────────────
        script_prompt = f"""Write a viral video script about:
Title: {title}
Details: {item['description']}
{opinions_block}
Include: strong hook, main story, call to action. Invent realistic reactions from relevant roles.
The script MUST be AT LEAST 3 FULL MINUTES (~450-600 words).
Use multiple roles (Studio Anchor, Expert, Bystander) each with a different voice: alloy, echo, fable, onyx, nova, shimmer.

CRITICAL FORMAT — interleave 8-second timestamps, [Video Prompt] tags, [Voice: X] dialogue:

Song Title: {title}
Style: Interleaved (Batch Processed)

> 00:00-00:08 [Video Prompt] Dynamic slow-motion tracking shot...
[Voice: onyx] Breaking news tonight...

> 00:08-00:16 [Video Prompt] Fast zoom onto reporter...
[Voice: nova] Experts now say...

Continue in 8-second chunks for the full 3 minutes."""

        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✍️  Generating script...")
        script_content = generate_text(script_prompt)
        with open(os.path.join(project_dir, "lyrics_with_prompts.md"), "w", encoding="utf-8") as f:
            prefix = "" if "Song Title:" in script_content else f"Song Title: {title}\nStyle: Interleaved (Batch Processed)\n\n"
            f.write(prefix + script_content + "\n")

        # ── Cover image ───────────────────────────────────────────────────────
        is_artistic, image_style, mode_desc = pick_visual_style(title)
        anthro_mod = ""
        if random.random() < ANTHROPOMORPHIC_PERCENTAGE:
            anthro_types = [
                "anthropomorphic animals", "cute anthropomorphic cats and dogs",
                "anthropomorphic animals in business attire", "whimsical anthropomorphic wildlife",
            ]
            anthro_mod = f" KEY DIRECTIVE: Re-imagine all human subjects as {random.choice(anthro_types)}."

        image_prompt = (
            f"Compelling cover thumbnail (no text) for: '{title}'. "
            f"Style: {image_style}. Details: {item['description']}.{anthro_mod}"
        )
        pic_url = item.get("picture", "") or None
        tag = " [Anthro]" if anthro_mod else ""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🖼️  Generating cover — {image_style}{tag}")
        result = generate_image_with_retry(image_prompt, os.path.join(project_dir, "cover.png"), pic_url)
        if result is None:
            print("   ⚠️  Skipping story (image failed).")
            continue

        # Character description
        char_content = generate_text(
            f"Describe the main characters, subjects, and environment for the cover image of: '{title}'. "
            f"Details: {item['description']}"
        ).strip()
        with open(os.path.join(project_dir, "charactor.md"), "w", encoding="utf-8") as f:
            f.write(f"# Cover Image & Environment Description\n\n{char_content}\n")

        # ── Reference dir ─────────────────────────────────────────────────────
        ref_dir = os.path.join(project_dir, "references")
        os.makedirs(ref_dir, exist_ok=True)

        # Intro script + audio
        intro_prompt = f"""Write an 8-second intro for: '{title}'.
Format:
> 00:00-00:05 [Video Prompt] Dynamic hook from cover image.
[Voice: onyx] (5-second punchy hook)

> 00:05-00:08 [Video Prompt] Intro/logo transition.
[Voice: alloy] (3-second wrap-up)

Details: {item['description']}"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✍️  Generating intro...")
        intro_content = generate_text(intro_prompt)
        with open(os.path.join(project_dir, "intro_script.md"), "w", encoding="utf-8") as f:
            f.write(intro_content)

        intro_blocks = re.findall(r"\[Voice:\s*(\w+)\]\s*(.*?)(?=\n>|\Z)", intro_content, re.DOTALL | re.IGNORECASE)
        intro_voice_file = os.path.join(project_dir, "intro_audio.mp3")
        if not intro_blocks:
            clean_intro = re.sub(r"\[.*?\]|>.*?$", "", intro_content, flags=re.MULTILINE).strip()
            generate_audio(clean_intro, intro_voice_file, voice="onyx")
        else:
            chunk_files = []
            for i, (v, t) in enumerate(intro_blocks):
                txt = t.strip().replace("**", "").replace("*", "")
                if not txt: continue
                v = v.lower() if v.lower() in ["alloy","echo","fable","onyx","nova","shimmer"] else "onyx"
                cf = os.path.join(ref_dir, f"intro_chunk_{i:03d}.mp3")
                if generate_audio(txt[:500], cf, voice=v):
                    chunk_files.append(cf)
            if chunk_files:
                concat_txt = os.path.join(ref_dir, "intro_concat.txt")
                with open(concat_txt, "w") as f:
                    f.writelines(f"file '{os.path.basename(x)}'\n" for x in chunk_files)
                try:
                    subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",concat_txt,"-c","copy",intro_voice_file],
                                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    import shutil; shutil.copy(chunk_files[0], intro_voice_file)
                for f in chunk_files:
                    try: os.remove(f)
                    except: pass
                try: os.remove(concat_txt)
                except: pass

        # Reference video
        try:
            import yt_dlp
            ydl_opts = {
                "format": "best",
                "outtmpl": os.path.join(ref_dir, "ref_video_%(id)s.%(ext)s"),
                "quiet": True, "noplaylist": True, "max_downloads": 1,
            }
            cookie_file = "cookies.txt"
            if os.path.exists(cookie_file):
                ydl_opts["cookiefile"] = cookie_file
            elif os.path.exists(os.path.expanduser("~/.browser-session")):
                ydl_opts["cookiesfrombrowser"] = ("chrome", os.path.expanduser("~/.browser-session"))
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(f"ytsearch1:{title} news footage", download=True)
        except Exception as e:
            print(f"   ⚠️  Reference video skipped: {e}")

        # ── Full voice audio ──────────────────────────────────────────────────
        voice_file  = os.path.join(ref_dir, "full_voice.mp3")
        bg_file     = os.path.join(ref_dir, "bg_music.mp3")
        final_audio = os.path.join(project_dir, "combined_audio.mp3")

        blocks = re.findall(r"\[Voice:\s*(\w+)\]\s*(.*?)(?=\n>|\Z)", script_content, re.DOTALL | re.IGNORECASE)
        if not blocks:
            clean_txt = re.sub(r"\[.*?\]|>.*?$", "", script_content, flags=re.MULTILINE)
            clean_txt = (clean_txt.replace(f"Song Title: {title}", "")
                         .replace("Style: Interleaved (Batch Processed)", "").strip())
            generate_audio(clean_txt[:4000] or title, voice_file, voice="alloy")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎙️  Generating {len(blocks)} audio chunks...")
            chunk_files = []
            for i, (v, t) in enumerate(blocks):
                txt = t.strip().replace("**","").replace("*","")
                if not txt: continue
                v = v.lower() if v.lower() in ["alloy","echo","fable","onyx","nova","shimmer"] else "alloy"
                cf = os.path.join(ref_dir, f"chunk_{i:03d}.mp3")
                if generate_audio(txt[:4000], cf, voice=v):
                    chunk_files.append(cf)
            if chunk_files:
                concat_txt = os.path.join(ref_dir, "concat.txt")
                with open(concat_txt, "w") as f:
                    f.writelines(f"file '{os.path.basename(x)}'\n" for x in chunk_files)
                try:
                    subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",concat_txt,"-c","copy",voice_file],
                                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    import shutil; shutil.copy(chunk_files[0], voice_file)
                for f in chunk_files:
                    try: os.remove(f)
                    except: pass
                try: os.remove(concat_txt)
                except: pass

        bg_style = generate_text(
            f"Give a 2-word genre for background news music about: '{title}' (e.g. 'tense cinematic'). "
            f"Return ONLY the genre."
        ).strip().strip('"\'- \n')
        if download_bg_music(bg_style, bg_file):
            if not combine_audio(voice_file, bg_file, final_audio):
                import shutil; shutil.copy(voice_file, final_audio)
        else:
            import shutil; shutil.copy(voice_file, final_audio)

        with open(os.path.join(ref_dir, "references.txt"), "w", encoding="utf-8") as f:
            f.write(f"Title: {title}\nURL: {item.get('link','')}\nDate: {item.get('pubDate','')}")

        # Upload to Drive
        upload_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "upload_to_drive.py"))
        if os.path.exists(upload_script):
            print("🚀 Uploading to Drive...")
            subprocess.run([sys.executable, upload_script, project_dir])
        else:
            print("⚠️  upload_to_drive.py not found — skipping Drive upload.")

        time.sleep(2)

    # ── End-of-run stats ──────────────────────────────────────────────────────
    save_run_stats(service, base_dir, news_items)
    print(f"\n✅ Pipeline complete. Projects saved in: {base_dir}")
    print(f"   LLM usage this run: {_stats['llm_source_hits']}")


if __name__ == "__main__":
    main()