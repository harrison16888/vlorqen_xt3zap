"""
check_cf_accounts.py
====================
Unified Cloudflare AI health checker + model discovery.

Usage:
    python check_cf_accounts.py              # ping all accounts
    python check_cf_accounts.py --models     # also list @cf/* models
    python check_cf_accounts.py --models --json   # machine-readable output
    python check_cf_accounts.py --best       # print the first healthy account ID
                                             #   (useful as a one-liner in shell scripts)
"""

import os
import sys
import json
import urllib.request
from urllib.error import HTTPError
import argparse

# ── ANSI colours (auto-disabled when not a TTY) ───────────────────────────────
_tty = sys.stdout.isatty()
G  = '\033[92m' if _tty else ''   # green
R  = '\033[91m' if _tty else ''   # red
Y  = '\033[93m' if _tty else ''   # yellow
B  = '\033[94m' if _tty else ''   # blue
DIM= '\033[2m'  if _tty else ''
RST= '\033[0m'  if _tty else ''
BOLD='\033[1m'  if _tty else ''

PING_MODEL = "@cf/meta/llama-3-8b-instruct"

# ──────────────────────────────────────────────────────────────────────────────
# ENV LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_env(path: str | None = None):
    """Load a .env file from the given path (or default location)."""
    candidates = [path] if path else [
        os.path.join(os.path.dirname(__file__), '.env'),
        os.path.join(os.getcwd(), '.env'),
    ]
    for env_path in candidates:
        if env_path and os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    key, val = line.split('=', 1)
                    os.environ.setdefault(key.strip(), val.strip())
            return env_path
    return None


# ──────────────────────────────────────────────────────────────────────────────
# ACCOUNT DISCOVERY  (mirrors the three sources in the original script)
# ──────────────────────────────────────────────────────────────────────────────
def discover_accounts() -> list[tuple[str, str, str]]:
    """
    Return a deduplicated list of (source_label, account_id, api_token).

    Sources checked (in order):
      1. CLOUDFLARE_ACCOUNTS_JSON  — JSON array [{id, token}, ...]
      2. CLOUDFLARE_ACCOUNT_ID_N / CLOUDFLARE_API_TOKEN_N  — numbered pairs
      3. CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN  — legacy single pair
      4. CF_ACCOUNT_ID / CF_AI_TOKEN  — pipeline.py convention
    """
    raw: list[tuple[str, str, str]] = []

    # Source 1 — JSON array
    json_str = os.environ.get('CLOUDFLARE_ACCOUNTS_JSON', '')
    if json_str:
        try:
            for acc in json.loads(json_str):
                if acc.get('id') and acc.get('token'):
                    raw.append(("JSON Array", acc['id'], acc['token']))
        except Exception as e:
            print(f"{Y}⚠  CLOUDFLARE_ACCOUNTS_JSON parse error: {e}{RST}")

    # Source 2 — numbered pairs  (CLOUDFLARE_ACCOUNT_ID_1 / CLOUDFLARE_API_TOKEN_1, etc.)
    for key in sorted(os.environ.keys()):
        if key.startswith('CLOUDFLARE_ACCOUNT_ID_'):
            idx   = key[len('CLOUDFLARE_ACCOUNT_ID_'):]
            token = os.environ.get(f'CLOUDFLARE_API_TOKEN_{idx}', '')
            acc_id = os.environ.get(key, '')
            if acc_id and token:
                raw.append((f"ENV _{idx}", acc_id, token))

    # Source 3 — legacy single pair
    legacy_id  = os.environ.get('CLOUDFLARE_ACCOUNT_ID', '')
    legacy_tok = os.environ.get('CLOUDFLARE_API_TOKEN', '')
    if legacy_id and legacy_tok:
        raw.append(("ENV Legacy", legacy_id, legacy_tok))

    # Source 4 — pipeline.py convention
    cf_id  = os.environ.get('CF_ACCOUNT_ID', '')
    cf_tok = os.environ.get('CF_AI_TOKEN', '')
    if cf_id and cf_tok:
        raw.append(("pipeline.py", cf_id, cf_tok))

    # Deduplicate by account_id (first occurrence wins for label)
    seen: dict[str, tuple[str, str, str]] = {}
    for src, acc_id, token in raw:
        if acc_id not in seen:
            seen[acc_id] = (src, acc_id, token)

    return list(seen.values())


# ──────────────────────────────────────────────────────────────────────────────
# HEALTH PING
# ──────────────────────────────────────────────────────────────────────────────
def ping_account(account_id: str, api_token: str) -> dict:
    """
    Send a minimal 1-token prompt to the ping model.
    Returns {"status": "healthy"|"limit"|"error", "code": int|None, "detail": str}
    """
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/ai/run/{PING_MODEL}"
    )
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    payload = json.dumps({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            if body.get("success"):
                return {"status": "healthy", "code": 200, "detail": "OK"}
            errors = body.get("errors", [])
            return {"status": "error", "code": 200, "detail": str(errors)}
    except HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = {}
        detail = str(body.get("errors", e.reason))
        if e.code == 429 or "daily free allocation" in detail.lower() or "neuron" in detail.lower():
            return {"status": "limit", "code": 429, "detail": "Neuron/daily limit reached"}
        return {"status": "error", "code": e.code, "detail": detail}
    except Exception as ex:
        return {"status": "error", "code": None, "detail": str(ex)}


def _ping_badge(result: dict) -> str:
    s = result["status"]
    if s == "healthy": return f"{G}✓ Healthy{RST}"
    if s == "limit":   return f"{R}✗ Neuron limit (429){RST}"
    code = f" ({result['code']})" if result.get("code") else ""
    return f"{Y}⚠ Error{code}{RST}"


# ──────────────────────────────────────────────────────────────────────────────
# MODEL DISCOVERY
# ──────────────────────────────────────────────────────────────────────────────
def list_cf_models(account_id: str, api_token: str, prefix: str = "@cf/") -> list[dict]:
    """
    Fetch all @cf/* models for the given account.
    Mirrors the JS checkModels() function (per_page=200, filters by @cf/ prefix).
    """
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/ai/models/search?per_page=200"
    )
    headers = {"Authorization": f"Bearer {api_token}"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = data.get("result", [])
        filtered = [
            m for m in models
            if (m.get("name") or m.get("id") or "").startswith(prefix)
        ]
        filtered.sort(key=lambda m: m.get("name") or m.get("id") or "")
        return filtered
    except Exception as e:
        print(f"{Y}⚠  Model list failed for {account_id[:12]}…: {e}{RST}")
        return []


def _print_models_table(models: list[dict]):
    """Compact table: id | task | description snippet"""
    if not models:
        print(f"  {DIM}(no models returned){RST}")
        return
    col_id   = max(len(m.get("name") or m.get("id") or "") for m in models) + 2
    col_task = 28
    col_id   = min(col_id, 52)
    header   = f"  {'Model':<{col_id}}  {'Task':<{col_task}}  Description"
    print(f"{DIM}{header}{RST}")
    print(f"  {'-'*(col_id+col_task+40)}")
    for m in models:
        name = (m.get("name") or m.get("id") or "")[:col_id]
        task = (m.get("task", {}).get("name") if m.get("task") else "N/A")[:col_task]
        desc = (m.get("description") or "")[:60]
        print(f"  {name:<{col_id}}  {task:<{col_task}}  {DIM}{desc}{RST}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Cloudflare AI account health checker + model discovery")
    parser.add_argument("--models",   action="store_true", help="List @cf/* models for healthy accounts")
    parser.add_argument("--json",     action="store_true", help="Output results as JSON (implies --models)")
    parser.add_argument("--best",     action="store_true", help="Print only the first healthy account ID (for scripting)")
    parser.add_argument("--env",      default=None,        help="Path to .env file (optional)")
    parser.add_argument("--prefix",   default="@cf/",      help="Model name prefix filter (default: @cf/)")
    args = parser.parse_args()

    env_file = load_env(args.env)
    if env_file:
        print(f"{DIM}Loaded env: {env_file}{RST}")

    accounts = discover_accounts()
    if not accounts:
        print(f"{R}No Cloudflare accounts found in environment.{RST}")
        print("Set one of: CLOUDFLARE_ACCOUNTS_JSON, CLOUDFLARE_ACCOUNT_ID_N/CLOUDFLARE_API_TOKEN_N,")
        print("            CLOUDFLARE_ACCOUNT_ID/CLOUDFLARE_API_TOKEN, or CF_ACCOUNT_ID/CF_AI_TOKEN")
        sys.exit(1)

    # --best: silent ping, print only the first healthy account ID, exit
    if args.best:
        for _, acc_id, token in accounts:
            if ping_account(acc_id, token)["status"] == "healthy":
                print(acc_id, end="")
                sys.exit(0)
        sys.exit(1)  # no healthy account — empty output, non-zero exit

    # Full table output
    results = []

    print(f"\n{BOLD}Cloudflare AI Account Health Check{RST}")
    print("─" * 72)
    print(f"  {'Source':<14} {'Account ID':<36} {'Status'}")
    print("─" * 72)

    for src, acc_id, token in accounts:
        ping = ping_account(acc_id, token)
        badge = _ping_badge(ping)
        print(f"  {src:<14} {acc_id:<36} {badge}")
        if ping["status"] != "healthy" and not args.json:
            print(f"  {DIM}  └─ {ping['detail']}{RST}")

        entry = {
            "source":     src,
            "account_id": acc_id,
            "status":     ping["status"],
            "http_code":  ping["code"],
            "detail":     ping["detail"],
            "models":     [],
        }

        if (args.models or args.json) and ping["status"] == "healthy":
            models = list_cf_models(acc_id, token, prefix=args.prefix)
            entry["models"] = models
            if not args.json:
                print(f"\n  {B}@cf/* models for {acc_id[:20]}…{RST}  ({len(models)} found)")
                _print_models_table(models)
                print()

        results.append(entry)

    if not args.json:
        print("─" * 72)
        healthy = sum(1 for r in results if r["status"] == "healthy")
        limited = sum(1 for r in results if r["status"] == "limit")
        errored = sum(1 for r in results if r["status"] == "error")
        print(f"\n  Total: {len(results)}  |  {G}{healthy} healthy{RST}  |  {R}{limited} at limit{RST}  |  {Y}{errored} error{RST}\n")
    else:
        # Strip ANSI from JSON output
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()