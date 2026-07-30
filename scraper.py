#!/usr/bin/env python3
"""
BSDI Partnership & Announcement Scraper
-----------------------------------------
For each member company, asks Claude (with web search enabled) to find recent
news about MoUs, Letters of Intent, contracts, or partnership announcements
with international partners, extracts structured data, and upserts it into
Supabase.

Intended to run daily via GitHub Actions (see .github/workflows/scrape.yml).

Required environment variables:
  ANTHROPIC_API_KEY   - Anthropic API key
  SUPABASE_URL        - e.g. https://xxxx.supabase.co
  SUPABASE_KEY        - service role key (kept secret, only used server-side)
"""

import os
import sys
import json
import time
import datetime
import urllib.request
import urllib.error

import anthropic

ANTHROPIC_MODEL = "claude-sonnet-4-6"
_lookback_raw = os.environ.get("LOOKBACK_DAYS", "").strip()
LOOKBACK_DAYS = int(_lookback_raw) if _lookback_raw else 9  # default covers weekly runs with a couple days' overlap
MEMBERS_FILE = os.path.join(os.path.dirname(__file__), "members.json")

RESEARCH_SYSTEM_PROMPT = """You are a research assistant that finds recent, real news about a specific \
defence/security company signing agreements with international partners \
(Memoranda of Understanding, Letters of Intent, contracts, or partnership announcements).

Only report items that are:
- Genuinely about the named company (not a similarly named unrelated company)
- Dated within the requested lookback window
- Backed by a real, checkable news source (note the exact source URL for each)

If you find nothing relevant, say so plainly. Do not invent or guess details.
You may explain your search process and reasoning freely in this response — a
second step will structure your findings, so plain prose is fine here.
"""

EXTRACTION_SYSTEM_PROMPT = """You convert research notes into strict JSON. You will be given a \
researcher's findings about a company's international agreements. Extract every \
concrete, sourced item into this exact schema. If the notes contain no concrete \
sourced items (e.g. they say nothing was found, or only mention unconfirmed/vague \
leads without a source URL), return an empty items list.

Respond ONLY with valid JSON, no markdown fences, no commentary, no explanation \
before or after — your entire response must be parseable by json.loads(). Match \
this schema exactly:

{
  "items": [
    {
      "partner_name": "string - the partner organization's name",
      "partner_country": "string or null",
      "agreement_type": "one of: MoU, LOI, Contract, Partnership, Announcement, Other",
      "title": "short headline, under 15 words",
      "summary": "2-3 sentence neutral summary IN YOUR OWN WORDS, not quoted text",
      "event_date": "YYYY-MM-DD or null if unknown",
      "source_url": "string",
      "source_name": "string, e.g. publication name",
      "confidence": "low, medium, or high"
    }
  ]
}
"""


def load_members(supabase_url: str, supabase_key: str) -> list:
    """Fetch the active member list from Supabase (source of truth).

    Falls back to the local members.json only if the Supabase call fails,
    so a transient network issue doesn't stop the whole run.
    """
    endpoint = f"{supabase_url}/rest/v1/members?select=name&active=eq.true&order=name.asc"
    req = urllib.request.Request(
        endpoint,
        method="GET",
        headers={
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            rows = json.loads(resp.read().decode())
            names = [r["name"] for r in rows]
            if names:
                print(f"Loaded {len(names)} active members from Supabase.")
                return names
            print("[warn] Supabase members table returned 0 active rows; falling back to members.json")
    except Exception as e:
        print(f"[warn] could not load members from Supabase ({e}); falling back to members.json", file=sys.stderr)

    with open(MEMBERS_FILE) as f:
        return json.load(f)


def extract_json(client: anthropic.Anthropic, research_notes: str) -> list:
    """Second-pass call: convert free-form research notes into strict JSON."""
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1500,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Researcher's notes:\n\n{research_notes}"},
                {"role": "assistant", "content": "{"},  # prefill forces JSON-only output
            ],
        )
    except Exception as e:
        print(f"  [error] extraction call failed: {e}", file=sys.stderr)
        return []

    text_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    full_text = "{" + "\n".join(text_parts).strip()  # re-add the prefilled brace

    try:
        data = json.loads(full_text)
        return data.get("items", [])
    except json.JSONDecodeError:
        # Fallback: try to salvage a JSON object from within the text
        start = full_text.find("{")
        end = full_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(full_text[start:end + 1])
                return data.get("items", [])
            except json.JSONDecodeError:
                pass
        print(f"  [warn] could not parse extraction JSON: {full_text[:200]}", file=sys.stderr)
        return []


def search_member(client: anthropic.Anthropic, member: str, since: str, max_retries: int = 3) -> list:
    """Research a member's news with web search, then extract structured items in a second call.

    Retries the research call on transient errors (e.g. 529 overloaded, rate
    limits) with exponential backoff before giving up on this member for this run.
    """
    user_prompt = (
        f"Company: \"{member}\"\n"
        f"Search for news published since {since} about this company signing an MoU, "
        f"Letter of Intent, contract, or partnership agreement with an international "
        f"partner (another company, government, or institution outside Belgium, or a "
        f"significant cross-border deal). Belgian domestic-only news without an "
        f"international partner does not count.\n"
        f"Describe what you find, including source URLs, dates, and partner names."
    )

    response = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=2000,
                system=RESEARCH_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
            )
            break  # success
        except anthropic.APIStatusError as e:
            # Transient server-side issues (529 overloaded, 500s) are worth retrying.
            # Client errors like 400/401/403 won't fix themselves on retry.
            transient = e.status_code in (429, 500, 502, 503, 529)
            if transient and attempt < max_retries:
                wait = 2 ** attempt  # 2s, 4s, 8s...
                print(f"  [retry {attempt}/{max_retries}] {member}: {e}. Waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"  [error] API call failed for {member} after {attempt} attempt(s): {e}", file=sys.stderr)
            return []
        except Exception as e:
            print(f"  [error] API call failed for {member}: {e}", file=sys.stderr)
            return []

    if response is None:
        return []

    # Concatenate all text blocks (web search results interleave tool_use/tool_result blocks)
    text_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    research_notes = "\n".join(text_parts).strip()

    if not research_notes:
        return []

    return extract_json(client, research_notes)


def upsert_supabase(rows: list, supabase_url: str, supabase_key: str):
    """POST rows to Supabase REST API with upsert (on conflict do nothing on the unique index)."""
    if not rows:
        return
    endpoint = f"{supabase_url}/rest/v1/announcements"
    body = json.dumps(rows).encode()
    req = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=ignore-duplicates,return=minimal",
        },
    )
    try:
        urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        print(f"  [error] Supabase insert failed: {e.read().decode()}", file=sys.stderr)


def main():
    anthropic_key = os.environ["ANTHROPIC_API_KEY"]
    supabase_url = os.environ["SUPABASE_URL"].rstrip("/")
    supabase_key = os.environ["SUPABASE_KEY"]

    client = anthropic.Anthropic(api_key=anthropic_key)
    members = load_members(supabase_url, supabase_key)
    since = (datetime.date.today() - datetime.timedelta(days=LOOKBACK_DAYS)).isoformat()

    total_found = 0
    errors = []

    print(f"Starting scrape for {len(members)} members, since {since}")

    for i, member in enumerate(members, 1):
        print(f"[{i}/{len(members)}] {member}")
        items = search_member(client, member, since)

        rows = []
        for item in items:
            rows.append({
                "member_name": member,
                "partner_name": item.get("partner_name", "Unknown"),
                "partner_country": item.get("partner_country"),
                "agreement_type": item.get("agreement_type", "Other"),
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "event_date": item.get("event_date"),
                "source_url": item.get("source_url", ""),
                "source_name": item.get("source_name"),
                "confidence": item.get("confidence", "medium"),
            })

        if rows:
            upsert_supabase(rows, supabase_url, supabase_key)
            total_found += len(rows)
            print(f"  -> found {len(rows)} item(s)")

        # gentle pacing to avoid rate limits across ~170 members
        time.sleep(1)

    print(f"Done. {total_found} new item(s) found across {len(members)} members.")


if __name__ == "__main__":
    main()
