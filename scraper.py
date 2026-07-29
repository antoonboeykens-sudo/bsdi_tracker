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

SYSTEM_PROMPT = """You are a research assistant that finds recent, real news about a specific \
defence/security company signing agreements with international partners \
(Memoranda of Understanding, Letters of Intent, contracts, or partnership announcements).

Only report items that are:
- Genuinely about the named company (not a similarly named unrelated company)
- Dated within the requested lookback window
- Backed by a real, checkable news source (return the exact source URL)

If you find nothing, return an empty items list. Do not invent or guess details.
Respond ONLY with valid JSON, no markdown fences, no commentary, matching this schema:

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


def load_members():
    with open(MEMBERS_FILE) as f:
        return json.load(f)


def search_member(client: anthropic.Anthropic, member: str, since: str) -> list:
    """Query Claude with web search for one member; return list of item dicts."""
    user_prompt = (
        f"Company: \"{member}\"\n"
        f"Search for news published since {since} about this company signing an MoU, "
        f"Letter of Intent, contract, or partnership agreement with an international "
        f"partner (another company, government, or institution outside Belgium, or a "
        f"significant cross-border deal). Belgian domestic-only news without an "
        f"international partner does not count.\n"
        f"Return the JSON object described in your instructions."
    )

    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
        )
    except Exception as e:
        print(f"  [error] API call failed for {member}: {e}", file=sys.stderr)
        return []

    # Concatenate all text blocks (web search results interleave tool_use/tool_result blocks)
    text_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    full_text = "\n".join(text_parts).strip()

    # Strip stray code fences if the model added them despite instructions
    if full_text.startswith("```"):
        full_text = full_text.strip("`")
        if full_text.lower().startswith("json"):
            full_text = full_text[4:]

    try:
        data = json.loads(full_text)
        return data.get("items", [])
    except json.JSONDecodeError:
        print(f"  [warn] could not parse JSON for {member}: {full_text[:200]}", file=sys.stderr)
        return []


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
    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_KEY"]

    client = anthropic.Anthropic(api_key=anthropic_key)
    members = load_members()
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
