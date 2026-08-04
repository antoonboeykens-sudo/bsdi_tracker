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
FORCE_LOOKBACK_DAYS = int(_lookback_raw) if _lookback_raw else None  # manual override for ALL members, ignores tracking
BACKFILL_START_DATE = os.environ.get("BACKFILL_START_DATE", "2026-01-01").strip()  # used for members never scraped before
CATCHUP_OVERLAP_DAYS = 3  # small overlap when resuming from last_scraped_until, in case a story was published right at the edge
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
    """Fetch active members (name + last_scraped_until) from Supabase.

    Returns a list of dicts: {"name": ..., "last_scraped_until": "YYYY-MM-DD" or None}
    Falls back to the local members.json (treated as never-scraped) only if
    the Supabase call fails, so a transient network issue doesn't stop the run.
    """
    endpoint = f"{supabase_url}/rest/v1/members?select=name,last_scraped_until&active=eq.true&order=name.asc"
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
            if rows:
                print(f"Loaded {len(rows)} active members from Supabase.")
                return [{"name": r["name"], "last_scraped_until": r.get("last_scraped_until")} for r in rows]
            print("[warn] Supabase members table returned 0 active rows; falling back to members.json")
    except urllib.error.HTTPError as e:
        # Print (masked) diagnostics without ever printing the key itself
        masked_url = endpoint.split("?")[0]
        print(f"[warn] could not load members from Supabase: HTTP {e.code} at {masked_url}", file=sys.stderr)
        print(f"[warn] response body: {e.read().decode()[:300]}", file=sys.stderr)
        print(f"[warn] SUPABASE_URL length={len(supabase_url)}, SUPABASE_KEY length={len(supabase_key)} "
              f"(sanity check — should be ~40-50 chars and ~200+ chars respectively, with no stray whitespace)",
              file=sys.stderr)
        print("[warn] falling back to members.json", file=sys.stderr)
    except Exception as e:
        print(f"[warn] could not load members from Supabase ({e}); falling back to members.json", file=sys.stderr)

    with open(MEMBERS_FILE) as f:
        names = json.load(f)
    return [{"name": n, "last_scraped_until": None} for n in names]


def update_last_scraped(member: str, until_date: str, supabase_url: str, supabase_key: str):
    """PATCH the member's last_scraped_until so the next run resumes from here."""
    import urllib.parse
    endpoint = f"{supabase_url}/rest/v1/members?name=eq.{urllib.parse.quote(member)}"
    body = json.dumps({"last_scraped_until": until_date}).encode()
    req = urllib.request.Request(
        endpoint,
        data=body,
        method="PATCH",
        headers={
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
    )
    try:
        urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        print(f"  [error] could not update last_scraped_until for {member}: {e.read().decode()}", file=sys.stderr)


def extract_json(client: anthropic.Anthropic, research_notes: str) -> list:
    """Second-pass call: convert free-form research notes into strict JSON.

    If the response gets cut off (hits the token limit) before finishing,
    retries once with a larger limit rather than trying to parse broken JSON.
    """
    for max_tokens in (4000, 8000):
        try:
            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=max_tokens,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": f"Researcher's notes:\n\n{research_notes}"},
                ],
            )
        except Exception as e:
            print(f"  [error] extraction call failed: {e}", file=sys.stderr)
            return []

        if response.stop_reason == "max_tokens":
            print(f"  [warn] extraction response truncated at max_tokens={max_tokens}, retrying with more room...", file=sys.stderr)
            continue

        text_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        full_text = "\n".join(text_parts).strip()

        if full_text.startswith("```"):
            full_text = full_text.strip("`")
            if full_text.lower().startswith("json"):
                full_text = full_text[4:]

        try:
            data = json.loads(full_text)
            return data.get("items", [])
        except json.JSONDecodeError:
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

    print("  [error] extraction still truncated after retry with larger limit; giving up for this member", file=sys.stderr)
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
                max_tokens=3500,
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
    """POST rows to Supabase REST API, skipping duplicates on (member_name, partner_name, source_url).

    on_conflict must name the exact columns behind the unique index, or
    PostgREST doesn't know what counts as a duplicate and the whole batch
    insert fails if any single row collides -- silently losing every other
    valid row in that same batch.
    """
    if not rows:
        return
    endpoint = f"{supabase_url}/rest/v1/announcements?on_conflict=member_name,partner_name,source_url"
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
    anthropic_key = os.environ["ANTHROPIC_API_KEY"].strip()
    supabase_url = os.environ["SUPABASE_URL"].strip().rstrip("/")
    if supabase_url.endswith("/rest/v1"):
        supabase_url = supabase_url[: -len("/rest/v1")]
    supabase_key = os.environ["SUPABASE_KEY"].strip()

    client = anthropic.Anthropic(api_key=anthropic_key)
    members = load_members(supabase_url, supabase_key)
    today = datetime.date.today()
    today_iso = today.isoformat()

    total_found = 0
    new_members_backfilled = 0

    print(f"Starting scrape for {len(members)} members (today: {today_iso})")
    if FORCE_LOOKBACK_DAYS is not None:
        print(f"Manual override active: searching all members since {FORCE_LOOKBACK_DAYS} days ago, ignoring tracking.")

    for i, m in enumerate(members, 1):
        member = m["name"]
        last_until = m.get("last_scraped_until")

        if FORCE_LOOKBACK_DAYS is not None:
            since = (today - datetime.timedelta(days=FORCE_LOOKBACK_DAYS)).isoformat()
        elif last_until:
            since = (datetime.date.fromisoformat(last_until) - datetime.timedelta(days=CATCHUP_OVERLAP_DAYS)).isoformat()
        else:
            since = BACKFILL_START_DATE
            new_members_backfilled += 1

        tag = "[NEW - backfilling]" if not last_until and FORCE_LOOKBACK_DAYS is None else ""
        print(f"[{i}/{len(members)}] {member} (since {since}) {tag}")

        items = search_member(client, member, since)

        rows = []
        skipped_out_of_range = 0
        for item in items:
            event_date = item.get("event_date")
            # Hard safety filter: the model is instructed to only report items
            # since `since`, but that's a soft instruction, not a real search
            # filter. Enforce it here rather than trusting compliance blindly.
            if event_date and event_date < since:
                skipped_out_of_range += 1
                continue
            rows.append({
                "member_name": member,
                "partner_name": item.get("partner_name", "Unknown"),
                "partner_country": item.get("partner_country"),
                "agreement_type": item.get("agreement_type", "Other"),
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "event_date": event_date,
                "source_url": item.get("source_url", ""),
                "source_name": item.get("source_name"),
                "confidence": item.get("confidence", "medium"),
            })

        if skipped_out_of_range:
            print(f"  [filtered] dropped {skipped_out_of_range} item(s) dated before {since}")

        if rows:
            upsert_supabase(rows, supabase_url, supabase_key)
            total_found += len(rows)
            print(f"  -> found {len(rows)} item(s)")

        # Record progress so next run resumes from today, regardless of whether
        # this was a manual override run — we've now checked up to today either way.
        update_last_scraped(member, today_iso, supabase_url, supabase_key)

        # gentle pacing to avoid rate limits across many members
        time.sleep(1)

    print(f"Done. {total_found} new item(s) found across {len(members)} members "
          f"({new_members_backfilled} newly backfilled from {BACKFILL_START_DATE}).")


if __name__ == "__main__":
    main()
