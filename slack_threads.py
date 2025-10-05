"""
Utilities to rebuild Slack threads from an export and feed them to an LLM.

Features
- Walks a Slack standard export directory (folder per channel, JSON per day)
- Reconstructs threads that may span multiple day-files
- Returns a normalized structure: {channel, thread_ts, root, replies}
- CLI to dump threads, or pipe each thread into your LLM Q&A extractor

Usage
-----
# Rebuild threads and print summary
python slack_threads.py /path/to/slack-export --list

# Dump all threads to JSON (one file with array of threads)
python slack_threads.py /path/to/slack-export --out threads.json

# Run your LLM extractor (expects LLM/qa_extractor.py on import path)
python slack_threads.py /path/to/slack-export --extract

Notes
-----
- A thread's root may be missing from the export slice you have (e.g., only replies
  appear). In that case, `root` will be None and replies will still be present.
- Message ordering is based on numeric `ts` (timestamp) ascending.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Message:
    channel: str
    ts: str
    text: str
    user: Optional[str]
    thread_ts: Optional[str]
    subtype: Optional[str]
    raw: dict

    @property
    def ts_float(self) -> float:
        try:
            return float(self.ts)
        except Exception:
            return 0.0


@dataclass
class Thread:
    channel: str
    thread_ts: str
    root: Optional[Message]
    replies: List[Message]


def _read_channel_day_file(path: Path, channel: str) -> Iterable[Message]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}", file=sys.stderr)
        return []

    messages = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            if item.get("subtype") and not item.get("text"):
                continue
        ts = item.get("ts")
        if not ts:
            continue
        msg = Message(
            channel=channel,
            ts=ts,
            text=item.get("text", ""),
            user=item.get("user") or item.get("bot_id") or item.get("username"),
            thread_ts=item.get("thread_ts"),
            subtype=item.get("subtype"),
            raw=item,
        )
        messages.append(msg)
    return messages


def load_all_messages(export_root: Path) -> List[Message]:
    msgs: List[Message] = []
    for channel_dir in sorted(p for p in export_root.iterdir() if p.is_dir()):
        channel = channel_dir.name
        json_files = sorted(channel_dir.glob("*.json"))
        for jf in json_files:
            msgs.extend(_read_channel_day_file(jf, channel))
    return msgs


def build_threads(messages: Iterable[Message]) -> List[Thread]:
    by_channel: Dict[str, List[Message]] = {}
    for m in messages:
        by_channel.setdefault(m.channel, []).append(m)

    threads: List[Thread] = []

    for channel, msgs in by_channel.items():
        roots = {}
        for m in msgs:
            if m.raw.get("replies") or m.raw.get("reply_count"):
                roots[m.ts] = m

        grouped: Dict[str, List[Message]] = {}
        for m in msgs:
            key = m.thread_ts or (m.ts if m.ts in roots else None)
            if not key:
                continue
            grouped.setdefault(key, []).append(m)

        for tkey, group in grouped.items():
            group.sort(key=lambda x: x.ts_float)
            root = None
            for m in group:
                if m.ts == tkey:
                    root = m
                    break
            if not root and tkey in roots:
                root = roots[tkey]

            replies = [m for m in group if root is None or m.ts != root.ts]
            threads.append(Thread(channel=channel, thread_ts=tkey, root=root, replies=replies))

    threads.sort(key=lambda th: (th.channel, (th.root.ts_float if th.root else (th.replies[0].ts_float if th.replies else 0.0))))
    return threads


def thread_to_minimal_dict(th: Thread) -> dict:
    def msg_min(m: Message) -> dict:
        return {
            "ts": m.ts,
            "text": m.text,
            "user": m.user,
            "thread_ts": m.thread_ts,
            "subtype": m.subtype,
        }

    return {
        "channel": th.channel,
        "thread_ts": th.thread_ts,
        "root": msg_min(th.root) if th.root else None,
        "replies": [msg_min(r) for r in th.replies],
    }


def thread_to_llm_text(th: Thread) -> str:
    lines: List[str] = []
    header = f"Channel: #{th.channel}\nThread: {th.thread_ts}"
    lines.append(header)
    lines.append("---")
    if th.root:
        lines.append(f"[ROOT] {th.root.user or 'unknown'} @ {th.root.ts}:\n{th.root.text}\n")
    else:
        lines.append("[ROOT] (missing in this export slice)\n")
    for r in th.replies:
        lines.append(f"[REPLY] {r.user or 'unknown'} @ {r.ts}:\n{r.text}\n")
    return "\n".join(lines)


def threads_from_json(path: Path) -> List[Thread]:
    """Load threads back from a dumped JSON file into Thread objects."""
    data = json.loads(path.read_text(encoding="utf-8"))
    threads: List[Thread] = []
    for t in data:
        root = None
        if t.get("root"):
            root = Message(
                channel=t["channel"],
                ts=t["root"]["ts"],
                text=t["root"]["text"],
                user=t["root"].get("user"),
                thread_ts=t["root"].get("thread_ts"),
                subtype=t["root"].get("subtype"),
                raw=t["root"],
            )
        replies = []
        for r in t.get("replies", []):
            replies.append(
                Message(
                    channel=t["channel"],
                    ts=r["ts"],
                    text=r["text"],
                    user=r.get("user"),
                    thread_ts=r.get("thread_ts"),
                    subtype=r.get("subtype"),
                    raw=r,
                )
            )
        threads.append(Thread(channel=t["channel"], thread_ts=t["thread_ts"], root=root, replies=replies))
    return threads


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Rebuild Slack threads from export and optionally run an extractor.")
    p.add_argument("export_dir", type=Path, help="Path to Slack export root directory")
    p.add_argument("--list", action="store_true", help="Print a human-readable summary of threads")
    p.add_argument("--out", type=Path, help="Write the output as json")
    p.add_argument("--extract", action="store_true", help="Run LLM/qa_extractor.extract_qas on each thread and emit results JSONL")
    p.add_argument("--model", default="gpt-4o-mini", help="Model to use with --extract")

    args = p.parse_args(argv)

    export_root = args.export_dir
    if not export_root.exists() or not export_root.is_dir():
        print(f"Error: {export_root} is not a directory", file=sys.stderr)
        return 2

    messages = load_all_messages(export_root)
    threads = build_threads(messages)

    if args.list:
        for th in threads:
            root_user = th.root.user if th.root else "(missing)"
            root_ts = th.root.ts if th.root else th.thread_ts
            print(f"#{th.channel} thread {th.thread_ts} | root {root_user} @ {root_ts} | replies: {len(th.replies)}")



    if args.extract:
        try:
            try:
                from LLM.qa_extractor import extract_qas
            except Exception:
                here = Path(__file__).resolve().parent
                llm_path = here / "LLM"
                if llm_path.exists():
                    sys.path.append(str(llm_path))
                from qa_extractor import extract_qas  # type: ignore
        except Exception as e:
            print("Error: Could not import extract_qas. Ensure LLM/qa_extractor.py is available.", file=sys.stderr)
            print(e, file=sys.stderr)
            return 3

        out_path = Path("tmp.json")
        with out_path.open("w", encoding="utf-8") as f:
            count = 0
            for th in threads:
                    thread_text = thread_to_llm_text(th)
                    try:
                        result = extract_qas(thread_text, model=args.model)
                        #print (count)
                    except Exception as e:
                        print(f"Extractor failed for thread {th.thread_ts}: {e}", file=sys.stderr)
                        continue
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    #print(json.dumps(result, ensure_ascii=False))
                    #print (count ,"is done")
                    count += 1
        print(f"Wrote {count} extraction results to {out_path}")


    if args.out:
        #all_threads = [thread_to_minimal_dict(th) for th in threads]
        #with args.out.open("w", encoding="utf-8") as f:
            #json.dump(all_threads, f, indent=2, ensure_ascii=False)
        #print(f"Wrote {len(all_threads)} threads to {args.out} (JSON array)")

        data = []
        input_file = "tmp.json"
        output_file = args.out

        # Read JSONL file line by line
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                item=json.loads(line.strip())
                if item['qas'] != []:
                   data.append(json.loads(line.strip()))

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)



    if not (args.list or args.out or args.extract):
        print("Loaded messages:", len(messages))
        print("Reconstructed threads:", len(threads))
        print("Use --list, --out, or --extract for actions.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
