import argparse
import hashlib
import json
from typing import Any, Dict, List, Tuple


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def extract_finish_reason(entry: dict) -> str:
    fr = entry.get("finish_reason")
    if fr:
        return fr
    resp = entry.get("response") or {}
    choices = resp.get("choices") or []
    if choices and isinstance(choices, list):
        ch0 = choices[0] or {}
        fr = ch0.get("finish_reason")
        if fr:
            return fr
    return None


def compute_messages_hash(entry: dict) -> str:
    """
    Compute a stable hash based on request content that should define a test case:
    - messages
    - tools
    - tool_choice

    Ignores model/base_url/api_key/other fields by design.
    """
    req = entry.get("request", {})
    key_obj = {
        "messages": req.get("messages", []),
        "tools": req.get("tools", None),
        "tool_choice": req.get("tool_choice", None),
    }
    # Canonicalize to a stable JSON string with sorted keys
    serialized = json.dumps(key_obj, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def build_map_by_messages_hash(rows: List[dict]) -> Tuple[Dict[str, dict], int]:
    """
    Returns (mapping: hash -> entry, duplicate_count)
    Keeps the first occurrence for any duplicate hash.
    """
    mapping: Dict[str, dict] = {}
    dups = 0
    for r in rows:
        h = compute_messages_hash(r)
        if h in mapping:
            dups += 1
            continue
        mapping[h] = r
    return mapping, dups


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute F1 using messages-hash alignment only.")
    parser.add_argument("--ours", required=True, help="Path to our results.jsonl")
    parser.add_argument("--official", required=True, help="Path to comparison results.jsonl")
    parser.add_argument("--out", required=True, help="Path to write summary JSON")
    args = parser.parse_args()

    ours_rows = load_jsonl(args.ours)
    off_rows = load_jsonl(args.official)

    ours_map, ours_dups = build_map_by_messages_hash(ours_rows)
    off_map, off_dups = build_map_by_messages_hash(off_rows)

    keys_ours = set(ours_map.keys())
    keys_off = set(off_map.keys())
    aligned = sorted(keys_ours & keys_off)

    TP = FP = FN = TN = 0
    count_finish_reason_tool_calls = 0
    count_successful_tool_call = 0

    def is_tool_calls(fr: Any) -> bool:
        return isinstance(fr, str) and fr == "tool_calls"

    for h in aligned:
        ours = ours_map[h]
        off = off_map[h]
        fr_ours = extract_finish_reason(ours)
        fr_off = extract_finish_reason(off)

        ours_tc = is_tool_calls(fr_ours)
        off_tc = is_tool_calls(fr_off)

        if ours_tc and off_tc:
            TP += 1
        elif ours_tc and not off_tc:
            FP += 1
        elif not ours_tc and off_tc:
            FN += 1
        else:
            TN += 1

        if ours_tc:
            count_finish_reason_tool_calls += 1
            if bool(ours.get("tool_calls_valid")):
                count_successful_tool_call += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    schema_accuracy = (
        (count_successful_tool_call / count_finish_reason_tool_calls) if count_finish_reason_tool_calls > 0 else 0.0
    )

    summary = {
        "alignment": {
            "ours_total_unique_hashes": len(ours_map),
            "official_total_unique_hashes": len(off_map),
            "aligned": len(aligned),
            "duplicate_ours_hashes": ours_dups,
            "duplicate_official_hashes": off_dups,
            "key_type": "messages_tools_tool_choice_hash",
        },
        "tool_call_trigger_similarity": {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "tool_call_f1": round(f1 * 100, 2),
        },
        "tool_call_schema_accuracy": {
            "count_finish_reason_tool_calls": count_finish_reason_tool_calls,
            "count_successful_tool_call": count_successful_tool_call,
            "schema_accuracy": round(schema_accuracy * 100, 2),
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
