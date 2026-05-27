import argparse
import json
import os
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

from dotenv import load_dotenv


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
EVAL_ENV_PATH = os.path.join(ROOT_DIR, "evaluation", ".env")
DEFAULT_INPUT_PATH = os.path.join(ROOT_DIR, "data", "locomo", "locomo10.json")
FALLBACK_INPUT_PATH = os.path.join(ROOT_DIR, "evaluation", "data", "locomo", "locomo10.json")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "results", "locomo_raw_search")


def score_from_judgments(item: dict[str, Any]) -> float:
    judgments = item.get("llm_judgments") or {}
    if not judgments:
        return 0.0
    return sum(1 for value in judgments.values() if value) / len(judgments)


def compact_timespec(timespec: Any) -> dict[str, Any] | None:
    if not isinstance(timespec, dict):
        return None
    anchor = timespec.get("time_anchor")
    if not isinstance(anchor, dict):
        return None
    return {
        "start": anchor.get("start"),
        "end": anchor.get("end"),
        "granularity": anchor.get("granularity"),
        "source": anchor.get("source"),
        "is_ongoing": timespec.get("is_ongoing"),
    }


def compact_history_entry(entry: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "archived_memory_id": entry.get("archived_memory_id"),
        "version": entry.get("version"),
        "update_type": entry.get("update_type"),
        "memory_form": entry.get("memory_form"),
        "timespec": compact_timespec(entry.get("timespec")),
        "created_at": entry.get("created_at"),
        "memory_preview": (entry.get("memory") or "")[:160],
    }
    return {k: v for k, v in compact.items() if v not in (None, [], {}, "")}


def build_search_metadata_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    internal_info = (
        metadata.get("internal_info") if isinstance(metadata.get("internal_info"), dict) else {}
    )
    info = metadata.get("info") if isinstance(metadata.get("info"), dict) else {}
    history = metadata.get("history") if isinstance(metadata.get("history"), list) else []
    history_compact = [compact_history_entry(item) for item in history if isinstance(item, dict)]
    summary = {
        "version": metadata.get("version"),
        "memory_type": metadata.get("memory_type"),
        "key": metadata.get("key"),
        "ref_id": metadata.get("ref_id"),
        "covered_history": metadata.get("covered_history"),
        "timespec": compact_timespec(internal_info.get("timespec") or info.get("timespec")),
        "selected_from_history": (
            internal_info.get("selected_from_history")
            if "selected_from_history" in internal_info
            else info.get("selected_from_history")
        ),
        "memory_form": internal_info.get("memory_form"),
        "history_len": len(history_compact),
        "history_with_timespec_count": sum(1 for item in history_compact if item.get("timespec")),
        "history_sample": history_compact[:3],
    }
    return {k: v for k, v in summary.items() if v not in (None, [], {}, "")}


def compact_db_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    internal_info = (
        metadata.get("internal_info") if isinstance(metadata.get("internal_info"), dict) else {}
    )
    info = metadata.get("info") if isinstance(metadata.get("info"), dict) else {}
    history = metadata.get("history") if isinstance(metadata.get("history"), list) else []
    history_compact = [compact_history_entry(item) for item in history if isinstance(item, dict)]
    summary = {
        "version": metadata.get("version"),
        "memory_type": metadata.get("memory_type"),
        "key": metadata.get("key"),
        "ref_id": metadata.get("ref_id"),
        "covered_history": metadata.get("covered_history"),
        "timespec": compact_timespec(internal_info.get("timespec") or info.get("timespec")),
        "selected_from_history": (
            internal_info.get("selected_from_history")
            if "selected_from_history" in internal_info
            else info.get("selected_from_history")
        ),
        "memory_form": internal_info.get("memory_form"),
        "history_len": len(history_compact),
        "history_with_timespec_count": sum(1 for item in history_compact if item.get("timespec")),
        "history_sample": history_compact[:3],
    }
    return {k: v for k, v in summary.items() if v not in (None, [], {}, "")}


def rendered_time_prefix(memory_text: str) -> bool:
    stripped = (memory_text or "").strip()
    return stripped.startswith(
        (
            "[Time:",
            "[Since:",
            "[Valid:",
            "[As of:",
            "[Until:",
            "[时间:",
            "[自",
            "[截至:",
            "[有效期:",
        )
    )


def normalize_search_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "cube_id": bucket.get("cube_id"),
        "memories": [],
    }

    for mem in bucket.get("memories", []):
        if not isinstance(mem, dict):
            continue
        metadata = mem.get("metadata") or {}
        normalized["memories"].append(
            {
                "id": mem.get("id"),
                "memory": mem.get("memory"),
                "version": metadata.get("version"),
                "history_len": len(metadata.get("history") or []),
                "metadata": metadata,
            }
        )

    return normalized


def slim_history_entry(entry: dict[str, Any]) -> dict[str, Any]:
    slim = {
        "memory": entry.get("memory"),
        "version": entry.get("version"),
        "update_type": entry.get("update_type"),
        "archived_memory_id": entry.get("archived_memory_id"),
        "created_at": entry.get("created_at"),
    }
    if "timespec" in entry:
        slim["timespec"] = compact_timespec(entry.get("timespec"))
    if "memory_form" in entry:
        slim["memory_form"] = entry.get("memory_form")
    return {k: v for k, v in slim.items() if v not in (None, [], {}, "")}


def slim_memory_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    internal_info = (
        metadata.get("internal_info") if isinstance(metadata.get("internal_info"), dict) else {}
    )
    slim = {
        "version": metadata.get("version"),
        "history": [
            slim_history_entry(h) for h in (metadata.get("history") or []) if isinstance(h, dict)
        ],
        "covered_history": metadata.get("covered_history"),
        "memory_type": metadata.get("memory_type"),
        "key": metadata.get("key"),
        "tags": metadata.get("tags"),
        "created_at": metadata.get("created_at"),
        "updated_at": metadata.get("updated_at"),
        "ref_id": metadata.get("ref_id"),
        "timespec": compact_timespec(internal_info.get("timespec")),
        "selected_from_history": internal_info.get("selected_from_history"),
        "memory_form": internal_info.get("memory_form"),
    }
    return {k: v for k, v in slim.items() if v not in (None, [], {}, "")}


def slim_memory_item(mem: dict[str, Any]) -> dict[str, Any]:
    metadata = slim_memory_metadata(mem.get("metadata") or {})
    slim = {
        "id": mem.get("id"),
        "memory": mem.get("memory"),
        "ref_id": mem.get("ref_id"),
        "relativity": (mem.get("metadata") or {}).get("relativity"),
        "rendered_time_prefix": rendered_time_prefix(mem.get("memory") or ""),
        "metadata": metadata,
        "selector_debug": mem.get("selector_debug"),
    }
    return {k: v for k, v in slim.items() if v not in (None, [], {}, "")}


def slim_search_data(search_data: dict[str, Any]) -> dict[str, Any]:
    slim_text_mem = []
    for bucket in search_data.get("text_mem", []) or []:
        slim_bucket = {
            "cube_id": bucket.get("cube_id"),
            "memories": [
                slim_memory_item(mem)
                for mem in (bucket.get("memories") or [])
                if isinstance(mem, dict)
            ],
        }
        slim_text_mem.append({k: v for k, v in slim_bucket.items() if v not in (None, [], {}, "")})

    slim = {
        "text_mem": slim_text_mem,
        "pref_string": search_data.get("pref_string"),
    }
    return {k: v for k, v in slim.items() if v not in (None, [], {}, "")}


def unwrap_full_item_metadata(item: dict[str, Any]) -> dict[str, Any]:
    """
    /product/get_memory_by_ids may return either:
    1. a normal item with metadata fields directly under item["metadata"], or
    2. a nested wrapper where item["metadata"]["metadata"] contains the real metadata.
    """
    metadata = item.get("metadata") or {}
    if isinstance(metadata, dict):
        nested = metadata.get("metadata")
        if isinstance(nested, dict):
            return nested
    return metadata if isinstance(metadata, dict) else {}


class SearchCaptureClient:
    def __init__(self, base_url: str, api_key: str | None, mode: str, pref_top_k: int):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = api_key
        self.mode = mode
        self.pref_top_k = pref_top_k
        self.max_retries = 5

    def search(self, query: str, user_id: str, top_k: int) -> dict[str, Any]:
        payload = {
            "query": query,
            "user_id": user_id,
            "mem_cube_id": user_id,
            "conversation_id": "",
            "top_k": top_k,
            "mode": self.mode,
            "include_preference": True,
            "pref_top_k": self.pref_top_k,
        }
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/product/search",
                    headers=self.headers,
                    data=json.dumps(payload, ensure_ascii=False),
                    timeout=120,
                )
                response.raise_for_status()
                body = response.json()
                if body.get("message") != "Search completed successfully":
                    raise RuntimeError(f"Unexpected search response: {body}")
                return body.get("data") or {}
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_seconds = 2**attempt
                    print(
                        f"⚠️  locomo_capture search retry {attempt + 1}/{self.max_retries} failed"
                        f" for user_id={user_id}, query={query[:80]}..., error={e}."
                        f" Retrying in {wait_seconds}s."
                    )
                    time.sleep(wait_seconds)
                else:
                    raise last_error from e

    def get_memory_by_ids(self, memory_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not memory_ids:
            return {}

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/product/get_memory_by_ids",
                    headers=self.headers,
                    data=json.dumps(memory_ids, ensure_ascii=False),
                    timeout=120,
                )
                response.raise_for_status()
                body = response.json()
                memories = (body.get("data") or {}).get("memories") or []
                result: dict[str, dict[str, Any]] = {}

                for item in memories:
                    if not isinstance(item, dict):
                        continue
                    item_id = item.get("id")
                    if item_id:
                        result[item_id] = item

                return result
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_seconds = 2**attempt
                    print(
                        f"⚠️  locomo_capture get_memory_by_ids retry {attempt + 1}/{self.max_retries}"
                        f" failed for {len(memory_ids)} ids, error={e}. Retrying in {wait_seconds}s."
                    )
                    time.sleep(wait_seconds)
                else:
                    raise last_error from e


def enrich_search_data(
    client: SearchCaptureClient, search_data: dict[str, Any], force_enrich: bool = False
) -> tuple[dict[str, Any], int]:
    ids_to_fetch: list[str] = []

    for bucket in search_data.get("text_mem", []) or []:
        memories = bucket.get("memories") or []
        for mem in memories:
            if not isinstance(mem, dict):
                continue
            if mem.get("id"):
                ids_to_fetch.append(mem["id"])

    fetched = client.get_memory_by_ids(sorted(set(ids_to_fetch)))
    enriched_count = 0

    for bucket in search_data.get("text_mem", []) or []:
        memories = bucket.get("memories") or []
        for mem in memories:
            if not isinstance(mem, dict):
                continue
            item_id = mem.get("id")
            full_item = fetched.get(item_id)
            if not item_id or not full_item:
                continue

            full_metadata = unwrap_full_item_metadata(full_item)
            current_metadata = mem.get("metadata") or {}

            if force_enrich or "history" not in current_metadata:
                current_metadata["history"] = full_metadata.get("history") or []
            if force_enrich or "version" not in current_metadata:
                current_metadata["version"] = full_metadata.get("version")
            if force_enrich or (
                "covered_history" not in current_metadata and "covered_history" in full_metadata
            ):
                current_metadata["covered_history"] = full_metadata.get("covered_history")
            full_internal = (
                full_metadata.get("internal_info")
                if isinstance(full_metadata.get("internal_info"), dict)
                else {}
            )
            current_internal = (
                current_metadata.get("internal_info")
                if isinstance(current_metadata.get("internal_info"), dict)
                else {}
            )
            for field in ("timespec", "memory_form", "selected_from_history"):
                if force_enrich or (field not in current_internal and field in full_internal):
                    current_internal[field] = full_internal.get(field)
            if current_internal:
                current_metadata["internal_info"] = current_internal

            mem["metadata"] = current_metadata
            mem["selector_debug"] = {
                "search_metadata": build_search_metadata_summary(current_metadata),
                "db_metadata": compact_db_metadata(full_metadata),
                "rendered_time_prefix": rendered_time_prefix(mem.get("memory") or ""),
            }
            enriched_count += 1

    return search_data, enriched_count


def resolve_judged_path(path: str) -> str:
    if os.path.isdir(path):
        judged_path = os.path.join(path, "memos-api_locomo_judged.json")
        if os.path.exists(judged_path):
            return judged_path
        raise FileNotFoundError(f"No memos-api_locomo_judged.json found under directory: {path}")
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Judged file or result directory not found: {path}")


def load_wrong_question_filter(path: str, threshold: float) -> dict[int, set[str]]:
    judged_path = resolve_judged_path(path)
    with open(judged_path, encoding="utf-8") as f:
        judged = json.load(f)
    if not isinstance(judged, dict):
        raise ValueError(f"Expected dict judged file in {judged_path}")

    wrong_by_conv: dict[int, set[str]] = {}
    for conv_key, items in judged.items():
        if not isinstance(items, list):
            continue
        try:
            conv_idx = int(str(conv_key).rsplit("_", 1)[-1])
        except Exception:
            continue
        wrong_questions = {
            item.get("question", "")
            for item in items
            if isinstance(item, dict)
            and item.get("question")
            and score_from_judgments(item) < threshold
        }
        if wrong_questions:
            wrong_by_conv[conv_idx] = wrong_questions
    return wrong_by_conv


def capture_single_question(
    client: SearchCaptureClient,
    conv_idx: int,
    qa: dict[str, Any],
    speaker_a_user_id: str,
    speaker_b_user_id: str,
    top_k: int,
    force_enrich: bool,
) -> dict[str, Any]:
    query = qa.get("question", "")
    category = qa.get("category")
    answer = qa.get("answer")
    evidence = qa.get("evidence") or []

    start = time.time()
    speaker_a_raw = client.search(query=query, user_id=speaker_a_user_id, top_k=top_k)
    speaker_b_raw = client.search(query=query, user_id=speaker_b_user_id, top_k=top_k)
    speaker_a_raw, enriched_a = enrich_search_data(client, speaker_a_raw, force_enrich=force_enrich)
    speaker_b_raw, enriched_b = enrich_search_data(client, speaker_b_raw, force_enrich=force_enrich)
    speaker_a_raw = slim_search_data(speaker_a_raw)
    speaker_b_raw = slim_search_data(speaker_b_raw)
    elapsed_ms = (time.time() - start) * 1000

    return {
        "conv_idx": conv_idx,
        "question": query,
        "category": category,
        "golden_answer": answer,
        "evidence": evidence,
        "search_duration_ms": elapsed_ms,
        "speaker_a": {
            "user_id": speaker_a_user_id,
            "raw": speaker_a_raw,
            "enriched_items": enriched_a,
        },
        "speaker_b": {
            "user_id": speaker_b_user_id,
            "raw": speaker_b_raw,
            "enriched_items": enriched_b,
        },
    }


def process_conversation(
    client: SearchCaptureClient,
    row: dict[str, Any],
    conv_idx: int,
    version: str,
    top_k: int,
    force_enrich: bool,
    workers: int,
    allowed_questions: set[str] | None = None,
) -> list[dict[str, Any]]:
    qa_set = row.get("qa") or []
    qa_set = [qa for qa in qa_set if qa.get("category") != 5]
    if allowed_questions is not None:
        qa_set = [qa for qa in qa_set if qa.get("question", "") in allowed_questions]

    speaker_a_user_id = f"locomo_exp_user_{conv_idx}_speaker_a_{version}"
    speaker_b_user_id = f"locomo_exp_user_{conv_idx}_speaker_b_{version}"

    results: list[dict[str, Any]] = []
    total_questions = len(qa_set)
    completed_questions = 0
    conv_start = time.time()
    print(
        f"[locomo_capture] conversation={conv_idx} start total_questions={total_questions} "
        f"workers={workers} top_k={top_k} force_enrich={force_enrich}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                capture_single_question,
                client,
                conv_idx,
                qa,
                speaker_a_user_id,
                speaker_b_user_id,
                top_k,
                force_enrich,
            )
            for qa in qa_set
        ]

        for future in as_completed(futures):
            results.append(future.result())
            completed_questions += 1
            elapsed = time.time() - conv_start
            remaining = total_questions - completed_questions
            print(
                f"[locomo_capture] conversation={conv_idx} progress="
                f"{completed_questions}/{total_questions} remaining={remaining} "
                f"elapsed_sec={elapsed:.1f}",
                flush=True,
            )

    results.sort(key=lambda item: item["question"])
    print(
        f"[locomo_capture] conversation={conv_idx} done total_questions={total_questions} "
        f"elapsed_sec={time.time() - conv_start:.1f}",
        flush=True,
    )
    return results


def load_dataset(path: str) -> list[dict[str, Any]]:
    if (
        not os.path.exists(path)
        and path == DEFAULT_INPUT_PATH
        and os.path.exists(FALLBACK_INPUT_PATH)
    ):
        path = FALLBACK_INPUT_PATH
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset in {path}, got {type(data).__name__}")
    return data


def load_existing_capture(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict capture file in {path}, got {type(data).__name__}")
    if "results" not in data or not isinstance(data["results"], dict):
        raise ValueError(f"Capture file {path} is missing dict field 'results'")
    return data


def main() -> None:
    load_dotenv()
    load_dotenv(EVAL_ENV_PATH, override=False)

    parser = argparse.ArgumentParser(
        description="Capture raw LoCoMo search results, including memory history when available."
    )
    parser.add_argument(
        "--input", type=str, default=DEFAULT_INPUT_PATH, help="LoCoMo dataset path."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Defaults to results/locomo_raw_search/locomo_search_with_history_<version>.json",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("MEMOS_URL", "http://127.0.0.1:8001"),
        help="Base URL for memos-api.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("MEMOS_KEY"),
        help="Optional Authorization header value.",
    )
    parser.add_argument(
        "--version", type=str, default="default", help="Version suffix for user ids."
    )
    parser.add_argument("--top-k", type=int, default=20, help="Search top_k.")
    parser.add_argument(
        "--mode",
        type=str,
        default=os.getenv("SEARCH_MODE", "fast"),
        choices=["fast", "fine", "mix", "deep", "agentic"],
        help="Search mode to send to /product/search.",
    )
    parser.add_argument("--pref-top-k", type=int, default=6, help="Preference top_k.")
    parser.add_argument(
        "--conv-idx",
        type=int,
        action="append",
        default=None,
        help="Only process the given LoCoMo conversation index. Repeatable.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel question workers within a conversation.",
    )
    parser.add_argument(
        "--force-enrich",
        action="store_true",
        help="Always call get_memory_by_ids to fetch full memory items, even if search already returns history.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output file by skipping conversations already present.",
    )
    parser.add_argument(
        "--wrong-only-from",
        type=str,
        default="",
        help="Optional judged JSON path or LoCoMo result directory. When set, only capture questions "
        "whose judged score is below --wrong-threshold.",
    )
    parser.add_argument(
        "--wrong-threshold",
        type=float,
        default=1.0,
        help="Questions with judged score below this threshold are treated as wrong for --wrong-only-from.",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(
            DEFAULT_OUTPUT_DIR, f"locomo_search_with_history_{args.version}.json"
        )

    dataset = load_dataset(args.input)
    wrong_filter = (
        load_wrong_question_filter(args.wrong_only_from, threshold=args.wrong_threshold)
        if args.wrong_only_from
        else {}
    )
    selected_indices = args.conv_idx if args.conv_idx else list(range(len(dataset)))
    if wrong_filter:
        selected_indices = [idx for idx in selected_indices if wrong_filter.get(idx)]
    client = SearchCaptureClient(
        base_url=args.base_url,
        api_key=args.api_key,
        mode=args.mode,
        pref_top_k=args.pref_top_k,
    )

    captured: dict[str, Any] = {
        "meta": {
            "input": args.input,
            "base_url": args.base_url,
            "version": args.version,
            "top_k": args.top_k,
            "mode": args.mode,
            "pref_top_k": args.pref_top_k,
            "selected_conversations": selected_indices,
            "wrong_only_from": args.wrong_only_from or None,
            "wrong_threshold": args.wrong_threshold if args.wrong_only_from else None,
        },
        "results": {},
    }
    if args.resume:
        existing_capture = load_existing_capture(output_path)
        if existing_capture is not None:
            captured = existing_capture
            captured["meta"].update(
                {
                    "input": args.input,
                    "base_url": args.base_url,
                    "version": args.version,
                    "top_k": args.top_k,
                    "mode": args.mode,
                    "pref_top_k": args.pref_top_k,
                    "selected_conversations": selected_indices,
                    "wrong_only_from": args.wrong_only_from or None,
                    "wrong_threshold": args.wrong_threshold if args.wrong_only_from else None,
                    "resumed": True,
                }
            )
            print(
                f"[locomo_capture] resume enabled loaded_existing_output={output_path} "
                f"existing_conversations={len(captured['results'])}",
                flush=True,
            )

    for conv_idx in selected_indices:
        row = dataset[conv_idx]
        conv_key = f"locomo_exp_user_{conv_idx}"
        if args.resume and conv_key in captured["results"]:
            existing_count = len(captured["results"][conv_key] or [])
            print(
                f"[locomo_capture] skipping conversation {conv_idx} because it already exists "
                f"in output with {existing_count} questions",
                flush=True,
            )
            continue
        print(
            f"[locomo_capture] starting conversation {conv_idx} "
            f"({selected_indices.index(conv_idx) + 1}/{len(selected_indices)})",
            flush=True,
        )
        conv_results = process_conversation(
            client=client,
            row=row,
            conv_idx=conv_idx,
            version=args.version,
            top_k=args.top_k,
            force_enrich=args.force_enrich,
            workers=args.workers,
            allowed_questions=wrong_filter.get(conv_idx),
        )
        captured["results"][conv_key] = conv_results
        print(f"Captured conversation {conv_idx}: {len(conv_results)} questions -> {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(captured, f, ensure_ascii=False, indent=2)
        print(
            f"[locomo_capture] checkpoint saved after conversation {conv_idx} -> {output_path}",
            flush=True,
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(captured, f, ensure_ascii=False, indent=2)

    print(f"Saved raw search capture to {output_path}")


if __name__ == "__main__":
    main()
