import argparse
import json
import os
import sys

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from time import time

import pandas as pd

from tqdm import tqdm


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
EVAL_SCRIPTS_DIR = os.path.join(ROOT_DIR, "evaluation", "scripts")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, EVAL_SCRIPTS_DIR)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.prompts import (  # noqa: E402
    MEM0_CONTEXT_TEMPLATE,
    MEM0_GRAPH_CONTEXT_TEMPLATE,
    MEMOS_CONTEXT_TEMPLATE,
)


try:
    from locomo.locomo_capture_search_history import (
        build_search_metadata_summary,
        compact_db_metadata,
        rendered_time_prefix,
        slim_search_data,
    )
except ModuleNotFoundError:
    from evaluation.scripts.locomo.locomo_capture_search_history import (
        build_search_metadata_summary,
        compact_db_metadata,
        rendered_time_prefix,
        slim_search_data,
    )


def _attach_inline_selector_debug(search_results):
    for bucket in search_results.get("text_mem", []) or []:
        for mem in bucket.get("memories", []) or []:
            if not isinstance(mem, dict):
                continue
            metadata = mem.get("metadata") or {}
            metadata_summary = build_search_metadata_summary(metadata)
            mem["selector_debug"] = {
                "search_metadata": metadata_summary,
                "db_metadata": compact_db_metadata(metadata),
                "rendered_time_prefix": rendered_time_prefix(mem.get("memory") or ""),
            }
    return search_results


def mem0_search(client, query, user_id, top_k):
    start = time()
    results = client.search(query, user_id, top_k)
    memory = [f"{memory['created_at']}: {memory['memory']}" for memory in results["results"]]
    if client.enable_graph:
        graph = "\n".join(
            [
                f"  - 'source': {item.get('source', '?')} -> 'target': {item.get('target', '?')} "
                f"(relationship: {item.get('relationship', '?')})"
                for item in results.get("relations", [])
            ]
        )
        context = MEM0_GRAPH_CONTEXT_TEMPLATE.format(
            user_id=user_id, memories=memory, relations=graph
        )
    else:
        context = MEM0_CONTEXT_TEMPLATE.format(user_id=user_id, memories=memory)
    duration_ms = (time() - start) * 1000
    return context, duration_ms


def memos_search(client, query, user_id, top_k, reference_time=None):
    start = time()
    results = client.search(
        query=query,
        user_id=user_id,
        top_k=top_k,
        reference_time=reference_time,
    )
    text_mem = results.get("text_mem") or []
    memory_lines = []
    for bucket in text_mem:
        memories = bucket.get("memories") or []
        for item in memories:
            if isinstance(item, dict) and item.get("memory"):
                memory_lines.append(item["memory"])

    pref_string = results.get("pref_string", "")
    context_parts = []
    if memory_lines:
        context_parts.append("\n".join(memory_lines))
    if pref_string:
        context_parts.append(pref_string)

    context = "\n".join(context_parts)
    context = MEMOS_CONTEXT_TEMPLATE.format(user_id=user_id, memories=context)
    duration_ms = (time() - start) * 1000

    enriched = _attach_inline_selector_debug(results)
    capture_payload = {
        "user_id": user_id,
        "raw": slim_search_data(enriched),
        "enriched_items": sum(
            len(bucket.get("memories") or []) for bucket in (enriched.get("text_mem") or [])
        ),
    }
    return context, duration_ms, capture_payload


def memobase_search(client, query, user_id, top_k):
    start = time()
    context = client.search(query=query, user_id=user_id, top_k=top_k)
    duration_ms = (time() - start) * 1000
    return context, duration_ms


def memu_search(client, query, user_id, top_k):
    start = time()
    results = client.search(query, user_id, top_k)
    context = "\n".join(results)
    duration_ms = (time() - start) * 1000
    return context, duration_ms


def supermemory_search(client, query, user_id, top_k):
    start = time()
    context = client.search(query, user_id, top_k)
    duration_ms = (time() - start) * 1000
    return context, duration_ms


def process_user(lme_df, conv_idx, frame, version, top_k=20):
    row = lme_df.iloc[conv_idx]
    question = row["question"]
    sessions = row["haystack_sessions"]
    question_type = row["question_type"]
    question_date = row["question_date"]
    answer = row["answer"]
    answer_session_ids = set(row["answer_session_ids"])
    haystack_session_ids = row["haystack_session_ids"]
    user_id = f"lme_exper_user_{version}_{conv_idx}"
    id_to_session = dict(zip(haystack_session_ids, sessions, strict=False))
    answer_sessions = [id_to_session[sid] for sid in answer_session_ids if sid in id_to_session]
    answer_evidences = []

    for session in answer_sessions:
        for turn in session:
            if turn.get("has_answer"):
                data = turn.get("role") + " : " + turn.get("content")
                answer_evidences.append(data)

    search_results = defaultdict(list)
    capture_results = defaultdict(list)
    print("\n" + "-" * 80)
    print(f"🔎 [{conv_idx + 1}/{len(lme_df)}] Processing conversation {conv_idx}")
    print(f"❓ Question: {question}")
    print(f"📅 Date: {question_date}")
    print(f"🏷️  Type: {question_type}")
    print("-" * 80)

    existing_results, existing_capture, exists = load_existing_results(frame, version, conv_idx)
    if exists:
        print(f"♻️  Using existing results for conversation {conv_idx}")
        return existing_results, existing_capture

    capture_payload = None
    if "mem0" in frame:
        from utils.client import Mem0Client

        client = Mem0Client(enable_graph="graph" in frame)
        context, duration_ms = mem0_search(client, question, user_id, top_k)
    elif frame == "memobase":
        from utils.client import MemobaseClient

        client = MemobaseClient()
        context, duration_ms = memobase_search(client, question, user_id, top_k)
    elif frame == "memos-api":
        from utils.client import MemosApiClient

        client = MemosApiClient()
        context, duration_ms, capture_payload = memos_search(
            client,
            question,
            user_id,
            top_k,
            reference_time=question_date,
        )
    elif frame == "memos-api-online":
        from utils.client import MemosApiOnlineClient

        client = MemosApiOnlineClient()
        context, duration_ms, capture_payload = memos_search(
            client,
            question,
            user_id,
            top_k,
            reference_time=question_date,
        )
    elif frame == "memu":
        from utils.client import MemuClient

        client = MemuClient()
        context, duration_ms = memu_search(client, question, user_id, top_k)
    elif frame == "supermemory":
        from utils.client import SupermemoryClient

        client = SupermemoryClient()
        context, duration_ms = supermemory_search(client, question, user_id, top_k)

    search_results[user_id].append(
        {
            "question": question,
            "category": question_type,
            "date": question_date,
            "golden_answer": answer,
            "answer_evidences": answer_evidences,
            "search_context": context,
            "search_duration_ms": duration_ms,
        }
    )

    if capture_payload is not None:
        capture_results[user_id].append(
            {
                "conv_idx": conv_idx,
                "user_id": user_id,
                "question": question,
                "category": question_type,
                "question_date": question_date,
                "golden_answer": answer,
                "answer_session_ids": list(answer_session_ids),
                "answer_evidences": answer_evidences,
                "search_duration_ms": duration_ms,
                "raw": capture_payload["raw"],
                "enriched_items": capture_payload["enriched_items"],
            }
        )

    os.makedirs(f"results/lme/{frame}-{version}/tmp", exist_ok=True)
    with open(
        f"results/lme/{frame}-{version}/tmp/{frame}_lme_search_results_{conv_idx}.json", "w"
    ) as f:
        json.dump(search_results, f, indent=4)
    if capture_results:
        with open(
            f"results/lme/{frame}-{version}/tmp/{frame}_lme_search_with_history_{conv_idx}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(dict(capture_results), f, ensure_ascii=False, indent=2)
    print(f"💾 Search results for conversation {conv_idx} saved...")
    print("-" * 80)

    return search_results, capture_results


def load_existing_results(frame, version, group_idx):
    result_path = f"results/lme/{frame}-{version}/tmp/{frame}_lme_search_results_{group_idx}.json"
    capture_path = (
        f"results/lme/{frame}-{version}/tmp/{frame}_lme_search_with_history_{group_idx}.json"
    )
    if os.path.exists(result_path):
        try:
            with open(result_path) as f:
                search_data = json.load(f)
            capture_data = {}
            if os.path.exists(capture_path):
                with open(capture_path) as f:
                    capture_data = json.load(f)
            return search_data, capture_data, True
        except Exception as e:
            print(f"❌ Error loading existing results for group {group_idx}: {e}")
    return {}, {}, False


def main(frame, version, top_k=20, num_workers=2):
    print("\n" + "=" * 80)
    print(f"🔍 LONGMEMEVAL SEARCH - {frame.upper()} v{version}".center(80))
    print("=" * 80)

    lme_df = pd.read_json("data/longmemeval/longmemeval_s.json")
    print("📚 Loaded LongMemeval dataset from data/longmemeval/longmemeval_s.json")
    num_multi_sessions = len(lme_df)
    print(f"👥 Number of users: {num_multi_sessions}")
    print(f"⚙️  Search parameters: top_k={top_k}, workers={num_workers}")
    print("-" * 80)

    all_search_results = defaultdict(list)
    all_capture_results = defaultdict(list)
    start_time = datetime.now()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(process_user, lme_df, idx, frame, version, top_k): idx
            for idx in range(num_multi_sessions)
        }

        for future in tqdm(
            as_completed(future_to_idx), total=num_multi_sessions, desc="📊 Processing users"
        ):
            _idx = future_to_idx[future]
            search_results, capture_results = future.result()
            for user_id, results in search_results.items():
                all_search_results[user_id].extend(results)
            for user_id, results in capture_results.items():
                all_capture_results[user_id].extend(results)

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    elapsed_time_str = str(elapsed_time).split(".")[0]

    print("\n" + "=" * 80)
    print("✅ SEARCH COMPLETE".center(80))
    print("=" * 80)
    print(f"⏱️  Total time taken to search {num_multi_sessions} users: {elapsed_time_str}")
    print(f"🔄 Framework: {frame} | Version: {version} | Workers: {num_workers}")

    with open(f"results/lme/{frame}-{version}/{frame}_lme_search_results.json", "w") as f:
        json.dump(dict(all_search_results), f, indent=4)
    print(f"📁 Results saved to: results/lme/{frame}-{version}/{frame}_lme_search_results.json")

    if all_capture_results:
        capture_output = {
            "meta": {
                "input": "data/longmemeval/longmemeval_s.json",
                "frame": frame,
                "version": version,
                "top_k": top_k,
                "mode": os.getenv("SEARCH_MODE", "fast"),
                "pref_top_k": 6,
                "selected_indices": list(range(num_multi_sessions)),
            },
            "results": dict(all_capture_results),
        }
        capture_path = f"results/lme/{frame}-{version}/{frame}_lme_search_with_history.json"
        with open(capture_path, "w", encoding="utf-8") as f:
            json.dump(capture_output, f, ensure_ascii=False, indent=2)
        print(f"📁 Search-with-history saved to: {capture_path}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LongMemeval Search Script")
    parser.add_argument(
        "--lib",
        type=str,
        choices=[
            "mem0",
            "mem0_graph",
            "memos-api",
            "memos-api-online",
            "memobase",
            "memu",
            "supermemory",
        ],
        default="memos-api",
    )
    parser.add_argument(
        "--version", type=str, default="default", help="Version of the evaluation framework."
    )
    parser.add_argument(
        "--top_k", type=int, default=30, help="Number of top results to retrieve from the search."
    )
    parser.add_argument(
        "--workers", type=int, default=30, help="Number of runs for LLM-as-a-Judge evaluation."
    )

    args = parser.parse_args()

    main(frame=args.lib, version=args.version, top_k=args.top_k, num_workers=args.workers)
