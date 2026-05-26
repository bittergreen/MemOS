from __future__ import annotations

import logging

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from memos.dream.contextualization import CONTEXT_MEMORY_TYPE


if TYPE_CHECKING:
    from memos.api.product_models import APISearchRequest


logger = logging.getLogger(__name__)

_DEFAULT_CONTEXT_RECALL_TOP_K = 2
_CONTEXT_RETURN_FIELDS = [
    "memory",
    "key",
    "created_at",
    "updated_at",
    "source",
    "internal_info",
]


@dataclass
class DreamContextSearchExtension:
    """Dream-owned search extension for recalling Context nodes.

    The core SearchHandler only exposes a generic plugin hook. This extension
    owns Dream-specific retrieval details such as the Context memory type,
    graph scope, metadata formatting, and fallback behavior.
    """

    top_k: int = _DEFAULT_CONTEXT_RECALL_TOP_K

    def merge_context_recall(
        self,
        *,
        handler,
        search_req: APISearchRequest,
        results: dict[str, Any],
    ) -> dict[str, Any]:
        top_k = max(0, int(self.top_k or 0))
        if top_k <= 0:
            return results

        context_buckets = self._recall_context_buckets(
            handler=handler,
            search_req=search_req,
            top_k=top_k,
        )
        if context_buckets:
            results.setdefault("text_mem", []).extend(context_buckets)
        return results

    def _recall_context_buckets(
        self, *, handler, search_req: APISearchRequest, top_k: int
    ) -> list[dict[str, Any]]:
        graph_db = getattr(handler, "graph_db", None) or getattr(
            handler.searcher, "graph_store", None
        )
        embedder = getattr(handler, "embedder", None) or getattr(handler.searcher, "embedder", None)
        if graph_db is None or embedder is None:
            logger.info("[Dream Search] Context recall skipped: graph_db or embedder unavailable.")
            return []

        try:
            query_embedding = embedder.embed([search_req.query])[0]
        except Exception:
            logger.warning("[Dream Search] Context recall embedding failed.", exc_info=True)
            return []

        buckets: list[dict[str, Any]] = []
        for cube_id in _resolve_cube_ids(search_req):
            try:
                hits = graph_db.search_by_embedding(
                    query_embedding,
                    top_k=top_k,
                    scope=CONTEXT_MEMORY_TYPE,
                    status="activated",
                    user_name=cube_id,
                    return_fields=_CONTEXT_RETURN_FIELDS,
                )
            except Exception:
                logger.warning(
                    "[Dream Search] Context recall search failed for cube=%s.",
                    cube_id,
                    exc_info=True,
                )
                continue

            hydrated_hits = _hydrate_context_hits(graph_db, hits or [], cube_id)
            memories = [
                _format_context_hit(hit) for hit in hydrated_hits if _context_hit_memory(hit)
            ]
            if not memories:
                continue
            buckets.append(
                {
                    "cube_id": cube_id,
                    "memories": memories,
                    "total_nodes": len(memories),
                }
            )
        return buckets


def _resolve_cube_ids(search_req: APISearchRequest) -> list[str]:
    if search_req.readable_cube_ids:
        return list(dict.fromkeys(search_req.readable_cube_ids))
    return [search_req.user_id]


def _format_context_hit(hit: dict[str, Any]) -> dict[str, Any]:
    context_id = str(hit.get("id", ""))
    score = float(hit.get("score", 0.0) or 0.0)
    memory = _context_hit_memory(hit)
    key = _context_hit_field(hit, "key", "")
    metadata = {
        "id": context_id,
        "memory": memory,
        "memory_type": CONTEXT_MEMORY_TYPE,
        "source": _context_hit_field(hit, "source", "dream") or "dream",
        "key": key,
        "relativity": score,
        "score": score,
        "embedding": [],
        "sources": [],
        "usage": [],
        "ref_id": f"[{context_id.split('-')[0]}]" if context_id else "[context]",
    }
    for field in ("created_at", "updated_at", "internal_info"):
        value = _context_hit_field(hit, field)
        if value is not None:
            metadata[field] = value

    return {
        "id": context_id,
        "memory": memory,
        "metadata": metadata,
        "ref_id": metadata["ref_id"],
    }


def _hydrate_context_hits(
    graph_db, hits: list[dict[str, Any]], cube_id: str
) -> list[dict[str, Any]]:
    missing_ids = [
        str(hit.get("id", ""))
        for hit in hits
        if isinstance(hit, dict) and hit.get("id") and not _context_hit_memory(hit)
    ]
    if not missing_ids:
        return hits

    nodes_by_id = _fetch_context_nodes(graph_db, missing_ids, cube_id)
    if not nodes_by_id:
        return hits

    hydrated: list[dict[str, Any]] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        node = nodes_by_id.get(str(hit.get("id", "")))
        if node is None or _context_hit_memory(hit):
            hydrated.append(hit)
            continue
        hydrated.append(_merge_context_hit_with_node(hit, node))
    return hydrated


def _fetch_context_nodes(graph_db, ids: list[str], cube_id: str) -> dict[str, dict[str, Any]]:
    unique_ids = list(dict.fromkeys(ids))
    nodes: list[dict[str, Any]] = []

    get_nodes = getattr(graph_db, "get_nodes", None)
    if callable(get_nodes):
        try:
            batch_nodes = get_nodes(unique_ids, user_name=cube_id, include_embedding=False)
            if isinstance(batch_nodes, list):
                nodes.extend(node for node in batch_nodes if isinstance(node, dict))
        except TypeError:
            try:
                batch_nodes = get_nodes(unique_ids, user_name=cube_id)
                if isinstance(batch_nodes, list):
                    nodes.extend(node for node in batch_nodes if isinstance(node, dict))
            except Exception:
                logger.warning("[Dream Search] Context get_nodes fallback failed.", exc_info=True)
        except Exception:
            logger.warning("[Dream Search] Context get_nodes fallback failed.", exc_info=True)

    found_ids = {str(node.get("id", "")) for node in nodes}
    missing_ids = [node_id for node_id in unique_ids if node_id not in found_ids]
    get_node = getattr(graph_db, "get_node", None)
    if callable(get_node):
        for node_id in missing_ids:
            try:
                node = get_node(node_id, user_name=cube_id, include_embedding=False)
            except TypeError:
                try:
                    node = get_node(node_id, user_name=cube_id)
                except Exception:
                    logger.warning(
                        "[Dream Search] Context get_node fallback failed for id=%s.",
                        node_id,
                        exc_info=True,
                    )
                    continue
            except Exception:
                logger.warning(
                    "[Dream Search] Context get_node fallback failed for id=%s.",
                    node_id,
                    exc_info=True,
                )
                continue
            if isinstance(node, dict):
                nodes.append(node)

    return {str(node.get("id", "")): node for node in nodes if node.get("id")}


def _merge_context_hit_with_node(hit: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    merged = dict(hit)
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    merged["memory"] = node.get("memory", "") or metadata.get("memory", "")
    for field in _CONTEXT_RETURN_FIELDS:
        if field == "memory":
            continue
        if node.get(field) is not None:
            merged[field] = node[field]
        elif metadata.get(field) is not None:
            merged[field] = metadata[field]
    return merged


def _context_hit_memory(hit: dict[str, Any]) -> str:
    memory = hit.get("memory", "")
    if memory:
        return str(memory)
    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    return str(metadata.get("memory", "") or "")


def _context_hit_field(hit: dict[str, Any], field: str, default: Any | None = None) -> Any:
    if hit.get(field) is not None:
        return hit[field]
    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    return metadata.get(field, default)
