"""Frozen query-centered P/G preprocessing candidate identity.

This module contains immutable configuration only.  It performs no model
inference and exposes no evaluator information to search-facing artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from ufuzz.pg_prompts import PG_PREPROCESS_V1_PROMPTS
from ufuzz.semantic_sidecar import canonical_bytes, canonical_sha256

PG_PREPROCESS_V1 = "PG_PREPROCESS_V1"
PG_PREPROCESS_V2 = "PG_PREPROCESS_V2"
SEMANTIC_SCHEMA_VERSION = "semantic-sidecar-v1"
NORMALIZATION_VERSION = "pg-normalization-v1"
VALIDATION_VERSION = "pg-validation-v1"
INFORMATION_FLOW_VERSION = "pg-information-flow-v1"
SELECTOR_SNAPSHOT_SHA256 = "4098f0932bcacb59deac1d8c7ab71ccdda82b67a65d437553c90261cb0443615"

_EXPECTED_PROMPT_DIGESTS = {
    "query_slot": "82ea064fedddc77f3e4b88f0418f32977391ac99ec946c0494985f221e6f9e1e",
    "bounded_proposition_extraction": "7c19ab5daa7ebd7506b97ca795e6a544824cf2964e8ff2ec58791bc27ffd2df5",
    "existing_memory_state": "b2a503fbe8898bd6aba0f5e3eb181489e50a601512455a01433133755ad936e3",
    "query_relevance": "c89e5f3f4eb243ccbcae20fe12e65c5be5df663607722788758bc958f062df38",
    "preference_slot": "2b88241dab483689ce17aa44a981c23f1cc7b8d829778f308f3af84b588ec40f",
    "relation_entity_match": "bc0cae1077fb0db735be0251619ab0e687a6444c2d253cb04470342cfacd17b8",
    "g_evidence_mapping": "28b8852296b478a37822bacc476b2aaa3bbc06aaa66f1f553e76a02e51c58e67",
}

BENCHMARK_ARTIFACTS = {
    "locomo": {
        "repository": "snap-research/locomo",
        "revision": "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376",
        "filename": "data/locomo10.json",
        "sha256": "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
    },
    "longmemeval-s-cleaned": {
        "repository": "xiaowu0162/longmemeval-cleaned",
        "revision": "98d7416c24c778c2fee6e6f3006e7a073259d48f",
        "filename": "longmemeval_s_cleaned.json",
        "sha256": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
    },
}

SEARCH_FORBIDDEN_FIELDS = (
    "gold_answer", "gold_answers", "accepted_answer", "accepted_answers",
    "answer_session_ids", "has_answer", "native_evidence", "evaluator_reference",
    "correctness", "failure_surface", "cfs", "uf", "cov", "backend_result",
    "backend_response", "method_identity",
)

PLACEHOLDER_IDENTITIES = ("", "none", "null", "unknown", "unresolved", "n/a")

VALIDATION_RULES: dict[str, Any] = {
    "normalization": {
        "unicode": "NFKC", "trim": True, "collapse_internal_whitespace": True,
        "casefold_identity_fields_only": True,
        "preserve_semantic_punctuation_and_numbers": True,
    },
    "grounding": {
        "exact_unique_nonempty_span": True, "provenance_required": True,
        "source_ordinal_required": True, "fuzzy_repair": False,
    },
    "seed_local_relation_slot": (
        "root_checkpoint_id", "query_id", "slot_index",
        "normalized_relation_label", "query_anchor",
    ),
    "placeholder_precheck": {
        "fields": ("entity", "relation", "value"),
        "reject_after_normalization": PLACEHOLDER_IDENTITIES,
    },
    "preference_holder": {
        "first_person_requires_holder": "user|speaker|questioner",
        "reject_object_only_subject": True,
        "native_type_may_constrain": "relation_family_only",
        "selection": "query-only if resolved, otherwise query-plus-native-type",
    },
    "opportunity_rules": {
        "meaning_preserving": "resolved canonical seed-local intent",
        "target_changing": "one replaceable slot plus exact grounded alternative",
        "unsupported": "deterministic session-local identifier with exact normalized raw-state absence",
        "update": "exact grounded query-relevant mutable proposition",
        "delete": "exact grounded query-relevant proposition",
        "unrelated_change": "query-independent existing-memory ELIGIBLE and separate relevance UNRELATED",
    },
    "no_global_completeness_claim": True,
}

INFORMATION_FLOW_POLICY: dict[str, Any] = {
    "search_safe_inputs": (
        "raw_query", "checkpoint_source_content", "source_provenance",
        "source_timestamps", "fixed_selector", "native_question_type_relation_family_hint_only",
    ),
    "search_safe_forbidden": SEARCH_FORBIDDEN_FIELDS,
    "evaluator_only_additional_inputs": (
        "benchmark_gold_answer", "locomo_evidence", "answer_session_ids",
        "has_answer", "certified_mutation_operation",
    ),
    "g_never_flows_to_search": True,
}


@dataclass(frozen=True, slots=True)
class PGPreprocessCandidateManifest:
    contract_label: str
    semantic_schema_version: str
    normalization_version: str
    validation_version: str
    information_flow_version: str
    benchmark_artifacts: dict[str, Any]
    model: dict[str, Any]
    selector: dict[str, Any]
    prompts: dict[str, str]
    prompt_sha256: dict[str, str]
    generation_limits: dict[str, int]
    g_chunking: dict[str, int]
    rules: dict[str, Any]
    information_flow: dict[str, Any]
    pilot_artifacts: dict[str, str]
    validation_rule_sha256: str
    information_flow_policy_sha256: str

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self)

    @property
    def sha256_digest(self) -> str:
        return sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class PGPreprocessV2Manifest:
    """Production-executable successor that leaves the V1 bytes untouched."""

    contract_label: str
    supersedes: str
    supersession_reason: str
    semantic_schema_version: str
    normalization_version: str
    validation_version: str
    information_flow_version: str
    benchmark_artifacts: dict[str, Any]
    model: dict[str, Any]
    selector: dict[str, Any]
    selector_snapshot_sha256: str
    prompts: dict[str, str]
    prompt_sha256: dict[str, str]
    generation_limits: dict[str, int]
    g_chunking: dict[str, int]
    rules: dict[str, Any]
    information_flow: dict[str, Any]
    executable_contract: dict[str, Any]
    validation_rule_sha256: str
    information_flow_policy_sha256: str
    executable_contract_sha256: str

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self)

    @property
    def sha256_digest(self) -> str:
        return sha256(self.canonical_bytes).hexdigest()


def prompt_digests() -> dict[str, str]:
    values = {name: sha256(text.encode("utf-8")).hexdigest() for name, text in PG_PREPROCESS_V1_PROMPTS.items()}
    if values != _EXPECTED_PROMPT_DIGESTS:
        raise RuntimeError("frozen PG_PREPROCESS_V1 prompt text differs from its registered digest")
    return values


def pg_preprocess_v1_manifest() -> PGPreprocessCandidateManifest:
    return PGPreprocessCandidateManifest(
        contract_label=PG_PREPROCESS_V1,
        semantic_schema_version=SEMANTIC_SCHEMA_VERSION,
        normalization_version=NORMALIZATION_VERSION,
        validation_version=VALIDATION_VERSION,
        information_flow_version=INFORMATION_FLOW_VERSION,
        benchmark_artifacts=BENCHMARK_ARTIFACTS,
        model={
            "model_id": "Qwen/Qwen3-14B",
            "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
            "dtype": "bfloat16", "runtime": "transformers-4.56.2/torch-2.8.0+cu128",
            "attention": "sdpa", "do_sample": False, "enable_thinking": False,
            "seed": 1729, "retry_policy": "none",
            "tokenizer_json_sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
            "tokenizer_config_sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
            "model_config_sha256": "e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a",
            "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        },
        selector={
            "transport_label": "nomic-embed-text:latest",
            "immutable_model_manifest_sha256": "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f",
            "weights_layer_sha256": "970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6",
            "dimension": 768, "distance": "cosine", "transport_batch_size": 16,
            "top_k": 64,
            "relevant_order": "descending cosine, source ordinal, provenance_id",
            "outside_pool_size": 16,
            "outside_order": "sha256(checkpoint_id + NUL + query_id + NUL + urc-pool-v1 + NUL + provenance_id), then ordinal/provenance",
            "selector_artifact_sha256": "59b2a2cf6e08373d599ef141d82c5e8ae603f6efc642f213d6b0f5820dbe658c",
        },
        prompts=PG_PREPROCESS_V1_PROMPTS,
        prompt_sha256=prompt_digests(),
        generation_limits={
            "query_slot": 512, "bounded_proposition_extraction": 1536,
            "existing_memory_state": 128, "query_relevance": 128,
            "preference_slot": 512, "relation_entity_match": 256,
            "g_evidence_mapping": 1024,
        },
        g_chunking={"max_sources": 10, "max_source_characters": 12000},
        rules=VALIDATION_RULES,
        information_flow=INFORMATION_FLOW_POLICY,
        pilot_artifacts={
            "sample_sha256": "68fcd9d9ca158f6794d80e365c8dd7d6a4cabdbadea7d1b698a299ce8dcc4382",
            "frozen_g_sha256": "383921e8d073f7a5995fe4ad46060c189359c5be464c00935865a8dda1814b59",
        },
        validation_rule_sha256=canonical_sha256(VALIDATION_RULES),
        information_flow_policy_sha256=canonical_sha256(INFORMATION_FLOW_POLICY),
    )


V2_STAGE_ORDER = (
    "source_selection",
    "canonical_candidate_presentation",
    "query_slot_parse",
    "bounded_relevant_and_outside_proposition_extraction",
    "relevant_relation_entity_verification",
    "relevant_existing_memory_state_eligibility",
    "target_changing_relation_entity_verification",
    "target_changing_existing_memory_state_eligibility",
    "outside_existing_memory_state_eligibility",
    "outside_query_relevance",
    "final_opportunity_registration",
    "g_native_evidence_scope_construction",
    "g_bounded_chunk_enumeration",
    "g_one_generation_call_per_chunk",
    "canonical_g_merge",
    "final_g_validation",
)

V2_EXECUTABLE_CONTRACT: dict[str, Any] = {
    "selected_source_membership": "read only from immutable PG_SELECTOR_SNAPSHOT_V2",
    "selected_source_presentation": "source_ordinal ascending, then provenance_id ascending",
    "outside_membership": "first 16 by frozen event-keyed SHA-256 order outside top-64 membership",
    "outside_presentation": "source_ordinal ascending, then provenance_id ascending",
    "generation_call_unit": "exactly one scientific task per model.generate invocation",
    "semantic_batching": False,
    "parallelism": "independent query workers or shards only",
    "within_query_stage_order": V2_STAGE_ORDER,
    "g_merge": {
        "e_plus_order": ("provenance_id", "char_start", "char_end", "component_id"),
        "e_plus_deduplication": "exact canonical region only",
        "accepted_option_order": "canonical sorted component membership, then canonical option bytes",
        "accepted_option_deduplication": "exact canonical option only",
    },
    "selector_service": {
        "purpose": "one-epoch snapshot construction only; never P/G generation",
        "host_scope": "localhost_only",
        "ollama_num_parallel": 1,
        "gpu": 5,
    },
}


def pg_preprocess_v2_manifest() -> PGPreprocessV2Manifest:
    """Return V2 with V1 scientific semantics and explicit execution shape."""

    v1 = pg_preprocess_v1_manifest()
    selector = dict(v1.selector)
    selector.update(
        {
            "artifact_label": "PG_SELECTOR_SNAPSHOT_V2",
            "selector_snapshot_sha256": SELECTOR_SNAPSHOT_SHA256,
            "membership_consumption": "immutable snapshot only; live recomputation forbidden",
            "relevant_presentation_order": "source_ordinal ascending, then provenance_id ascending",
            "outside_presentation_order": "source_ordinal ascending, then provenance_id ascending",
            "ollama_num_parallel": 1,
        }
    )
    return PGPreprocessV2Manifest(
        contract_label=PG_PREPROCESS_V2,
        supersedes=PG_PREPROCESS_V1,
        supersession_reason="EXECUTABLE_CONTRACT_UNDERSPECIFICATION_BEFORE_HUMAN_VALIDATION",
        semantic_schema_version=v1.semantic_schema_version,
        normalization_version=v1.normalization_version,
        validation_version="pg-validation-v2",
        information_flow_version=v1.information_flow_version,
        benchmark_artifacts=v1.benchmark_artifacts,
        model=v1.model,
        selector=selector,
        selector_snapshot_sha256=SELECTOR_SNAPSHOT_SHA256,
        prompts=v1.prompts,
        prompt_sha256=v1.prompt_sha256,
        generation_limits=v1.generation_limits,
        g_chunking=v1.g_chunking,
        rules=v1.rules,
        information_flow=v1.information_flow,
        executable_contract=V2_EXECUTABLE_CONTRACT,
        validation_rule_sha256=v1.validation_rule_sha256,
        information_flow_policy_sha256=v1.information_flow_policy_sha256,
        executable_contract_sha256=canonical_sha256(V2_EXECUTABLE_CONTRACT),
    )
