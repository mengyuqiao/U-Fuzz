"""Pure production-configuration and pilot-protocol contract for RQ1--RQ4.

This module freezes configuration principles and pre-registers decision
protocols.  It performs no scheduling, backend access, model inference,
evaluation, serving, logging, or experiment execution.  Exact model choices,
numeric retrieval depth, retry caps, prompts, and native backend profiles stay
unbound until the corresponding registered pilot or capability validation is
complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import hmac
from types import MappingProxyType

from ufuzz.evaluation_contract import (
    ALL_METHODS,
    BACKENDS,
    ENGINEERING_TARGET,
    NATIVE_MEMORY_LLM_CONDITIONS,
    Backend,
    CampaignSpec,
    MethodSpec,
    NativeMemoryLLMCondition,
    NativeMemoryLLMProvider,
)
from ufuzz.scheduler_contract import (
    EngineeringOnlySetting,
    ProductionBindingRequirement,
    RandomStreamName,
    required_production_bindings,
)


class BackendProfileUse(StrEnum):
    PRIMARY_NATIVE = "primary_native"
    DETERMINISTIC_VALIDATION = "deterministic_validation"


class BackendReadiness(StrEnum):
    VALIDATED_VALIDATION_TRACK = "validated_validation_track"
    REQUIRES_NATIVE_PROFILE_VALIDATION = "requires_native_profile_validation"
    REQUIRES_MATERIALIZER_AND_REPLAY_PROOF = (
        "requires_materializer_and_replay_proof"
    )
    PRODUCTION_READY = "production_ready"


@dataclass(frozen=True, slots=True)
class BackendProfileStatus:
    backend: Backend
    profile_id: str
    profile_use: BackendProfileUse
    readiness: BackendReadiness
    intended_native_profile: bool
    exact_configuration_frozen: bool
    live_validation_complete: bool
    materializer_complete: bool

    def __post_init__(self) -> None:
        if not isinstance(self.backend, Backend):
            raise TypeError("backend must be a Backend")
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError("profile_id must be a non-empty string")
        if not isinstance(self.profile_use, BackendProfileUse):
            raise TypeError("profile_use must be a BackendProfileUse")
        if not isinstance(self.readiness, BackendReadiness):
            raise TypeError("readiness must be a BackendReadiness")
        flags = (
            self.intended_native_profile,
            self.exact_configuration_frozen,
            self.live_validation_complete,
            self.materializer_complete,
        )
        if any(not isinstance(value, bool) for value in flags):
            raise TypeError("backend readiness flags must be bools")
        if self.profile_use is BackendProfileUse.DETERMINISTIC_VALIDATION:
            if self.intended_native_profile:
                raise ValueError("a validation profile cannot claim native-primary status")
            if self.readiness is not BackendReadiness.VALIDATED_VALIDATION_TRACK:
                raise ValueError("validation profiles require validation-track readiness")
        elif not self.intended_native_profile:
            raise ValueError("a primary profile must be an intended native profile")

    @property
    def primary_ready(self) -> bool:
        return (
            self.profile_use is BackendProfileUse.PRIMARY_NATIVE
            and self.intended_native_profile
            and self.readiness is BackendReadiness.PRODUCTION_READY
            and self.exact_configuration_frozen
            and self.live_validation_complete
            and self.materializer_complete
        )


MEM0_PROFILE_A = BackendProfileStatus(
    Backend.MEM0,
    "mem0-profile-a-deterministic-validation",
    BackendProfileUse.DETERMINISTIC_VALIDATION,
    BackendReadiness.VALIDATED_VALIDATION_TRACK,
    False,
    True,
    True,
    True,
)
MEM0_PROFILE_B_TARGET = BackendProfileStatus(
    Backend.MEM0,
    "mem0-profile-b-native-infer-true-unbound",
    BackendProfileUse.PRIMARY_NATIVE,
    BackendReadiness.REQUIRES_NATIVE_PROFILE_VALIDATION,
    True,
    False,
    False,
    False,
)
AMEM_NATIVE_TARGET = BackendProfileStatus(
    Backend.AMEM,
    "a-mem-native-primary-unbound",
    BackendProfileUse.PRIMARY_NATIVE,
    BackendReadiness.REQUIRES_MATERIALIZER_AND_REPLAY_PROOF,
    True,
    False,
    False,
    False,
)
GRAPHITI_NATIVE_TARGET = BackendProfileStatus(
    Backend.GRAPHITI,
    "graphiti-native-primary-unbound",
    BackendProfileUse.PRIMARY_NATIVE,
    BackendReadiness.REQUIRES_MATERIALIZER_AND_REPLAY_PROOF,
    True,
    False,
    False,
    False,
)
MEMOS_NATIVE_TARGET = BackendProfileStatus(
    Backend.MEMOS,
    "memos-native-primary-unbound",
    BackendProfileUse.PRIMARY_NATIVE,
    BackendReadiness.REQUIRES_MATERIALIZER_AND_REPLAY_PROOF,
    True,
    False,
    False,
    False,
)

PRIMARY_BACKEND_PROFILE_TARGETS = MappingProxyType(
    {
        Backend.MEM0: MEM0_PROFILE_B_TARGET,
        Backend.AMEM: AMEM_NATIVE_TARGET,
        Backend.GRAPHITI: GRAPHITI_NATIVE_TARGET,
        Backend.MEMOS: MEMOS_NATIVE_TARGET,
    }
)


@dataclass(frozen=True, slots=True)
class BackendFidelityContract:
    primary_backends: frozenset[Backend]
    primary_uses_intended_native_profiles: bool
    cross_backend_internal_component_equality_required: bool
    same_profile_within_backend_across_methods: bool
    validation_profiles_must_be_labeled: bool
    validation_profiles_may_substitute_for_primary: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "primary_backends", frozenset(self.primary_backends))
        if self.primary_backends != frozenset(BACKENDS):
            raise ValueError("all four frozen backends require primary profiles")
        if not all(
            (
                self.primary_uses_intended_native_profiles,
                self.same_profile_within_backend_across_methods,
                self.validation_profiles_must_be_labeled,
            )
        ):
            raise ValueError("native fidelity and within-backend fairness are mandatory")
        if (
            self.cross_backend_internal_component_equality_required
            or self.validation_profiles_may_substitute_for_primary
        ):
            raise ValueError("synthetic component equality cannot replace native fidelity")


BACKEND_FIDELITY = BackendFidelityContract(
    frozenset(BACKENDS), True, False, True, True, False
)


RQ4_FORBIDDEN_CREDENTIAL_FIELDS = frozenset(
    {"api_key", "api_secret", "bearer_token", "credential", "credentials"}
)


@dataclass(frozen=True, slots=True)
class NativeMemoryLLMConfigurationManifest:
    """Self-contained causal RQ4 memory-native LLM configuration."""

    provider: NativeMemoryLLMProvider
    model_identifier: str
    model_revision: str
    memory_system_prompt_config_artifact: bytes
    structured_output_schema_artifact: bytes
    parser_version: str
    decoding_parameters_artifact: bytes
    retry_resampling_policy_artifact: bytes
    api_runtime_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.provider, NativeMemoryLLMProvider):
            raise TypeError("provider must be a NativeMemoryLLMProvider")
        for name in (
            "model_identifier",
            "model_revision",
            "parser_version",
            "api_runtime_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "memory_system_prompt_config_artifact",
            "structured_output_schema_artifact",
            "decoding_parameters_artifact",
            "retry_resampling_policy_artifact",
        ):
            value = getattr(self, name)
            if not isinstance(value, bytes) or not value:
                raise ValueError(f"{name} must contain exact non-empty bytes")

    @property
    def artifact_bytes(self) -> bytes:
        return _canonical_fields(
            "UFUZZ_NATIVE_MEMORY_LLM_MANIFEST_V1",
            self.provider.value,
            self.model_identifier,
            self.model_revision,
            self.memory_system_prompt_config_artifact,
            self.structured_output_schema_artifact,
            self.parser_version,
            self.decoding_parameters_artifact,
            self.retry_resampling_policy_artifact,
            self.api_runtime_version,
        )

    @property
    def sha256_digest(self) -> str:
        return sha256(self.artifact_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class InvariantBackendSubstrateManifest:
    """Exact RQ4 backend substrate held fixed across provider conditions."""

    backend: Backend
    backend_source_version: str
    backend_profile_version: str
    storage_database_configuration_artifact: bytes
    embedding_provider_family: str
    embedding_model_identifier: str
    embedding_revision_or_weights_digest: str
    vector_dimension: int
    vector_normalization: str
    retrieval_search_configuration_artifact: bytes
    reranker_cross_encoder_configuration_artifact: bytes
    retrieval_k: int
    isolation_replay_profile_version: str

    def __post_init__(self) -> None:
        if self.backend not in (Backend.MEM0, Backend.GRAPHITI):
            raise ValueError("RQ4 substrate applies only to Mem0 and Graphiti")
        for name in (
            "backend_source_version",
            "backend_profile_version",
            "embedding_provider_family",
            "embedding_model_identifier",
            "embedding_revision_or_weights_digest",
            "vector_normalization",
            "isolation_replay_profile_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "storage_database_configuration_artifact",
            "retrieval_search_configuration_artifact",
            "reranker_cross_encoder_configuration_artifact",
        ):
            value = getattr(self, name)
            if not isinstance(value, bytes) or not value:
                raise ValueError(f"{name} must contain exact non-empty bytes")
        if (
            isinstance(self.vector_dimension, bool)
            or not isinstance(self.vector_dimension, int)
            or self.vector_dimension <= 0
        ):
            raise ValueError("vector_dimension must be a positive integer")
        if (
            isinstance(self.retrieval_k, bool)
            or not isinstance(self.retrieval_k, int)
            or self.retrieval_k < 2
        ):
            raise ValueError("retrieval_k must be an integer of at least two")

    @property
    def artifact_bytes(self) -> bytes:
        return _canonical_fields(
            "UFUZZ_INVARIANT_BACKEND_SUBSTRATE_V1",
            self.backend.value,
            self.backend_source_version,
            self.backend_profile_version,
            self.storage_database_configuration_artifact,
            self.embedding_provider_family,
            self.embedding_model_identifier,
            self.embedding_revision_or_weights_digest,
            self.vector_dimension,
            self.vector_normalization,
            self.retrieval_search_configuration_artifact,
            self.reranker_cross_encoder_configuration_artifact,
            self.retrieval_k,
            self.isolation_replay_profile_version,
        )

    @property
    def sha256_digest(self) -> str:
        return sha256(self.artifact_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class RQ4EmbeddingControlContract:
    conditions: tuple[NativeMemoryLLMCondition, ...]
    invariant_fields: frozenset[str]
    provider_specific_retrieval_k_allowed: bool
    impossible_condition_is_ineligible: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "conditions", tuple(self.conditions))
        object.__setattr__(self, "invariant_fields", frozenset(self.invariant_fields))
        if self.conditions != NATIVE_MEMORY_LLM_CONDITIONS:
            raise ValueError("RQ4 provider conditions are OpenAI, Anthropic, and Google")
        required = {
            "embedding_model",
            "embedding_revision",
            "vector_dimensions",
            "vector_normalization",
            "storage",
            "retrieval_search",
            "reranker_cross_encoder",
            "retrieval_k",
        }
        if self.invariant_fields != required:
            raise ValueError("the RQ4 invariant backend substrate is frozen")
        if self.provider_specific_retrieval_k_allowed:
            raise ValueError("RQ4 provider conditions cannot change retrieval k")
        if not self.impossible_condition_is_ineligible:
            raise ValueError("a coupled provider condition must be marked ineligible")


RQ4_EMBEDDING_CONTROL = RQ4EmbeddingControlContract(
    NATIVE_MEMORY_LLM_CONDITIONS,
    frozenset(
        {
            "embedding_model",
            "embedding_revision",
            "vector_dimensions",
            "vector_normalization",
            "storage",
            "retrieval_search",
            "reranker_cross_encoder",
            "retrieval_k",
        }
    ),
    False,
    True,
)


@dataclass(frozen=True, slots=True)
class ProviderSubstrateBinding:
    provider: NativeMemoryLLMProvider
    substrate_manifest: InvariantBackendSubstrateManifest

    def __post_init__(self) -> None:
        if not isinstance(self.provider, NativeMemoryLLMProvider):
            raise TypeError("provider must be a NativeMemoryLLMProvider")
        if not isinstance(self.substrate_manifest, InvariantBackendSubstrateManifest):
            raise TypeError("substrate_manifest has the wrong type")


@dataclass(frozen=True, slots=True)
class RQ4EmbeddingControlEvidence:
    """Cross-provider proof that only native memory LLM configuration varies."""

    backend: Backend
    provider_substrates: tuple[ProviderSubstrateBinding, ...]

    def __post_init__(self) -> None:
        if self.backend not in (Backend.MEM0, Backend.GRAPHITI):
            raise ValueError("RQ4 embedding control applies only to Mem0 and Graphiti")
        values = tuple(
            sorted(tuple(self.provider_substrates), key=lambda item: item.provider.value)
        )
        if any(not isinstance(item, ProviderSubstrateBinding) for item in values):
            raise TypeError("provider_substrates has an invalid value")
        if {item.provider for item in values} != set(NativeMemoryLLMProvider):
            raise ValueError("embedding control requires all three provider families")
        if len(values) != len(NativeMemoryLLMProvider):
            raise ValueError("provider substrate bindings must be unique")
        if any(item.substrate_manifest.backend is not self.backend for item in values):
            raise ValueError("provider substrate backend must match evidence backend")
        if len({item.substrate_manifest.sha256_digest for item in values}) != 1:
            raise ValueError("RQ4 provider conditions must share one substrate")
        object.__setattr__(self, "provider_substrates", values)

    @property
    def substrate_manifest_sha256(self) -> str:
        return self.provider_substrates[0].substrate_manifest.sha256_digest


class ModelRole(StrEnum):
    MUTATION_GENERATOR = "mutation_generator"
    UNGUIDED_CONTROLLER = "unguided_controller"
    ONLINE_JUDGE = "online_judge"
    MODEL_ASSISTED_SEMANTIC_VALIDATOR = "model_assisted_semantic_validator"
    COMMON_RESPONSE_READER = "common_response_reader"
    FINAL_EVALUATOR_CFS = "final_evaluator_cfs"


class ModelTier(StrEnum):
    ONLINE_SEARCH = "online_search"
    OFFLINE_MEASUREMENT = "offline_measurement"


class DecodingPolicyFamily(StrEnum):
    SEEDED_STOCHASTIC_PREFERRED = "seeded_stochastic_preferred"
    DETERMINISTIC_GREEDY_DEFAULT_CANDIDATE = (
        "deterministic_greedy_default_candidate"
    )


@dataclass(frozen=True, slots=True)
class RolePolicy:
    role: ModelRole
    tier: ModelTier
    decoding_policy: DecodingPolicyFamily


ROLE_POLICIES = (
    RolePolicy(
        ModelRole.MUTATION_GENERATOR,
        ModelTier.ONLINE_SEARCH,
        DecodingPolicyFamily.SEEDED_STOCHASTIC_PREFERRED,
    ),
    RolePolicy(
        ModelRole.UNGUIDED_CONTROLLER,
        ModelTier.ONLINE_SEARCH,
        DecodingPolicyFamily.SEEDED_STOCHASTIC_PREFERRED,
    ),
    RolePolicy(
        ModelRole.ONLINE_JUDGE,
        ModelTier.ONLINE_SEARCH,
        DecodingPolicyFamily.DETERMINISTIC_GREEDY_DEFAULT_CANDIDATE,
    ),
    RolePolicy(
        ModelRole.MODEL_ASSISTED_SEMANTIC_VALIDATOR,
        ModelTier.ONLINE_SEARCH,
        DecodingPolicyFamily.DETERMINISTIC_GREEDY_DEFAULT_CANDIDATE,
    ),
    RolePolicy(
        ModelRole.COMMON_RESPONSE_READER,
        ModelTier.OFFLINE_MEASUREMENT,
        DecodingPolicyFamily.DETERMINISTIC_GREEDY_DEFAULT_CANDIDATE,
    ),
    RolePolicy(
        ModelRole.FINAL_EVALUATOR_CFS,
        ModelTier.OFFLINE_MEASUREMENT,
        DecodingPolicyFamily.DETERMINISTIC_GREEDY_DEFAULT_CANDIDATE,
    ),
)


@dataclass(frozen=True, slots=True)
class ModelRoleSeparationContract:
    roles: frozenset[ModelRole]
    distinct_role_identifiers: bool
    distinct_prompt_manifests: bool
    distinct_information_boundaries: bool
    distinct_output_schemas: bool
    distinct_stochastic_streams_when_applicable: bool
    shared_base_weights_allowed: bool
    shared_server_requires_semantic_equivalence: bool
    cross_role_context_or_gold_leakage_allowed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", frozenset(self.roles))
        if self.roles != frozenset(ModelRole):
            raise ValueError("every scientific model role must remain distinct")
        if not all(
            (
                self.distinct_role_identifiers,
                self.distinct_prompt_manifests,
                self.distinct_information_boundaries,
                self.distinct_output_schemas,
                self.distinct_stochastic_streams_when_applicable,
                self.shared_base_weights_allowed,
                self.shared_server_requires_semantic_equivalence,
            )
        ):
            raise ValueError("role separation and serving equivalence are mandatory")
        if self.cross_role_context_or_gold_leakage_allowed:
            raise ValueError("role sharing cannot leak context or evaluator data")


MODEL_ROLE_SEPARATION = ModelRoleSeparationContract(
    frozenset(ModelRole), True, True, True, True, True, True, True, False
)


@dataclass(frozen=True, slots=True)
class ServingTopologyContract:
    engineering_settings: frozenset[EngineeringOnlySetting]
    excluded_from_scientific_identity_when_equivalent: bool
    semantic_equivalence_required: bool
    non_equivalent_precision_or_batching_becomes_scientific_or_ineligible: bool
    online_roles_may_be_starved_by_offline_backlog: bool
    cross_role_context_cache_leakage_allowed: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "engineering_settings", frozenset(self.engineering_settings)
        )
        if self.engineering_settings != frozenset(EngineeringOnlySetting):
            raise ValueError("all frozen engineering-only settings must be represented")
        if not all(
            (
                self.excluded_from_scientific_identity_when_equivalent,
                self.semantic_equivalence_required,
                self.non_equivalent_precision_or_batching_becomes_scientific_or_ineligible,
            )
        ):
            raise ValueError("serving is engineering-only only under equivalence")
        if (
            self.online_roles_may_be_starved_by_offline_backlog
            or self.cross_role_context_cache_leakage_allowed
        ):
            raise ValueError("serving cannot starve online roles or leak role context")


SERVING_TOPOLOGY = ServingTopologyContract(
    frozenset(EngineeringOnlySetting), True, True, True, False, False
)


@dataclass(frozen=True, slots=True)
class CommonResponseReaderContract:
    backends: frozenset[Backend]
    method_ids: frozenset[str]
    one_primary_configuration: bool
    exact_query_and_resolved_public_content_only: bool
    backend_selects_primary_reader: bool
    backend_native_secondary_analysis_allowed: bool
    exact_model_id: None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "backends", frozenset(self.backends))
        object.__setattr__(self, "method_ids", frozenset(self.method_ids))
        if self.backends != frozenset(BACKENDS):
            raise ValueError("the primary reader must cover every backend")
        if self.method_ids != frozenset(method.method_id for method in ALL_METHODS):
            raise ValueError("the primary reader must cover every frozen method")
        if not all(
            (
                self.one_primary_configuration,
                self.exact_query_and_resolved_public_content_only,
                self.backend_native_secondary_analysis_allowed,
            )
        ):
            raise ValueError("one common primary response reader is required")
        if self.backend_selects_primary_reader:
            raise ValueError("backend identity cannot select the primary reader")
        if self.exact_model_id is not None:
            raise ValueError("the common reader model remains pilot-selected")


COMMON_RESPONSE_READER = CommonResponseReaderContract(
    frozenset(BACKENDS),
    frozenset(method.method_id for method in ALL_METHODS),
    True,
    True,
    False,
    True,
)


@dataclass(frozen=True, slots=True)
class StructuralCertificationContract:
    deterministic_certification_first: bool
    model_assistance_optional: bool
    independent_relation_evidence_required: bool
    generator_self_assertion_sufficient: bool
    llm_only_certification_admissible: bool

    def __post_init__(self) -> None:
        if not all(
            (
                self.deterministic_certification_first,
                self.model_assistance_optional,
                self.independent_relation_evidence_required,
            )
        ):
            raise ValueError("deterministic independent certification is required")
        if self.generator_self_assertion_sufficient or self.llm_only_certification_admissible:
            raise ValueError("an LLM assertion alone cannot certify a mutation")


STRUCTURAL_CERTIFICATION = StructuralCertificationContract(
    True, True, True, False, False
)


class EvaluatorWorkItem(StrEnum):
    RELATION_SPECIFIC_CORRECTNESS = "relation_specific_correctness_predicate"
    RETRIEVAL_CORRECTNESS = "retrieval_correctness_predicate"
    ANSWER_CORRECTNESS = "answer_correctness_predicate"
    FAILURE_SURFACE = "r_a_ra_surface_classification"
    QUERY_INTENT_CANONICALIZATION = "query_intent_canonicalization"
    MUTATION_TARGET_CANONICALIZATION = "mutation_target_canonicalization"
    ENTITY_NORMALIZATION = "entity_normalization"
    RELATION_NORMALIZATION = "relation_normalization"
    TEMPORAL_NORMALIZATION = "temporal_normalization"
    ANSWER_CONSTRAINT_NORMALIZATION = "answer_affecting_constraint_normalization"
    OPEN_ANSWER_EQUIVALENCE = "open_answer_equivalence"
    REFERENCE_VAULT = "reference_vault_construction"
    CFS_SERIALIZATION = "deterministic_cfs_serialization_dedup"


@dataclass(frozen=True, slots=True)
class EvaluatorArchitectureContract:
    work_items: frozenset[EvaluatorWorkItem]
    every_valid_execution_processed: bool
    deterministic_exact_components_first: bool
    semantic_model_only_when_exact_logic_insufficient: bool
    exact_router_may_finalize_or_route_uncertain: bool
    router_may_suppress_potential_fault_for_speed: bool
    sample_valid_executions: bool
    affects_search_trajectory: bool
    exact_model_id: None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "work_items", frozenset(self.work_items))
        if self.work_items != frozenset(EvaluatorWorkItem):
            raise ValueError("every evaluator/CFS work item is mandatory")
        if not all(
            (
                self.every_valid_execution_processed,
                self.deterministic_exact_components_first,
                self.semantic_model_only_when_exact_logic_insufficient,
                self.exact_router_may_finalize_or_route_uncertain,
            )
        ):
            raise ValueError("hybrid evaluation must preserve complete exact coverage")
        if (
            self.router_may_suppress_potential_fault_for_speed
            or self.sample_valid_executions
            or self.affects_search_trajectory
        ):
            raise ValueError("evaluation cannot sample, suppress faults, or guide search")
        if self.exact_model_id is not None:
            raise ValueError("the evaluator model remains pilot-selected")


HYBRID_EVALUATOR = EvaluatorArchitectureContract(
    frozenset(EvaluatorWorkItem), True, True, True, True, False, False, False
)


PRNG_VERSION = "UFUZZ_PRNG_V1"
PRNG_DOMAIN_SEPARATOR = b"ufuzz/evaluation/prng/v1"
PRNG_DOMAIN_MIGRATED_BEFORE_PRODUCTION = True
_PRNG_FRAME_MAGIC = b"ufuzz/canonical-fields/v1\x00"


def _canonical_fields(*values: str | bytes | int) -> bytes:
    """Encode typed fields with explicit big-endian lengths and integers."""

    encoded = bytearray(_PRNG_FRAME_MAGIC)
    encoded.extend(len(values).to_bytes(4, "big"))
    for value in values:
        if isinstance(value, str):
            tag = b"s"
            payload = value.encode("utf-8")
        elif isinstance(value, bytes):
            tag = b"b"
            payload = value
        elif isinstance(value, int) and not isinstance(value, bool):
            if value < 0 or value >= 2**64:
                raise ValueError("canonical integers must fit unsigned 64 bits")
            tag = b"u"
            payload = value.to_bytes(8, "big")
        else:
            raise TypeError("canonical fields accept only str, bytes, or unsigned int")
        encoded.extend(tag)
        encoded.extend(len(payload).to_bytes(8, "big"))
        encoded.extend(payload)
    return bytes(encoded)


def _require_sha256_hex(value: str, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return value.lower()


def derive_repetition_key(benchmark_manifest_sha256: str, repetition_index: int) -> bytes:
    """Derive a method-neutral benchmark/repetition key with HMAC-SHA-256."""

    digest = _require_sha256_hex(
        benchmark_manifest_sha256, name="benchmark_manifest_sha256"
    )
    if repetition_index not in (0, 1, 2):
        raise ValueError("repetition_index must be 0, 1, or 2")
    message = _canonical_fields(
        PRNG_VERSION,
        bytes.fromhex(digest),
        repetition_index,
    )
    return hmac.new(PRNG_DOMAIN_SEPARATOR, message, "sha256").digest()


def derive_role_event_seed(
    repetition_key: bytes,
    stream: RandomStreamName,
    scientific_event_identity: str,
    role_attempt_index: int,
) -> bytes:
    """Derive one event seed without runtime ordering or method identity."""

    if not isinstance(repetition_key, bytes) or len(repetition_key) != 32:
        raise ValueError("repetition_key must be a 32-byte HMAC-SHA-256 result")
    if not isinstance(stream, RandomStreamName):
        raise TypeError("stream must be a RandomStreamName")
    if not isinstance(scientific_event_identity, str) or not scientific_event_identity:
        raise ValueError("scientific_event_identity must be a non-empty string")
    if (
        isinstance(role_attempt_index, bool)
        or not isinstance(role_attempt_index, int)
        or role_attempt_index < 0
    ):
        raise ValueError("role_attempt_index must be a nonnegative integer")
    message = _canonical_fields(
        PRNG_VERSION,
        stream.value,
        scientific_event_identity,
        role_attempt_index,
    )
    return hmac.new(repetition_key, message, "sha256").digest()


@dataclass(frozen=True, slots=True)
class PRNGDerivationContract:
    algorithm: str
    version: str
    domain_separator: bytes
    benchmark_manifest_and_repetition_source: bool
    repetition_indices: tuple[int, ...]
    method_identity_in_repetition_key: bool
    event_keyed_role_derivation: bool
    fixed_output_mapping_vectors_required: bool

    def __post_init__(self) -> None:
        if self.algorithm != "HMAC-SHA-256" or self.version != PRNG_VERSION:
            raise ValueError("production PRNG derivation is HMAC-SHA-256 UFUZZ_PRNG_V1")
        if self.domain_separator != PRNG_DOMAIN_SEPARATOR:
            raise ValueError("the PRNG domain separator is frozen")
        if self.repetition_indices != (0, 1, 2):
            raise ValueError("the three repetition indices are frozen")
        if not all(
            (
                self.benchmark_manifest_and_repetition_source,
                self.event_keyed_role_derivation,
                self.fixed_output_mapping_vectors_required,
            )
        ):
            raise ValueError("event-keyed cross-machine derivation is mandatory")
        if self.method_identity_in_repetition_key:
            raise ValueError("benchmark repetition keys must remain method-neutral")


PRNG_DERIVATION = PRNGDerivationContract(
    "HMAC-SHA-256",
    PRNG_VERSION,
    PRNG_DOMAIN_SEPARATOR,
    True,
    (0, 1, 2),
    False,
    True,
    True,
)


@dataclass(frozen=True, slots=True, order=True)
class ManifestParameter:
    name: str
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("manifest parameter name must be non-empty")
        if not isinstance(self.value, str) or not self.value:
            raise ValueError("manifest parameter value must be non-empty")


@dataclass(frozen=True, slots=True)
class ModelRoleManifest:
    role: ModelRole
    provider_runtime_family: str
    model_identifier: str
    model_revision_or_weights_digest: str
    tokenizer_revision: str
    chat_template_revision: str
    system_prompt: bytes
    user_template: bytes
    few_shot_examples: tuple[bytes, ...]
    serialization_format_version: str
    field_order: tuple[str, ...]
    output_schema: str
    parser_version: str
    decoding_parameters: tuple[ManifestParameter, ...]
    max_output_tokens: int
    stop_rules: tuple[str, ...]
    precision_quantization: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, ModelRole):
            raise TypeError("role must be a ModelRole")
        for name in (
            "provider_runtime_family",
            "model_identifier",
            "model_revision_or_weights_digest",
            "tokenizer_revision",
            "chat_template_revision",
            "serialization_format_version",
            "output_schema",
            "parser_version",
            "precision_quantization",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("system_prompt", "user_template"):
            if not isinstance(getattr(self, name), bytes):
                raise TypeError(f"{name} must contain exact bytes")
        examples = tuple(self.few_shot_examples)
        if any(not isinstance(value, bytes) for value in examples):
            raise TypeError("few-shot examples must contain exact bytes")
        field_order = tuple(self.field_order)
        if not field_order or any(not isinstance(value, str) or not value for value in field_order):
            raise ValueError("field_order must contain non-empty field names")
        if len(set(field_order)) != len(field_order):
            raise ValueError("field_order cannot contain duplicates")
        parameters = tuple(sorted(tuple(self.decoding_parameters)))
        if any(not isinstance(value, ManifestParameter) for value in parameters):
            raise TypeError("decoding_parameters must be ManifestParameter values")
        if len({value.name for value in parameters}) != len(parameters):
            raise ValueError("decoding parameter names must be unique")
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        stop_rules = tuple(self.stop_rules)
        if any(not isinstance(value, str) for value in stop_rules):
            raise TypeError("stop rules must be strings")
        object.__setattr__(self, "few_shot_examples", examples)
        object.__setattr__(self, "field_order", field_order)
        object.__setattr__(self, "decoding_parameters", parameters)
        object.__setattr__(self, "stop_rules", stop_rules)

    @property
    def artifact_bytes(self) -> bytes:
        fields: list[str | bytes | int] = [
            "UFUZZ_MODEL_ROLE_MANIFEST_V1",
            self.role.value,
            self.provider_runtime_family,
            self.model_identifier,
            self.model_revision_or_weights_digest,
            self.tokenizer_revision,
            self.chat_template_revision,
            self.system_prompt,
            self.user_template,
            len(self.few_shot_examples),
            *self.few_shot_examples,
            self.serialization_format_version,
            len(self.field_order),
            *self.field_order,
            self.output_schema,
            self.parser_version,
            len(self.decoding_parameters),
        ]
        for parameter in self.decoding_parameters:
            fields.extend((parameter.name, parameter.value))
        fields.extend(
            (
                self.max_output_tokens,
                len(self.stop_rules),
                *self.stop_rules,
                self.precision_quantization,
            )
        )
        return _canonical_fields(*fields)

    @property
    def sha256_digest(self) -> str:
        return sha256(self.artifact_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class ConfigurationDevManifest:
    manifest_id: str
    development_item_ids: frozenset[str]
    final_root_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest_id, str) or not self.manifest_id:
            raise ValueError("manifest_id must be non-empty")
        dev = frozenset(self.development_item_ids)
        final = frozenset(self.final_root_ids)
        if not dev or not final:
            raise ValueError("development and final partitions must both be non-empty")
        if any(not isinstance(value, str) or not value for value in dev | final):
            raise ValueError("data item IDs must be non-empty strings")
        if dev & final:
            raise ValueError("configuration-development data must exclude final roots")
        object.__setattr__(self, "development_item_ids", dev)
        object.__setattr__(self, "final_root_ids", final)


class PilotMetric(StrEnum):
    PARSER_SCHEMA_SUCCESS = "parser_schema_success"
    INVALID_OUTPUT_RATE = "invalid_output_rate"
    SEMANTIC_VALIDITY_AGREEMENT = "semantic_validity_agreement"
    STRUCTURAL_CERTIFICATION_AGREEMENT = "structural_certification_agreement"
    REFERENCE_SUPPORT_TRUNCATION_RECALL = "reference_support_truncation_recall"
    REPLAY_CONSISTENCY = "replay_consistency"
    MODEL_REPEAT_CONSISTENCY = "model_repeat_consistency"
    LATENCY = "latency"
    TOKEN_COUNT = "token_count"
    GPU_MEMORY = "gpu_memory"
    THROUGHPUT = "throughput"
    BACKEND_CAPABILITY_CORRECTNESS = "backend_capability_correctness"
    INFRASTRUCTURE_FAILURE_RATE = "infrastructure_failure_rate"
    DUPLICATE_REALIZATION_RATE = "duplicate_realization_rate"
    FRONTIER_ITEM_COUNT = "frontier_item_count"
    FRONTIER_SERIALIZED_TOKENS = "frontier_serialized_tokens"
    FRONTIER_GROWTH = "frontier_growth"
    ONLINE_JUDGE_PRIORITY_RELEVANCE = "online_judge_priority_relevance"
    NORMALIZED_ANSWER_AGREEMENT = "normalized_answer_agreement"
    CONTEXT_TRUNCATION = "context_truncation"
    ROUTER_FALSE_ROUTE_RATE = "router_false_route_rate"
    BACKEND_EMPTY_TRUNCATED_RETRIEVAL_BEHAVIOR = (
        "backend_empty_truncated_retrieval_behavior"
    )
    FINITE_RANGE_VALID_OUTPUT_RATE = "finite_range_valid_output_rate"
    MALFORMED_NONANSWER_RATE = "malformed_nonanswer_rate"
    CANONICALIZATION_CONSISTENCY = "canonicalization_consistency"


class ForbiddenPilotOutcome(StrEnum):
    UF_AT_B = "UF@B"
    COV_AT_B = "Cov@B"
    FAILURE_COUNT_ADVANTAGE = "failure_count_advantage"
    UFUZZ_BASELINE_GAP = "ufuzz_minus_baseline_gap"
    CONFIGURATION_SELECTED_FOR_LARGER_UFUZZ_WIN = (
        "configuration_selected_for_larger_ufuzz_win"
    )


FORBIDDEN_PILOT_OUTCOMES = frozenset(ForbiddenPilotOutcome)


def validate_pilot_observables(values: frozenset[PilotMetric]) -> frozenset[PilotMetric]:
    result = frozenset(values)
    if any(not isinstance(value, PilotMetric) for value in result):
        raise TypeError("pilot observables must be PilotMetric values")
    return result


class ModelCapabilityClass(StrEnum):
    FAST_LOCAL_7B_8B = "fast_local_7b_8b"
    LOCAL_APPROX_14B = "local_approx_14b"
    LARGER_IF_NECESSARY = "larger_if_necessary"


@dataclass(frozen=True, slots=True)
class RetrievalDepthPilotContract:
    candidates: frozenset[int]
    requires_all_intended_native_backends: bool
    required_backends: frozenset[Backend]
    requires_configuration_dev_manifest: bool
    metrics: frozenset[PilotMetric]
    selection_rule: str
    adequacy_threshold_frozen: bool
    ambiguous_result_requires_user_approval: bool
    may_inspect_forbidden_outcomes: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", frozenset(self.candidates))
        object.__setattr__(self, "required_backends", frozenset(self.required_backends))
        object.__setattr__(self, "metrics", validate_pilot_observables(self.metrics))
        if self.candidates != frozenset({3, 5, 10, 20}):
            raise ValueError("retrieval-k candidates are exactly 3, 5, 10, and 20")
        if self.required_backends != frozenset(BACKENDS):
            raise ValueError("retrieval-k pilot requires all four native backends")
        if not all(
            (
                self.requires_all_intended_native_backends,
                self.requires_configuration_dev_manifest,
                self.ambiguous_result_requires_user_approval,
            )
        ):
            raise ValueError("k selection requires native profiles and blinded approval")
        required = {
            PilotMetric.REFERENCE_SUPPORT_TRUNCATION_RECALL,
            PilotMetric.TOKEN_COUNT,
            PilotMetric.LATENCY,
            PilotMetric.CONTEXT_TRUNCATION,
            PilotMetric.BACKEND_EMPTY_TRUNCATED_RETRIEVAL_BEHAVIOR,
        }
        if not required.issubset(self.metrics):
            raise ValueError("the retrieval-k pilot is missing required measurements")
        if self.selection_rule != "smallest_adequate_across_all_native_backends":
            raise ValueError("the retrieval-k selection rule is frozen")
        if self.adequacy_threshold_frozen or self.may_inspect_forbidden_outcomes:
            raise ValueError("no arbitrary threshold or final outcome may be introduced")


RETRIEVAL_DEPTH_PILOT = RetrievalDepthPilotContract(
    frozenset({3, 5, 10, 20}),
    True,
    frozenset(BACKENDS),
    True,
    frozenset(
        {
            PilotMetric.REFERENCE_SUPPORT_TRUNCATION_RECALL,
            PilotMetric.TOKEN_COUNT,
            PilotMetric.LATENCY,
            PilotMetric.CONTEXT_TRUNCATION,
            PilotMetric.BACKEND_EMPTY_TRUNCATED_RETRIEVAL_BEHAVIOR,
        }
    ),
    "smallest_adequate_across_all_native_backends",
    False,
    True,
    False,
)


@dataclass(frozen=True, slots=True)
class ModelPilotContract:
    role: ModelRole
    candidate_classes: tuple[ModelCapabilityClass, ...]
    metrics: frozenset[PilotMetric]
    requires_configuration_dev_manifest: bool
    prefer_smallest_meeting_quality_contract: bool
    may_inspect_forbidden_outcomes: bool
    exact_model_id: None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, ModelRole):
            raise TypeError("role must be a ModelRole")
        candidates = tuple(self.candidate_classes)
        if candidates != tuple(ModelCapabilityClass):
            raise ValueError("model pilots use the three pre-registered classes")
        object.__setattr__(self, "candidate_classes", candidates)
        object.__setattr__(self, "metrics", validate_pilot_observables(self.metrics))
        if not self.requires_configuration_dev_manifest:
            raise ValueError("model pilots require disjoint development data")
        if not self.prefer_smallest_meeting_quality_contract:
            raise ValueError("pilots prefer the smallest qualifying model")
        if self.may_inspect_forbidden_outcomes:
            raise ValueError("model pilots cannot inspect UF/Cov or method advantage")
        if self.exact_model_id is not None:
            raise ValueError("exact models remain unbound")


GENERATOR_PILOT = ModelPilotContract(
    ModelRole.MUTATION_GENERATOR,
    tuple(ModelCapabilityClass),
    frozenset(
        {
            PilotMetric.PARSER_SCHEMA_SUCCESS,
            PilotMetric.SEMANTIC_VALIDITY_AGREEMENT,
            PilotMetric.STRUCTURAL_CERTIFICATION_AGREEMENT,
            PilotMetric.DUPLICATE_REALIZATION_RATE,
            PilotMetric.TOKEN_COUNT,
            PilotMetric.LATENCY,
            PilotMetric.THROUGHPUT,
            PilotMetric.REPLAY_CONSISTENCY,
        }
    ),
    True,
    True,
    False,
)
ONLINE_JUDGE_PILOT = ModelPilotContract(
    ModelRole.ONLINE_JUDGE,
    tuple(ModelCapabilityClass),
    frozenset(
        {
            PilotMetric.PARSER_SCHEMA_SUCCESS,
            PilotMetric.FINITE_RANGE_VALID_OUTPUT_RATE,
            PilotMetric.MODEL_REPEAT_CONSISTENCY,
            PilotMetric.ONLINE_JUDGE_PRIORITY_RELEVANCE,
            PilotMetric.TOKEN_COUNT,
            PilotMetric.LATENCY,
            PilotMetric.THROUGHPUT,
        }
    ),
    True,
    True,
    False,
)
RESPONSE_READER_PILOT = ModelPilotContract(
    ModelRole.COMMON_RESPONSE_READER,
    tuple(ModelCapabilityClass),
    frozenset(
        {
            PilotMetric.NORMALIZED_ANSWER_AGREEMENT,
            PilotMetric.PARSER_SCHEMA_SUCCESS,
            PilotMetric.MALFORMED_NONANSWER_RATE,
            PilotMetric.MODEL_REPEAT_CONSISTENCY,
            PilotMetric.CONTEXT_TRUNCATION,
            PilotMetric.TOKEN_COUNT,
            PilotMetric.LATENCY,
            PilotMetric.THROUGHPUT,
        }
    ),
    True,
    True,
    False,
)
EVALUATOR_PILOT = ModelPilotContract(
    ModelRole.FINAL_EVALUATOR_CFS,
    tuple(ModelCapabilityClass),
    frozenset(
        {
            PilotMetric.SEMANTIC_VALIDITY_AGREEMENT,
            PilotMetric.MODEL_REPEAT_CONSISTENCY,
            PilotMetric.PARSER_SCHEMA_SUCCESS,
            PilotMetric.STRUCTURAL_CERTIFICATION_AGREEMENT,
            PilotMetric.ROUTER_FALSE_ROUTE_RATE,
            PilotMetric.CANONICALIZATION_CONSISTENCY,
            PilotMetric.LATENCY,
            PilotMetric.THROUGHPUT,
        }
    ),
    True,
    True,
    False,
)


@dataclass(frozen=True, slots=True)
class ControllerFeasibilityPilotContract:
    complete_frontier_required: bool
    hidden_top_n_shortlist_allowed: bool
    lossless_serialization_first: bool
    metrics: frozenset[PilotMetric]
    overflow_requires_methodology_freeze: bool
    implements_fallback_now: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", validate_pilot_observables(self.metrics))
        if not all(
            (
                self.complete_frontier_required,
                self.lossless_serialization_first,
                self.overflow_requires_methodology_freeze,
            )
        ):
            raise ValueError("the controller must receive a complete lossless frontier")
        if self.hidden_top_n_shortlist_allowed or self.implements_fallback_now:
            raise ValueError("frontier truncation and unfrozen fallback are forbidden")
        required = {
            PilotMetric.FRONTIER_ITEM_COUNT,
            PilotMetric.FRONTIER_SERIALIZED_TOKENS,
            PilotMetric.FRONTIER_GROWTH,
        }
        if not required.issubset(self.metrics):
            raise ValueError("the controller feasibility pilot is incomplete")


UNGUIDED_CONTROLLER_FEASIBILITY_PILOT = ControllerFeasibilityPilotContract(
    True,
    False,
    True,
    frozenset(
        {
            PilotMetric.FRONTIER_ITEM_COUNT,
            PilotMetric.FRONTIER_SERIALIZED_TOKENS,
            PilotMetric.FRONTIER_GROWTH,
        }
    ),
    True,
    False,
)


class RetryPilotCategory(StrEnum):
    MUTATION_REALIZATION = "mutation_realization"
    TRANSIENT_INFRASTRUCTURE = "transient_infrastructure"
    UNGUIDED_CONTROLLER = "unguided_controller"
    ONLINE_JUDGE = "online_judge"


@dataclass(frozen=True, slots=True)
class RetryCapPilotContract:
    categories: frozenset[RetryPilotCategory]
    failure_rate_criteria_only: bool
    mutation_cap_shared_across_same_realization_contract: bool
    controller_and_judge_caps_role_specific: bool
    infrastructure_cap_may_be_backend_specific: bool
    infrastructure_cap_same_across_methods_for_backend: bool
    numeric_caps: None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "categories", frozenset(self.categories))
        if self.categories != frozenset(RetryPilotCategory):
            raise ValueError("all four retry categories require measurement")
        if not all(
            (
                self.failure_rate_criteria_only,
                self.mutation_cap_shared_across_same_realization_contract,
                self.controller_and_judge_caps_role_specific,
                self.infrastructure_cap_may_be_backend_specific,
                self.infrastructure_cap_same_across_methods_for_backend,
            )
        ):
            raise ValueError("retry-cap fairness and failure-rate criteria are mandatory")
        if self.numeric_caps is not None:
            raise ValueError("numeric retry caps remain unbound")


RETRY_CAP_PILOT = RetryCapPilotContract(
    frozenset(RetryPilotCategory), True, True, True, True, True
)


@dataclass(frozen=True, slots=True)
class ProductionLoadContract:
    load_id: str
    valid_executions: int
    response_records: int
    evaluator_records: int
    wall_clock_seconds: int | None
    minimum_average_valid_executions_per_second: float | None
    reduce_evaluation_coverage_for_speed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.load_id, str) or not self.load_id:
            raise ValueError("load_id must be non-empty")
        if self.response_records != self.valid_executions or (
            self.evaluator_records != self.valid_executions
        ):
            raise ValueError("every planned valid execution needs response/evaluation")
        if self.wall_clock_seconds is None:
            if self.minimum_average_valid_executions_per_second is not None:
                raise ValueError("a load without an SLA cannot have a minimum rate")
        else:
            if self.wall_clock_seconds != 24 * 60 * 60:
                raise ValueError("the local engineering target is 24 hours")
            expected = self.valid_executions / self.wall_clock_seconds
            if self.minimum_average_valid_executions_per_second != expected:
                raise ValueError("valid-execution throughput must match the load")
        if self.reduce_evaluation_coverage_for_speed:
            raise ValueError("evaluation coverage cannot be reduced for speed")


LOCAL_RQ1_RQ3_LOAD = ProductionLoadContract(
    "local-rq1-rq3",
    1_440_000,
    1_440_000,
    1_440_000,
    ENGINEERING_TARGET.wall_clock_seconds,
    ENGINEERING_TARGET.minimum_average_valid_executions_per_second,
    False,
)
RQ4_API_LOAD = ProductionLoadContract(
    "rq4-api", 108_000, 108_000, 108_000, None, None, False
)
COMBINED_PLANNING_LOAD = ProductionLoadContract(
    "combined-planning", 1_548_000, 1_548_000, 1_548_000, None, None, False
)
# Compatibility alias for callers that mean the local six-GPU production load.
PRODUCTION_LOAD = LOCAL_RQ1_RQ3_LOAD


@dataclass(frozen=True, slots=True)
class ThroughputPilotContract:
    metrics: frozenset[str]
    requires_exact_frozen_models_and_backends: bool
    production_may_launch_when_materially_over_target: bool
    engineering_optimization_requires_semantic_equivalence: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", frozenset(self.metrics))
        required = {
            "role_calls_per_second",
            "input_output_tokens_per_second",
            "batch_fill",
            "queue_wait",
            "gpu_utilization",
            "vram",
            "p50_p95_latency",
            "backend_latency",
            "materialization_replay_depth_cost",
            "invalid_attempt_rate",
            "response_backlog_drain_rate",
            "evaluator_backlog_drain_rate",
        }
        if self.metrics != required:
            raise ValueError("the throughput pilot metric set is frozen")
        if not all(
            (
                self.requires_exact_frozen_models_and_backends,
                self.engineering_optimization_requires_semantic_equivalence,
            )
        ):
            raise ValueError("throughput work follows scientific configuration freeze")
        if self.production_may_launch_when_materially_over_target:
            raise ValueError("production cannot launch materially over the target")


THROUGHPUT_PILOT = ThroughputPilotContract(
    frozenset(
        {
            "role_calls_per_second",
            "input_output_tokens_per_second",
            "batch_fill",
            "queue_wait",
            "gpu_utilization",
            "vram",
            "p50_p95_latency",
            "backend_latency",
            "materialization_replay_depth_cost",
            "invalid_attempt_rate",
            "response_backlog_drain_rate",
            "evaluator_backlog_drain_rate",
        }
    ),
    True,
    False,
    True,
)


@dataclass(frozen=True, slots=True, order=True)
class ProductionConfigurationBinding:
    requirement: ProductionBindingRequirement
    configuration_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.requirement, ProductionBindingRequirement):
            raise TypeError("requirement must be a ProductionBindingRequirement")
        if not isinstance(self.configuration_id, str) or not self.configuration_id:
            raise ValueError("configuration_id must be non-empty")


@dataclass(frozen=True, slots=True)
class ProductionReadinessManifest:
    campaign: CampaignSpec
    backend_profile: BackendProfileStatus
    bindings: tuple[ProductionConfigurationBinding, ...]
    primary_production: bool
    opportunity_enumeration_complete: bool
    evaluator_cfs_complete: bool
    configuration_dev_manifest_frozen: bool
    serving_semantic_equivalence_proven: bool
    native_memory_llm_manifest: NativeMemoryLLMConfigurationManifest | None = None
    rq4_embedding_control_evidence: RQ4EmbeddingControlEvidence | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.campaign, CampaignSpec):
            raise TypeError("campaign must be a CampaignSpec")
        if not isinstance(self.backend_profile, BackendProfileStatus):
            raise TypeError("backend_profile must be a BackendProfileStatus")
        if self.backend_profile.backend is not self.campaign.backend:
            raise ValueError("backend profile must match the campaign backend")
        bindings = tuple(sorted(tuple(self.bindings)))
        if any(not isinstance(value, ProductionConfigurationBinding) for value in bindings):
            raise TypeError("bindings must be ProductionConfigurationBinding values")
        dimensions = [value.requirement for value in bindings]
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("production binding dimensions must be unique")
        object.__setattr__(self, "bindings", bindings)
        if self.native_memory_llm_manifest is not None and not isinstance(
            self.native_memory_llm_manifest, NativeMemoryLLMConfigurationManifest
        ):
            raise TypeError("native_memory_llm_manifest has the wrong type")
        if self.rq4_embedding_control_evidence is not None and not isinstance(
            self.rq4_embedding_control_evidence, RQ4EmbeddingControlEvidence
        ):
            raise TypeError("rq4_embedding_control_evidence has the wrong type")
        if self.campaign.native_memory_llm_condition is None and (
            self.native_memory_llm_manifest is not None
            or self.rq4_embedding_control_evidence is not None
        ):
            raise ValueError(
                "non-RQ4 campaigns cannot carry native-memory-LLM artifacts"
            )

    @property
    def method(self) -> MethodSpec:
        return self.campaign.method


@dataclass(frozen=True, slots=True)
class ProductionReadinessAssessment:
    ready: bool
    missing_bindings: frozenset[ProductionBindingRequirement]
    unexpected_bindings: frozenset[ProductionBindingRequirement]
    reasons: tuple[str, ...]


_PLACEHOLDER_MARKERS = ("placeholder", "provisional", "unbound", "todo", "tbd")


def assess_production_readiness(
    manifest: ProductionReadinessManifest,
) -> ProductionReadinessAssessment:
    """Fail-closed pure readiness assessment; it executes no backend or model."""

    if not isinstance(manifest, ProductionReadinessManifest):
        raise TypeError("manifest must be a ProductionReadinessManifest")
    required = set(required_production_bindings(manifest.campaign.method))
    condition = manifest.campaign.native_memory_llm_condition
    is_rq4 = condition is not None
    if is_rq4:
        required.add(
            ProductionBindingRequirement.NATIVE_MEMORY_LLM_CONFIGURATION
        )
    supplied = {binding.requirement for binding in manifest.bindings}
    missing = set(required) - supplied
    unexpected = supplied - set(required)
    reasons: list[str] = []
    if missing:
        reasons.append("required scientific configuration bindings are missing")
    if unexpected:
        reasons.append("method-inapplicable scientific bindings are present")
    if any(
        marker in binding.configuration_id.lower()
        for binding in manifest.bindings
        for marker in _PLACEHOLDER_MARKERS
    ):
        reasons.append("placeholder or provisional configuration ID is forbidden")
    if manifest.primary_production and not manifest.backend_profile.primary_ready:
        reasons.append("primary production requires a validated native backend profile")
    if not manifest.opportunity_enumeration_complete:
        reasons.append("opportunity enumeration/index implementation is incomplete")
    if not manifest.evaluator_cfs_complete:
        reasons.append("final evaluator/CFS implementation is unresolved")
    if not manifest.configuration_dev_manifest_frozen:
        reasons.append("configuration-development/final-data partition is unfrozen")
    if not manifest.serving_semantic_equivalence_proven:
        reasons.append("serving semantic equivalence is unproven")
    if is_rq4:
        if manifest.native_memory_llm_manifest is None:
            reasons.append("RQ4 native-memory-LLM manifest is missing")
        elif (
            manifest.native_memory_llm_manifest.provider
            is not condition.provider
        ):
            reasons.append("RQ4 provider family mismatches the planning condition")
        else:
            native_binding = next(
                (
                    item
                    for item in manifest.bindings
                    if item.requirement
                    is ProductionBindingRequirement.NATIVE_MEMORY_LLM_CONFIGURATION
                ),
                None,
            )
            if native_binding is not None and (
                native_binding.configuration_id
                != manifest.native_memory_llm_manifest.sha256_digest
            ):
                reasons.append(
                    "RQ4 native-memory-LLM binding does not match manifest digest"
                )
        evidence = manifest.rq4_embedding_control_evidence
        if evidence is None:
            reasons.append("RQ4 cross-provider substrate evidence is missing")
        elif evidence.backend is not manifest.backend_profile.backend:
            reasons.append("RQ4 substrate evidence has the wrong backend")
    elif (
        manifest.native_memory_llm_manifest is not None
        or manifest.rq4_embedding_control_evidence is not None
    ):
        reasons.append("native-memory-LLM configuration is only valid for RQ4")
    return ProductionReadinessAssessment(
        not reasons, frozenset(missing), frozenset(unexpected), tuple(reasons)
    )
