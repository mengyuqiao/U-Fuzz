from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from hashlib import sha256
import unittest

from ufuzz.evaluation_contract import (
    ALL_METHODS,
    BACKENDS,
    LLM_AS_JUDGE,
    RANDOM_MUTATION,
    UNGUIDED_LLM,
    Backend,
    Benchmark,
    CampaignSpec,
    NATIVE_MEMORY_LLM_CONDITIONS,
    NativeMemoryLLMProvider,
)
from ufuzz.production_config_contract import (
    AMEM_NATIVE_TARGET,
    BACKEND_FIDELITY,
    COMMON_RESPONSE_READER,
    EVALUATOR_PILOT,
    FORBIDDEN_PILOT_OUTCOMES,
    GENERATOR_PILOT,
    GRAPHITI_NATIVE_TARGET,
    HYBRID_EVALUATOR,
    MEM0_PROFILE_A,
    MEM0_PROFILE_B_TARGET,
    MEMOS_NATIVE_TARGET,
    MEMOS_GENERAL_TEXT_PROFILE_MANIFEST,
    MODEL_ROLE_SEPARATION,
    ONLINE_JUDGE_PILOT,
    PRNG_DERIVATION,
    PRNG_DOMAIN_SEPARATOR,
    PRNG_DOMAIN_MIGRATED_BEFORE_PRODUCTION,
    PRNG_VERSION,
    PRODUCTION_LOAD,
    LOCAL_RQ1_RQ3_LOAD,
    RQ4_API_LOAD,
    COMBINED_PLANNING_LOAD,
    PRIMARY_BACKEND_PROFILE_TARGETS,
    RESPONSE_READER_PILOT,
    RQ4_EMBEDDING_CONTROL,
    RQ4_FORBIDDEN_CREDENTIAL_FIELDS,
    RETRIEVAL_DEPTH_PILOT,
    RETRY_CAP_PILOT,
    ROLE_POLICIES,
    SERVING_TOPOLOGY,
    STRUCTURAL_CERTIFICATION,
    THROUGHPUT_PILOT,
    UNGUIDED_CONTROLLER_FEASIBILITY_PILOT,
    BackendProfileStatus,
    BackendProfileUse,
    BackendReadiness,
    ConfigurationDevManifest,
    DecodingPolicyFamily,
    EvaluatorWorkItem,
    ForbiddenPilotOutcome,
    ManifestParameter,
    ModelCapabilityClass,
    ModelRole,
    ModelRoleManifest,
    ModelTier,
    NativeMemoryLLMConfigurationManifest,
    InvariantBackendSubstrateManifest,
    ProviderSubstrateBinding,
    RQ4EmbeddingControlEvidence,
    PilotMetric,
    ProductionConfigurationBinding,
    ProductionReadinessManifest,
    assess_production_readiness,
    derive_repetition_key,
    derive_role_event_seed,
    validate_pilot_observables,
)
from ufuzz.scheduler_contract import (
    EngineeringOnlySetting,
    ProductionBindingRequirement,
    RandomStreamName,
    required_production_bindings,
)


class ProductionConfigurationContractTests(unittest.TestCase):
    def campaign(self, method=RANDOM_MUTATION) -> CampaignSpec:
        return CampaignSpec(Benchmark.LOCOMO, Backend.MEM0, method, 0, 8000)

    def substrate(self, **changes) -> InvariantBackendSubstrateManifest:
        value = InvariantBackendSubstrateManifest(
            Backend.MEM0,
            "mem0-source-2.0.12",
            "native-profile-b-v1",
            b'{"qdrant":"local","collection":"isolated"}',
            "huggingface",
            "exact-embedding-model",
            "embedding-revision-sha256",
            768,
            "l2-normalized",
            b'{"search":"dense","metric":"cosine"}',
            b'{"reranker":"none"}',
            10,
            "isolated-replay-v1",
        )
        return replace(value, **changes)

    def native_manifest(
        self, provider=NativeMemoryLLMProvider.OPENAI, **changes
    ) -> NativeMemoryLLMConfigurationManifest:
        value = NativeMemoryLLMConfigurationManifest(
            provider,
            "exact-model-id",
            "exact-revision",
            b'{"system":"extract memory","template":"v1"}',
            b'{"type":"object","schema":"memory-v1"}',
            "parser-v1",
            b'{"temperature":0,"max_tokens":512}',
            b'{"transport_retries":3,"output_resamples":2}',
            "api-runtime-v1",
        )
        return replace(value, **changes)

    def test_native_backend_fidelity_is_primary_and_method_neutral(self) -> None:
        self.assertEqual(BACKEND_FIDELITY.primary_backends, frozenset(BACKENDS))
        self.assertTrue(BACKEND_FIDELITY.primary_uses_intended_native_profiles)
        self.assertTrue(BACKEND_FIDELITY.same_profile_within_backend_across_methods)
        self.assertFalse(
            BACKEND_FIDELITY.cross_backend_internal_component_equality_required
        )
        self.assertFalse(BACKEND_FIDELITY.validation_profiles_may_substitute_for_primary)

    def test_mem0_profile_a_is_labeled_validation_only(self) -> None:
        self.assertIs(
            MEM0_PROFILE_A.profile_use,
            BackendProfileUse.DETERMINISTIC_VALIDATION,
        )
        self.assertIs(
            MEM0_PROFILE_A.readiness,
            BackendReadiness.VALIDATED_VALIDATION_TRACK,
        )
        self.assertFalse(MEM0_PROFILE_A.intended_native_profile)
        self.assertFalse(MEM0_PROFILE_A.primary_ready)

    def test_primary_profile_targets_include_validated_memos_profile(self) -> None:
        self.assertEqual(
            set(PRIMARY_BACKEND_PROFILE_TARGETS),
            {Backend.MEM0, Backend.AMEM, Backend.GRAPHITI, Backend.MEMOS},
        )
        self.assertIs(PRIMARY_BACKEND_PROFILE_TARGETS[Backend.MEM0], MEM0_PROFILE_B_TARGET)
        self.assertIs(PRIMARY_BACKEND_PROFILE_TARGETS[Backend.AMEM], AMEM_NATIVE_TARGET)
        self.assertIs(
            PRIMARY_BACKEND_PROFILE_TARGETS[Backend.GRAPHITI],
            GRAPHITI_NATIVE_TARGET,
        )
        self.assertIs(
            PRIMARY_BACKEND_PROFILE_TARGETS[Backend.MEMOS],
            MEMOS_NATIVE_TARGET,
        )
        self.assertTrue(MEMOS_NATIVE_TARGET.primary_ready)
        self.assertTrue(AMEM_NATIVE_TARGET.materializer_complete)
        self.assertFalse(AMEM_NATIVE_TARGET.exact_configuration_frozen)
        self.assertFalse(AMEM_NATIVE_TARGET.live_validation_complete)
        self.assertIs(
            AMEM_NATIVE_TARGET.readiness,
            BackendReadiness.REQUIRES_NATIVE_PROFILE_VALIDATION,
        )
        self.assertTrue(
            all(
                profile.profile_use is BackendProfileUse.PRIMARY_NATIVE
                and profile.intended_native_profile
                and not profile.primary_ready
                for backend, profile in PRIMARY_BACKEND_PROFILE_TARGETS.items()
                if backend is not Backend.MEMOS
            )
        )

    def test_memos_general_text_profile_manifest_is_exact_and_chat_free(self) -> None:
        manifest = MEMOS_GENERAL_TEXT_PROFILE_MANIFEST
        self.assertEqual(manifest.memos_release, "2.0.33")
        self.assertEqual(
            manifest.memos_source_commit,
            "78a372a4fc853a24d2a78efa3b4bbbd27ab9f7ad",
        )
        self.assertEqual(manifest.embedding_dimension, 768)
        self.assertEqual(
            manifest.embedding_manifest_digest,
            "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f",
        )
        self.assertFalse(manifest.native_extract_enabled)
        self.assertFalse(manifest.native_chat_enabled)
        self.assertEqual(len(manifest.artifact_sha256), 64)

    def test_common_reader_covers_every_method_and_backend(self) -> None:
        self.assertEqual(COMMON_RESPONSE_READER.backends, frozenset(BACKENDS))
        self.assertEqual(
            COMMON_RESPONSE_READER.method_ids,
            frozenset(method.method_id for method in ALL_METHODS),
        )
        self.assertTrue(COMMON_RESPONSE_READER.one_primary_configuration)
        self.assertFalse(COMMON_RESPONSE_READER.backend_selects_primary_reader)
        self.assertIsNone(COMMON_RESPONSE_READER.exact_model_id)

    def test_rq4_embedding_control_is_provider_invariant(self) -> None:
        self.assertEqual(
            RQ4_EMBEDDING_CONTROL.conditions, NATIVE_MEMORY_LLM_CONDITIONS
        )
        self.assertIn("embedding_model", RQ4_EMBEDDING_CONTROL.invariant_fields)
        self.assertIn("retrieval_k", RQ4_EMBEDDING_CONTROL.invariant_fields)
        self.assertFalse(RQ4_EMBEDDING_CONTROL.provider_specific_retrieval_k_allowed)
        self.assertTrue(RQ4_EMBEDDING_CONTROL.impossible_condition_is_ineligible)

    def test_model_roles_are_distinct_even_when_weights_may_be_shared(self) -> None:
        self.assertEqual(MODEL_ROLE_SEPARATION.roles, frozenset(ModelRole))
        self.assertTrue(MODEL_ROLE_SEPARATION.distinct_role_identifiers)
        self.assertTrue(MODEL_ROLE_SEPARATION.distinct_prompt_manifests)
        self.assertTrue(MODEL_ROLE_SEPARATION.distinct_information_boundaries)
        self.assertTrue(MODEL_ROLE_SEPARATION.shared_base_weights_allowed)
        self.assertFalse(MODEL_ROLE_SEPARATION.cross_role_context_or_gold_leakage_allowed)

    def test_two_tier_and_mixed_decoding_principles(self) -> None:
        policies = {policy.role: policy for policy in ROLE_POLICIES}
        self.assertEqual(set(policies), set(ModelRole))
        self.assertIs(
            policies[ModelRole.MUTATION_GENERATOR].tier,
            ModelTier.ONLINE_SEARCH,
        )
        self.assertIs(
            policies[ModelRole.MUTATION_GENERATOR].decoding_policy,
            DecodingPolicyFamily.SEEDED_STOCHASTIC_PREFERRED,
        )
        self.assertIs(
            policies[ModelRole.UNGUIDED_CONTROLLER].decoding_policy,
            DecodingPolicyFamily.SEEDED_STOCHASTIC_PREFERRED,
        )
        for role in (
            ModelRole.ONLINE_JUDGE,
            ModelRole.MODEL_ASSISTED_SEMANTIC_VALIDATOR,
            ModelRole.COMMON_RESPONSE_READER,
            ModelRole.FINAL_EVALUATOR_CFS,
        ):
            self.assertIs(
                policies[role].decoding_policy,
                DecodingPolicyFamily.DETERMINISTIC_GREEDY_DEFAULT_CANDIDATE,
            )
        self.assertIs(
            policies[ModelRole.COMMON_RESPONSE_READER].tier,
            ModelTier.OFFLINE_MEASUREMENT,
        )

    def test_llm_only_or_generator_self_certification_is_forbidden(self) -> None:
        self.assertTrue(STRUCTURAL_CERTIFICATION.deterministic_certification_first)
        self.assertTrue(STRUCTURAL_CERTIFICATION.independent_relation_evidence_required)
        self.assertFalse(STRUCTURAL_CERTIFICATION.generator_self_assertion_sufficient)
        self.assertFalse(STRUCTURAL_CERTIFICATION.llm_only_certification_admissible)
        with self.assertRaises(ValueError):
            replace(STRUCTURAL_CERTIFICATION, llm_only_certification_admissible=True)

    def test_hybrid_evaluator_processes_every_execution_without_sampling(self) -> None:
        self.assertEqual(HYBRID_EVALUATOR.work_items, frozenset(EvaluatorWorkItem))
        self.assertTrue(HYBRID_EVALUATOR.every_valid_execution_processed)
        self.assertTrue(HYBRID_EVALUATOR.deterministic_exact_components_first)
        self.assertTrue(HYBRID_EVALUATOR.semantic_model_only_when_exact_logic_insufficient)
        self.assertFalse(HYBRID_EVALUATOR.sample_valid_executions)
        self.assertFalse(HYBRID_EVALUATOR.router_may_suppress_potential_fault_for_speed)
        self.assertFalse(HYBRID_EVALUATOR.affects_search_trajectory)

    def test_prng_contract_is_hmac_sha256_method_neutral_and_versioned(self) -> None:
        self.assertEqual(PRNG_DERIVATION.algorithm, "HMAC-SHA-256")
        self.assertEqual(PRNG_DERIVATION.version, PRNG_VERSION)
        self.assertEqual(PRNG_DERIVATION.domain_separator, PRNG_DOMAIN_SEPARATOR)
        self.assertFalse(PRNG_DERIVATION.method_identity_in_repetition_key)
        self.assertTrue(PRNG_DOMAIN_MIGRATED_BEFORE_PRODUCTION)
        self.assertEqual(PRNG_DERIVATION.repetition_indices, (0, 1, 2))
        self.assertEqual(PRNG_DOMAIN_SEPARATOR, b"ufuzz/evaluation/prng/v1")

    def test_prng_derivation_has_fixed_cross_machine_vectors(self) -> None:
        repetition_key = derive_repetition_key("0" * 64, 1)
        self.assertEqual(
            repetition_key.hex(),
            "c392f24420fb3a572f9e7f22cf9bd21ed05c032f35837c4cb7ee011f33f6c613",
        )
        event_seed = derive_role_event_seed(
            repetition_key,
            RandomStreamName.ROOT_ORDER,
            "root-corpus-v1",
            0,
        )
        self.assertEqual(
            event_seed.hex(),
            "8a4582b87da5176e2134185f3afd958f8ec4586a5465a4829ed33116259cc1f0",
        )

    def test_prng_seed_source_rejects_invalid_manifest_and_repetition(self) -> None:
        with self.assertRaises(ValueError):
            derive_repetition_key("not-a-digest", 0)
        with self.assertRaises(ValueError):
            derive_repetition_key("0" * 64, 3)
        key = derive_repetition_key("0" * 64, 0)
        with self.assertRaises(ValueError):
            derive_role_event_seed(key, RandomStreamName.ROOT_ORDER, "event", -1)

    def _manifest(self, role: ModelRole = ModelRole.MUTATION_GENERATOR):
        return ModelRoleManifest(
            role=role,
            provider_runtime_family="local-runtime-v1",
            model_identifier="model",
            model_revision_or_weights_digest="revision",
            tokenizer_revision="tokenizer",
            chat_template_revision="chat-template",
            system_prompt=b"system\n",
            user_template=b"user: {input}\n",
            few_shot_examples=(b"example-1", b"example-2"),
            serialization_format_version="fields-v1",
            field_order=("query", "memory"),
            output_schema="schema-v1",
            parser_version="parser-v1",
            decoding_parameters=(
                ManifestParameter("temperature", "0.7"),
                ManifestParameter("top_p", "0.9"),
            ),
            max_output_tokens=256,
            stop_rules=("<end>",),
            precision_quantization="bf16",
        )

    def test_prompt_manifest_stores_exact_artifact_and_sha256_identity(self) -> None:
        manifest = self._manifest()
        self.assertIsInstance(manifest.artifact_bytes, bytes)
        self.assertEqual(len(manifest.sha256_digest), 64)
        self.assertEqual(manifest, self._manifest())
        self.assertEqual(manifest.sha256_digest, self._manifest().sha256_digest)
        changed = replace(manifest, system_prompt=b"changed")
        self.assertNotEqual(manifest.artifact_bytes, changed.artifact_bytes)
        self.assertNotEqual(manifest.sha256_digest, changed.sha256_digest)
        hash(manifest)
        with self.assertRaises(FrozenInstanceError):
            manifest.model_identifier = "other"  # type: ignore[misc]

    def test_role_identity_changes_manifest_identity(self) -> None:
        generator = self._manifest(ModelRole.MUTATION_GENERATOR)
        reader = self._manifest(ModelRole.COMMON_RESPONSE_READER)
        self.assertNotEqual(generator.artifact_bytes, reader.artifact_bytes)
        self.assertNotEqual(generator.sha256_digest, reader.sha256_digest)

    def test_pilot_final_data_firewall_requires_disjoint_nonempty_sets(self) -> None:
        manifest = ConfigurationDevManifest(
            "dev-manifest-v1", frozenset({"dev-1"}), frozenset({"final-1"})
        )
        self.assertFalse(manifest.development_item_ids & manifest.final_root_ids)
        with self.assertRaises(ValueError):
            ConfigurationDevManifest(
                "bad", frozenset({"same"}), frozenset({"same"})
            )
        with self.assertRaises(ValueError):
            ConfigurationDevManifest("bad", frozenset(), frozenset({"final"}))

    def test_only_registered_nonoutcome_pilot_metrics_are_accepted(self) -> None:
        self.assertEqual(
            validate_pilot_observables(frozenset({PilotMetric.LATENCY})),
            frozenset({PilotMetric.LATENCY}),
        )
        for outcome in FORBIDDEN_PILOT_OUTCOMES:
            self.assertIsInstance(outcome, ForbiddenPilotOutcome)
        with self.assertRaises(TypeError):
            validate_pilot_observables(  # type: ignore[arg-type]
                frozenset({ForbiddenPilotOutcome.UF_AT_B})
            )

    def test_retrieval_k_pilot_candidates_and_blinded_rule_are_exact(self) -> None:
        self.assertEqual(RETRIEVAL_DEPTH_PILOT.candidates, frozenset({3, 5, 10, 20}))
        self.assertTrue(RETRIEVAL_DEPTH_PILOT.requires_all_intended_native_backends)
        self.assertEqual(RETRIEVAL_DEPTH_PILOT.required_backends, frozenset(BACKENDS))
        self.assertTrue(RETRIEVAL_DEPTH_PILOT.requires_configuration_dev_manifest)
        self.assertEqual(
            RETRIEVAL_DEPTH_PILOT.selection_rule,
            "smallest_adequate_across_all_native_backends",
        )
        self.assertFalse(RETRIEVAL_DEPTH_PILOT.adequacy_threshold_frozen)
        self.assertFalse(RETRIEVAL_DEPTH_PILOT.may_inspect_forbidden_outcomes)
        self.assertIn(
            PilotMetric.BACKEND_EMPTY_TRUNCATED_RETRIEVAL_BEHAVIOR,
            RETRIEVAL_DEPTH_PILOT.metrics,
        )

    def test_model_pilots_share_classes_and_never_inspect_uf_or_cov(self) -> None:
        for pilot in (
            GENERATOR_PILOT,
            ONLINE_JUDGE_PILOT,
            RESPONSE_READER_PILOT,
            EVALUATOR_PILOT,
        ):
            self.assertEqual(pilot.candidate_classes, tuple(ModelCapabilityClass))
            self.assertTrue(pilot.requires_configuration_dev_manifest)
            self.assertTrue(pilot.prefer_smallest_meeting_quality_contract)
            self.assertFalse(pilot.may_inspect_forbidden_outcomes)
            self.assertIsNone(pilot.exact_model_id)

    def test_generator_pilot_measures_validity_certification_and_replay(self) -> None:
        self.assertIs(GENERATOR_PILOT.role, ModelRole.MUTATION_GENERATOR)
        self.assertTrue(
            {
                PilotMetric.PARSER_SCHEMA_SUCCESS,
                PilotMetric.SEMANTIC_VALIDITY_AGREEMENT,
                PilotMetric.STRUCTURAL_CERTIFICATION_AGREEMENT,
                PilotMetric.DUPLICATE_REALIZATION_RATE,
                PilotMetric.REPLAY_CONSISTENCY,
            }.issubset(GENERATOR_PILOT.metrics)
        )

    def test_controller_feasibility_forbids_hidden_frontier_truncation(self) -> None:
        pilot = UNGUIDED_CONTROLLER_FEASIBILITY_PILOT
        self.assertTrue(pilot.complete_frontier_required)
        self.assertTrue(pilot.lossless_serialization_first)
        self.assertFalse(pilot.hidden_top_n_shortlist_allowed)
        self.assertTrue(pilot.overflow_requires_methodology_freeze)
        self.assertFalse(pilot.implements_fallback_now)

    def test_online_judge_pilot_is_stability_not_outcome_driven(self) -> None:
        self.assertIs(ONLINE_JUDGE_PILOT.role, ModelRole.ONLINE_JUDGE)
        self.assertTrue(
            {
                PilotMetric.PARSER_SCHEMA_SUCCESS,
                PilotMetric.FINITE_RANGE_VALID_OUTPUT_RATE,
                PilotMetric.MODEL_REPEAT_CONSISTENCY,
                PilotMetric.ONLINE_JUDGE_PRIORITY_RELEVANCE,
            }.issubset(ONLINE_JUDGE_PILOT.metrics)
        )

    def test_common_reader_pilot_is_shared_and_answer_audited(self) -> None:
        self.assertIs(
            RESPONSE_READER_PILOT.role, ModelRole.COMMON_RESPONSE_READER
        )
        self.assertIn(
            PilotMetric.NORMALIZED_ANSWER_AGREEMENT,
            RESPONSE_READER_PILOT.metrics,
        )
        self.assertIn(
            PilotMetric.MALFORMED_NONANSWER_RATE,
            RESPONSE_READER_PILOT.metrics,
        )
        self.assertIn(PilotMetric.CONTEXT_TRUNCATION, RESPONSE_READER_PILOT.metrics)

    def test_evaluator_pilot_requires_specification_and_blinded_consistency_metrics(self) -> None:
        self.assertIs(EVALUATOR_PILOT.role, ModelRole.FINAL_EVALUATOR_CFS)
        self.assertIn(PilotMetric.ROUTER_FALSE_ROUTE_RATE, EVALUATOR_PILOT.metrics)
        self.assertIn(
            PilotMetric.CANONICALIZATION_CONSISTENCY,
            EVALUATOR_PILOT.metrics,
        )
        self.assertIn(PilotMetric.MODEL_REPEAT_CONSISTENCY, EVALUATOR_PILOT.metrics)
        self.assertEqual(HYBRID_EVALUATOR.work_items, frozenset(EvaluatorWorkItem))

    def test_retry_cap_pilot_is_failure_rate_based_and_caps_remain_unbound(self) -> None:
        self.assertTrue(RETRY_CAP_PILOT.failure_rate_criteria_only)
        self.assertTrue(
            RETRY_CAP_PILOT.mutation_cap_shared_across_same_realization_contract
        )
        self.assertTrue(RETRY_CAP_PILOT.controller_and_judge_caps_role_specific)
        self.assertTrue(RETRY_CAP_PILOT.infrastructure_cap_may_be_backend_specific)
        self.assertTrue(
            RETRY_CAP_PILOT.infrastructure_cap_same_across_methods_for_backend
        )
        self.assertIsNone(RETRY_CAP_PILOT.numeric_caps)

    def test_split_production_loads_require_complete_response_and_evaluation(self) -> None:
        self.assertIs(PRODUCTION_LOAD, LOCAL_RQ1_RQ3_LOAD)
        self.assertEqual(LOCAL_RQ1_RQ3_LOAD.valid_executions, 1_440_000)
        self.assertEqual(LOCAL_RQ1_RQ3_LOAD.response_records, 1_440_000)
        self.assertEqual(LOCAL_RQ1_RQ3_LOAD.evaluator_records, 1_440_000)
        self.assertEqual(
            LOCAL_RQ1_RQ3_LOAD.minimum_average_valid_executions_per_second,
            1_440_000 / 86400,
        )
        self.assertEqual(RQ4_API_LOAD.valid_executions, 108_000)
        self.assertIsNone(RQ4_API_LOAD.wall_clock_seconds)
        self.assertIsNone(RQ4_API_LOAD.minimum_average_valid_executions_per_second)
        self.assertEqual(COMBINED_PLANNING_LOAD.valid_executions, 1_548_000)
        self.assertFalse(LOCAL_RQ1_RQ3_LOAD.reduce_evaluation_coverage_for_speed)

    def test_throughput_pilot_follows_model_and_backend_freeze(self) -> None:
        self.assertTrue(THROUGHPUT_PILOT.requires_exact_frozen_models_and_backends)
        self.assertFalse(THROUGHPUT_PILOT.production_may_launch_when_materially_over_target)
        self.assertTrue(
            THROUGHPUT_PILOT.engineering_optimization_requires_semantic_equivalence
        )
        self.assertIn("response_backlog_drain_rate", THROUGHPUT_PILOT.metrics)
        self.assertIn("evaluator_backlog_drain_rate", THROUGHPUT_PILOT.metrics)

    def test_equivalent_serving_topology_is_engineering_only(self) -> None:
        self.assertEqual(
            SERVING_TOPOLOGY.engineering_settings,
            frozenset(EngineeringOnlySetting),
        )
        self.assertTrue(
            SERVING_TOPOLOGY.excluded_from_scientific_identity_when_equivalent
        )
        self.assertTrue(SERVING_TOPOLOGY.semantic_equivalence_required)
        self.assertFalse(SERVING_TOPOLOGY.online_roles_may_be_starved_by_offline_backlog)
        self.assertFalse(SERVING_TOPOLOGY.cross_role_context_cache_leakage_allowed)

    def _bindings_for(self, method):
        return tuple(
            ProductionConfigurationBinding(requirement, f"frozen:{requirement.value}:v1")
            for requirement in required_production_bindings(method)
        )

    def test_primary_readiness_rejects_profile_a_even_with_all_bindings(self) -> None:
        manifest = ProductionReadinessManifest(
            self.campaign(),
            MEM0_PROFILE_A,
            self._bindings_for(RANDOM_MUTATION),
            True,
            True,
            True,
            True,
            True,
        )
        result = assess_production_readiness(manifest)
        self.assertFalse(result.ready)
        self.assertFalse(result.missing_bindings)
        self.assertIn(
            "primary production requires a validated native backend profile",
            result.reasons,
        )

    def test_ready_native_profile_and_complete_bindings_can_pass_pure_assessment(self) -> None:
        ready_profile = BackendProfileStatus(
            Backend.MEM0,
            "mem0-native-profile-b-v1",
            BackendProfileUse.PRIMARY_NATIVE,
            BackendReadiness.PRODUCTION_READY,
            True,
            True,
            True,
            True,
        )
        manifest = ProductionReadinessManifest(
            self.campaign(),
            ready_profile,
            self._bindings_for(RANDOM_MUTATION),
            True,
            True,
            True,
            True,
            True,
        )
        result = assess_production_readiness(manifest)
        self.assertTrue(result.ready)
        self.assertFalse(result.missing_bindings)
        self.assertFalse(result.unexpected_bindings)
        self.assertFalse(result.reasons)

    def test_readiness_requires_method_specific_controller_and_judge_bindings(self) -> None:
        ready_profile = BackendProfileStatus(
            Backend.MEM0,
            "mem0-native-profile-b-v1",
            BackendProfileUse.PRIMARY_NATIVE,
            BackendReadiness.PRODUCTION_READY,
            True,
            True,
            True,
            True,
        )
        for method, requirements in (
            (
                UNGUIDED_LLM,
                {
                    ProductionBindingRequirement.UNGUIDED_CONTROLLER,
                    ProductionBindingRequirement.UNGUIDED_CONTROLLER_OUTPUT_RESAMPLING_CAP,
                },
            ),
            (
                LLM_AS_JUDGE,
                {
                    ProductionBindingRequirement.ONLINE_JUDGE,
                    ProductionBindingRequirement.ONLINE_JUDGE_OUTPUT_RESAMPLING_CAP,
                },
            ),
        ):
            complete = self._bindings_for(method)
            self.assertTrue(requirements.issubset({item.requirement for item in complete}))
            incomplete = tuple(item for item in complete if item.requirement not in requirements)
            result = assess_production_readiness(
                ProductionReadinessManifest(
                    self.campaign(method),
                    ready_profile,
                    incomplete,
                    True,
                    True,
                    True,
                    True,
                    True,
                )
            )
            self.assertFalse(result.ready)
            self.assertEqual(result.missing_bindings, frozenset(requirements))

    def test_readiness_rejects_placeholders_and_unresolved_evaluator(self) -> None:
        ready_profile = BackendProfileStatus(
            Backend.MEM0,
            "mem0-native-profile-b-v1",
            BackendProfileUse.PRIMARY_NATIVE,
            BackendReadiness.PRODUCTION_READY,
            True,
            True,
            True,
            True,
        )
        bindings = list(self._bindings_for(RANDOM_MUTATION))
        bindings[0] = replace(bindings[0], configuration_id="provisional-value")
        result = assess_production_readiness(
            ProductionReadinessManifest(
                self.campaign(),
                ready_profile,
                tuple(bindings),
                True,
                True,
                False,
                True,
                True,
            )
        )
        self.assertFalse(result.ready)
        self.assertTrue(any("placeholder" in reason for reason in result.reasons))
        self.assertTrue(any("evaluator" in reason for reason in result.reasons))

    def test_rq4_readiness_requires_causal_binding_manifest_and_embedding_proof(self) -> None:
        ready_profile = BackendProfileStatus(
            Backend.MEM0,
            "mem0-native-rq4-v1",
            BackendProfileUse.PRIMARY_NATIVE,
            BackendReadiness.PRODUCTION_READY,
            True,
            True,
            True,
            True,
        )
        condition = NATIVE_MEMORY_LLM_CONDITIONS[0]
        campaign = CampaignSpec(
            Benchmark.LOCOMO,
            Backend.MEM0,
            RANDOM_MUTATION,
            0,
            2000,
            condition,
        )
        native_manifest = self.native_manifest()
        bindings = self._bindings_for(RANDOM_MUTATION) + (
            ProductionConfigurationBinding(
                ProductionBindingRequirement.NATIVE_MEMORY_LLM_CONFIGURATION,
                native_manifest.sha256_digest,
            ),
        )
        substrate = self.substrate()
        substrate_evidence = RQ4EmbeddingControlEvidence(
            Backend.MEM0,
            tuple(
                ProviderSubstrateBinding(provider, substrate)
                for provider in NativeMemoryLLMProvider
            ),
        )
        result = assess_production_readiness(
            ProductionReadinessManifest(
                campaign,
                ready_profile,
                bindings,
                True,
                True,
                True,
                True,
                True,
                native_manifest,
                substrate_evidence,
            )
        )
        self.assertTrue(result.ready, result.reasons)

    def test_readiness_is_bound_to_campaign_without_duplicate_rq_fields(self) -> None:
        field_names = {item.name for item in fields(ProductionReadinessManifest)}
        self.assertIn("campaign", field_names)
        self.assertFalse(
            field_names & {"method", "backend", "native_memory_llm_condition", "is_rq4"}
        )
        with self.assertRaises(ValueError):
            CampaignSpec(
                Benchmark.LOCOMO,
                Backend.MEM0,
                RANDOM_MUTATION,
                0,
                2000,
            )

    def test_native_memory_llm_manifest_is_exact_self_contained_and_secret_free(self) -> None:
        manifest = self.native_manifest()
        self.assertEqual(len(manifest.sha256_digest), 64)
        self.assertEqual(
            manifest.sha256_digest,
            sha256(manifest.artifact_bytes).hexdigest(),
        )
        self.assertFalse(
            RQ4_FORBIDDEN_CREDENTIAL_FIELDS
            & {item.name.casefold() for item in fields(manifest)}
        )
        variants = (
            replace(manifest, provider=NativeMemoryLLMProvider.ANTHROPIC),
            replace(manifest, model_identifier="other-model"),
            replace(manifest, model_revision="other-revision"),
            replace(manifest, memory_system_prompt_config_artifact=b"other-prompt"),
            replace(manifest, structured_output_schema_artifact=b"other-schema"),
            replace(manifest, parser_version="other-parser"),
            replace(manifest, decoding_parameters_artifact=b"other-decoding"),
            replace(manifest, retry_resampling_policy_artifact=b"other-retry"),
            replace(manifest, api_runtime_version="other-runtime"),
        )
        self.assertTrue(
            all(item.sha256_digest != manifest.sha256_digest for item in variants)
        )

    def test_invariant_substrate_digest_covers_every_causal_component(self) -> None:
        substrate = self.substrate()
        variants = (
            replace(substrate, backend_source_version="other-source"),
            replace(substrate, backend_profile_version="other-profile"),
            replace(substrate, storage_database_configuration_artifact=b"other-storage"),
            replace(substrate, embedding_provider_family="other-provider"),
            replace(substrate, embedding_model_identifier="other-model"),
            replace(substrate, embedding_revision_or_weights_digest="other-revision"),
            replace(substrate, vector_dimension=1024),
            replace(substrate, vector_normalization="unit-normalized"),
            replace(substrate, retrieval_search_configuration_artifact=b"other-search"),
            replace(substrate, reranker_cross_encoder_configuration_artifact=b"other-reranker"),
            replace(substrate, retrieval_k=20),
            replace(substrate, isolation_replay_profile_version="other-isolation"),
        )
        self.assertEqual(
            substrate.sha256_digest,
            sha256(substrate.artifact_bytes).hexdigest(),
        )
        self.assertTrue(
            all(item.sha256_digest != substrate.sha256_digest for item in variants)
        )

    def test_rq4_readiness_fails_closed_for_each_missing_or_mismatched_proof(self) -> None:
        condition = NATIVE_MEMORY_LLM_CONDITIONS[0]
        campaign = CampaignSpec(
            Benchmark.LOCOMO, Backend.MEM0, RANDOM_MUTATION, 0, 2000, condition
        )
        profile = BackendProfileStatus(
            Backend.MEM0,
            "mem0-native-rq4-v1",
            BackendProfileUse.PRIMARY_NATIVE,
            BackendReadiness.PRODUCTION_READY,
            True,
            True,
            True,
            True,
        )
        native = self.native_manifest()
        substrate = self.substrate()
        evidence = RQ4EmbeddingControlEvidence(
            Backend.MEM0,
            tuple(
                ProviderSubstrateBinding(provider, substrate)
                for provider in NativeMemoryLLMProvider
            ),
        )
        native_binding = ProductionConfigurationBinding(
            ProductionBindingRequirement.NATIVE_MEMORY_LLM_CONFIGURATION,
            native.sha256_digest,
        )
        base = ProductionReadinessManifest(
            campaign,
            profile,
            self._bindings_for(RANDOM_MUTATION) + (native_binding,),
            True,
            True,
            True,
            True,
            True,
            native,
            evidence,
        )
        self.assertTrue(assess_production_readiness(base).ready)
        cases = (
            replace(base, native_memory_llm_manifest=None),
            replace(base, rq4_embedding_control_evidence=None),
            replace(
                base,
                bindings=tuple(
                    item
                    for item in base.bindings
                    if item.requirement
                    is not ProductionBindingRequirement.NATIVE_MEMORY_LLM_CONFIGURATION
                ),
            ),
            replace(
                base,
                native_memory_llm_manifest=self.native_manifest(
                    NativeMemoryLLMProvider.ANTHROPIC
                ),
            ),
            replace(
                base,
                rq4_embedding_control_evidence=RQ4EmbeddingControlEvidence(
                    Backend.GRAPHITI,
                    tuple(
                        ProviderSubstrateBinding(
                            provider,
                            replace(substrate, backend=Backend.GRAPHITI),
                        )
                        for provider in NativeMemoryLLMProvider
                    ),
                ),
            ),
        )
        for manifest in cases:
            with self.subTest(manifest=manifest):
                self.assertFalse(assess_production_readiness(manifest).ready)
        with self.assertRaisesRegex(ValueError, "share one substrate"):
            RQ4EmbeddingControlEvidence(
                Backend.MEM0,
                (
                    ProviderSubstrateBinding(NativeMemoryLLMProvider.OPENAI, substrate),
                    ProviderSubstrateBinding(NativeMemoryLLMProvider.ANTHROPIC, substrate),
                    ProviderSubstrateBinding(
                        NativeMemoryLLMProvider.GOOGLE,
                        replace(substrate, embedding_model_identifier="other-model"),
                    ),
                ),
            )

    def test_non_rq4_campaign_rejects_rq4_only_manifest_and_evidence(self) -> None:
        profile = BackendProfileStatus(
            Backend.MEM0,
            "mem0-native-profile-b-v1",
            BackendProfileUse.PRIMARY_NATIVE,
            BackendReadiness.PRODUCTION_READY,
            True,
            True,
            True,
            True,
        )
        substrate = self.substrate()
        evidence = RQ4EmbeddingControlEvidence(
            Backend.MEM0,
            tuple(
                ProviderSubstrateBinding(provider, substrate)
                for provider in NativeMemoryLLMProvider
            ),
        )
        with self.assertRaisesRegex(ValueError, "non-RQ4"):
            ProductionReadinessManifest(
                self.campaign(),
                profile,
                self._bindings_for(RANDOM_MUTATION),
                True,
                True,
                True,
                True,
                True,
                self.native_manifest(),
                evidence,
            )


if __name__ == "__main__":
    unittest.main()
