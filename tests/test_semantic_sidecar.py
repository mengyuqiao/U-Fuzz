from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import unittest

from ufuzz.benchmarks import LongMemEvalSLoader
from ufuzz.semantic_sidecar import (
    AcceptedAnswerComponent,
    AcceptedAnswerOption,
    AcceptedAnswerSet,
    AliasBinding,
    AliasRegistry,
    AnswerabilityStatus,
    CanonicalConstraint,
    CanonicalEntity,
    CanonicalQueryIntent,
    CompletenessScope,
    CompletenessValidation,
    ConstraintRegistry,
    EntityRegistry,
    EvaluationReferenceSeed,
    EvaluatorReferenceSidecar,
    GoldComponentSupport,
    GoldFieldStatus,
    NativeEvidenceConstraint,
    NativeEvidenceStatus,
    RelationDefinition,
    RelationRegistry,
    SearchSafeSemanticSidecar,
    SemanticArtifactIdentity,
    SourceAnchor,
    SourceCompletenessRecord,
    SourceDisposition,
    StaticArtifactManifest,
    StructuralFact,
    SupportAlternativeGroup,
    SupportRequirement,
    TemporalRegistry,
    canonical_bytes,
    canonical_sha256,
    canonicalize_temporal,
    normalize_text,
    resolve_locomo_gold_field,
    validate_evaluator_sidecar,
    validate_search_safe_information_flow,
    validate_search_safe_sidecar,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def _artifacts():
    checkpoint = next(
        LongMemEvalSLoader().load(
            FIXTURES / "longmemeval_s_minimal.json", verify_artifact=False
        )
    )
    source = checkpoint.sources[0]
    entity = CanonicalEntity("entity:user", "User")
    relation = RelationDefinition("relation:visited", "visited")
    temporal = canonicalize_temporal("time:session", "2023-01-01T10:00:00")
    constraint = CanonicalConstraint("place type", "city")
    fact = StructuralFact(
        "fact:visit-boston",
        entity.entity_id,
        relation.relation_id,
        "Boston",
        temporal.temporal_id,
        SourceAnchor.from_source(source, 0, len(source.text)),
        0,
    )
    intent = CanonicalQueryIntent(
        checkpoint.queries[0].query_id,
        entity.entity_id,
        relation.relation_id,
        temporal.temporal_id,
        (constraint,),
        AnswerabilityStatus.ANSWERABLE,
    )
    identity = SemanticArtifactIdentity(
        checkpoint.benchmark,
        checkpoint.artifact.sha256,
        checkpoint.checkpoint_id,
        "semantic-preprocessing-v1",
        "semantic-sidecar-v1",
    )
    completeness = CompletenessScope(
        (source.provenance_id,),
        ("explicit-persistent-fact",),
        (SourceCompletenessRecord(source.provenance_id, SourceDisposition.FACTS_EXTRACTED),),
        CompletenessValidation.PARTIAL,
    )
    search = SearchSafeSemanticSidecar(
        identity,
        (fact,),
        (intent,),
        EntityRegistry((entity,)),
        AliasRegistry((AliasBinding("The User", entity.entity_id),)),
        RelationRegistry((relation,)),
        TemporalRegistry((temporal,)),
        ConstraintRegistry((constraint,)),
        completeness,
    )
    component = AcceptedAnswerComponent("component:city", "Boston")
    support = SupportRequirement((SupportAlternativeGroup((fact.fact_id,)),))
    reference = EvaluationReferenceSeed(
        intent.query_id,
        AcceptedAnswerSet((AcceptedAnswerOption((component,)),)),
        NativeEvidenceConstraint(
            "longmemeval-answer-session-v1",
            (source.session_id,),
            (source.provenance_id,),
            NativeEvidenceStatus.VALID,
        ),
        (GoldComponentSupport(component.component_id, (support,)),),
        (),
        AnswerabilityStatus.ANSWERABLE,
        ("benchmark-gold", "native-session-scope"),
    )
    evaluator = EvaluatorReferenceSidecar(identity, (reference,))
    return checkpoint, search, evaluator


class NormalizationTests(unittest.TestCase):
    def test_nfkc_whitespace_and_explicit_casefold(self) -> None:
        self.assertEqual(normalize_text("  Ａlice\t Smith  "), "Alice Smith")
        self.assertEqual(
            normalize_text("  Ａlice\t Smith  ", case_insensitive=True),
            "alice smith",
        )
        self.assertEqual(normalize_text("Room #4: 10.5%"), "Room #4: 10.5%")

    def test_alias_ambiguity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ambiguous alias"):
            AliasRegistry(
                (
                    AliasBinding("Alice", "entity:alice"),
                    AliasBinding("ＡLICE", "entity:other"),
                )
            )

    def test_temporal_normalization_preserves_literal_and_never_uses_now(self) -> None:
        parsed = canonicalize_temporal("time:one", "2024-05-06 10:11:12")
        self.assertEqual(parsed.canonical_value, "2024-05-06T10:11:12")
        symbolic = canonicalize_temporal("time:two", "the following spring")
        self.assertEqual(symbolic.canonical_value, "the following spring")


class SidecarContractTests(unittest.TestCase):
    def test_search_safe_validation_and_evaluator_cross_validation(self) -> None:
        checkpoint, search, evaluator = _artifacts()
        validate_search_safe_sidecar(search, checkpoint)
        validate_evaluator_sidecar(evaluator, search)
        self.assertNotIn(b"gold", search.artifact_bytes.lower())
        self.assertIn(b"accepted_answer_set", evaluator.artifact_bytes)

    def test_source_anchor_rejects_invalid_span_or_digest(self) -> None:
        checkpoint, search, _ = _artifacts()
        fact = search.facts[0]
        bad_anchor = SourceAnchor(
            fact.source_anchor.provenance_id,
            fact.source_anchor.source_ordinal,
            fact.source_anchor.session_occurrence,
            fact.source_anchor.session_id,
            fact.source_anchor.turn_index,
            0,
            len(checkpoint.sources[0].text) + 1,
            fact.source_anchor.source_text_sha256,
        )
        bad = SearchSafeSemanticSidecar(
            search.identity,
            (StructuralFact(
                fact.fact_id,
                fact.canonical_entity,
                fact.canonical_relation,
                fact.canonical_value,
                fact.canonical_temporal_scope,
                bad_anchor,
                fact.source_order,
            ),),
            search.query_intents,
            search.entities,
            search.aliases,
            search.relations,
            search.temporals,
            search.constraints,
            search.completeness,
        )
        with self.assertRaisesRegex(ValueError, "source span"):
            validate_search_safe_sidecar(bad, checkpoint)

    def test_relation_registry_and_cross_artifact_support_fail_closed(self) -> None:
        checkpoint, search, evaluator = _artifacts()
        bad = SearchSafeSemanticSidecar(
            search.identity,
            search.facts,
            search.query_intents,
            search.entities,
            search.aliases,
            RelationRegistry((RelationDefinition("relation:other", "other"),)),
            search.temporals,
            search.constraints,
            search.completeness,
        )
        with self.assertRaisesRegex(ValueError, "unregistered relation"):
            validate_search_safe_sidecar(bad, checkpoint)
        missing_support = evaluator.references[0].component_support[0]
        bad_group = GoldComponentSupport(
            missing_support.component_id,
            (SupportRequirement((SupportAlternativeGroup(("fact:absent",)),)),),
        )
        ref = evaluator.references[0]
        bad_evaluator = EvaluatorReferenceSidecar(
            evaluator.identity,
            (EvaluationReferenceSeed(
                ref.query_id,
                ref.accepted_answer_set,
                ref.native_evidence_constraint,
                (bad_group,),
                ref.forbidden_or_superseded_fact_ids,
                ref.answerability,
                ref.reference_construction_provenance,
            ),),
        )
        with self.assertRaisesRegex(ValueError, "absent fact"):
            validate_evaluator_sidecar(bad_evaluator, search)

    def test_any_of_and_all_of_support_semantics(self) -> None:
        requirement = SupportRequirement(
            (
                SupportAlternativeGroup(("fact:a", "fact:a2")),
                SupportAlternativeGroup(("fact:b",)),
            )
        )
        self.assertTrue(requirement.satisfied_by(("fact:a2", "fact:b")))
        self.assertFalse(requirement.satisfied_by(("fact:a",)))

    def test_answerable_reference_rejects_unbounded_native_evidence(self) -> None:
        _, _, evaluator = _artifacts()
        reference = evaluator.references[0]
        with self.assertRaisesRegex(ValueError, "bounded native evidence"):
            EvaluationReferenceSeed(
                reference.query_id,
                reference.accepted_answer_set,
                NativeEvidenceConstraint(
                    "malformed-native-evidence",
                    (),
                    (),
                    NativeEvidenceStatus.MALFORMED,
                ),
                reference.component_support,
                (),
                reference.answerability,
                reference.reference_construction_provenance,
            )

    def test_completeness_cannot_be_validated_with_unresolved_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "unresolved"):
            CompletenessScope(
                ("source:1",),
                ("persistent",),
                (SourceCompletenessRecord("source:1", SourceDisposition.UNRESOLVED),),
                CompletenessValidation.VALIDATED,
            )

    def test_search_safe_information_flow_rejects_gold_and_outcomes(self) -> None:
        @dataclass(frozen=True)
        class Leaky:
            gold_answers: tuple[str, ...]
            evaluator_result: str
            uf: int

        with self.assertRaisesRegex(ValueError, "forbidden fields"):
            validate_search_safe_information_flow(Leaky(("secret",), "pass", 1))  # type: ignore[arg-type]

    def test_canonical_serialization_is_order_and_process_stable(self) -> None:
        _, search, _ = _artifacts()
        expected = search.sha256_digest
        reordered = EntityRegistry(tuple(reversed(search.entities.entities)))
        self.assertEqual(reordered, search.entities)
        script = (
            "from pathlib import Path; import sys; "
            f"sys.path.insert(0,{str(ROOT / 'src')!r}); "
            f"sys.path.insert(0,{str(ROOT / 'tests')!r}); "
            "from test_semantic_sidecar import _artifacts; "
            "print(_artifacts()[1].sha256_digest)"
        )
        observed = subprocess.check_output([sys.executable, "-c", script], text=True).strip()
        self.assertEqual(observed, expected)
        self.assertEqual(canonical_sha256(search), expected)
        self.assertEqual(canonical_bytes(search), search.artifact_bytes)

    def test_static_manifest_excludes_campaign_dimensions_and_is_stable(self) -> None:
        manifest = StaticArtifactManifest(
            "a" * 64,
            "contract-v1",
            "schema-v1",
            "normalization-v1",
            "Qwen/Qwen3-14B@revision+greedy",
            "b" * 64,
            "c" * 64,
            "native-evidence-v1",
        )
        self.assertEqual(canonical_sha256(manifest), sha256(manifest.artifact_bytes).hexdigest())
        encoded = manifest.artifact_bytes
        for forbidden in (b"backend", b"method_id", b"repetition", b"campaign_id"):
            self.assertNotIn(forbidden, encoded)

    def test_locomo_dual_gold_conflict_remains_unresolved(self) -> None:
        conflict = resolve_locomo_gold_field(
            {"category": 5, "answer": "No", "adversarial_answer": "Yes"}
        )
        self.assertIs(conflict.status, GoldFieldStatus.UNRESOLVED_CONFLICT)
        self.assertIsNone(conflict.value)
        sole = resolve_locomo_gold_field({"category": 5, "adversarial_answer": "Yes"})
        self.assertIs(sole.status, GoldFieldStatus.RESOLVED_SINGLE_FIELD)
        self.assertEqual(sole.source_field, "adversarial_answer")


if __name__ == "__main__":
    unittest.main()
