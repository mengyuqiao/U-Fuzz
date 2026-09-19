"""Adapter for graphiti-core 0.30.2 public episode APIs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from importlib import metadata, util
import json
from typing import Any, Mapping, Sequence
from uuid import NAMESPACE_URL, uuid4, uuid5

from ufuzz.backends.base import (
    BackendAdapter,
    BackendCapabilities,
    Capability,
    CapabilityStatus,
    InitializationArtifact,
    OperationReceipt,
    StateHandle,
    public_dataclass,
)
from ufuzz.domain import SourceUnit
from ufuzz.mapping import FactLevelMapper, MappingStatus, RetrievalRecord
from ufuzz.structural import CertificateStatus, StructuralIndex


GRAPHITI_VERSION = "0.30.2"
GRAPHITI_TAG_COMMIT = "eaa4128681bc53487138a4bbc22d58336ebe70d2"


@dataclass(frozen=True, slots=True)
class GraphitiDeletionCertificate:
    """One-use evidence that public episode deletion is structurally atomic."""

    token: str
    state_id: str
    episode_uuid: str
    intended_fact_id: str
    provenance_ids: tuple[str, ...]
    before_digest: str
    unaffected_entries: tuple[str, ...]


class GraphitiAdapter(BackendAdapter):
    def __init__(
        self,
        *,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        graphiti: Any | None = None,
        llm_client: Any | None = None,
        embedder: Any | None = None,
        cross_encoder: Any | None = None,
    ) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._graphiti = graphiti
        self.llm_client = llm_client
        self.embedder = embedder
        self.cross_encoder = cross_encoder
        self._states: dict[str, StateHandle] = {}
        self._episode_provenance: dict[str, dict[str, set[str]]] = {}
        self._active_episodes: dict[str, set[str]] = {}
        self._deletion_certificates: dict[
            str, dict[str, GraphitiDeletionCertificate]
        ] = {}

    def capabilities(self) -> BackendCapabilities:
        available = util.find_spec("graphiti_core") is not None
        try:
            version = metadata.version("graphiti-core")
        except metadata.PackageNotFoundError:
            version = "not-installed"
        return BackendCapabilities(
            backend="graphiti",
            package="graphiti-core",
            version_or_commit=(
                f"required={GRAPHITI_VERSION}; installed={version}; "
                f"tag-commit={GRAPHITI_TAG_COMMIT}"
            ),
            available=available,
            dependencies=(f"graphiti-core=={GRAPHITI_VERSION}", "neo4j==5.26.0"),
            services=("neo4j:5.26.2", "configured LLM/embedder/reranker"),
            operations={
                "ranked_retrieval": Capability(CapabilityStatus.SUPPORTED, "public search API"),
                "provenance": Capability(
                    CapabilityStatus.SUPPORTED,
                    "EntityEdge.episodes resolves to adapter-recorded source episodes",
                ),
                "update": Capability(
                    CapabilityStatus.SUPPORTED,
                    "native append-only add_episode semantics",
                ),
                "delete": Capability(
                    CapabilityStatus.CONDITIONAL,
                    "public remove_episode only; applicable only for a faithful atomic episode target",
                ),
                "unrelated_existing_region_change": Capability(
                    CapabilityStatus.SUPPORTED,
                    "append a later episode for an existing unrelated region",
                ),
                "exact_clone": Capability(CapabilityStatus.UNSUPPORTED, "no public clone API"),
                "replay": Capability(
                    CapabilityStatus.CONDITIONAL,
                    "fresh group_id replay; extraction may be nondeterministic",
                ),
                "observable_state": Capability(
                    CapabilityStatus.SUPPORTED,
                    "public episode and edge inspection",
                ),
                "initialization_equivalence": Capability(
                    CapabilityStatus.UNRESOLVED,
                    "canonical graph projection is possible; final fingerprint is deferred",
                ),
            },
            controllable_initialization=(
                "Neo4j image and database configuration",
                "LLM client/model/decoding configuration",
                "embedder and reranker configuration",
                "group_id",
                "episode UUIDs, order, and reference times",
            ),
            notes=(
                "low-level edge deletion is deliberately not exposed",
                "Neo4j group IDs partition data but do not provide exact database snapshots",
                "live smoke tests require a dedicated Neo4j instance or dedicated database",
            ),
        )

    def _client(self) -> Any:
        if self._graphiti is not None:
            return self._graphiti
        if util.find_spec("graphiti_core") is None:
            raise RuntimeError(f"graphiti-core=={GRAPHITI_VERSION} is not installed")
        if not all((self.uri, self.user, self.password)):
            raise RuntimeError("Graphiti requires Neo4j URI, user, and password")
        from graphiti_core import Graphiti

        graphiti_kwargs = {
            "llm_client": self.llm_client,
            "embedder": self.embedder,
            "cross_encoder": self.cross_encoder,
        }
        if self.database is None:
            self._graphiti = Graphiti(
                self.uri,
                self.user,
                self.password,
                **graphiti_kwargs,
            )
        else:
            from graphiti_core.driver.neo4j_driver import Neo4jDriver

            driver = Neo4jDriver(
                self.uri,
                self.user,
                self.password,
                database=self.database,
            )
            self._graphiti = Graphiti(graph_driver=driver, **graphiti_kwargs)
        return self._graphiti

    async def create_isolated_state(
        self, artifact: InitializationArtifact
    ) -> StateHandle:
        state_id = f"ufuzz-{artifact.checkpoint_id}-{uuid4().hex}"
        state = StateHandle(
            backend="graphiti",
            state_id=state_id,
            checkpoint_id=artifact.checkpoint_id,
            initialization_digest=artifact.digest,
            backend_state=self._client(),
            metadata={
                "group_id": state_id,
                "database": self.database,
                "artifact": artifact,
            },
        )
        self._states[state_id] = state
        self._episode_provenance[state_id] = {}
        self._active_episodes[state_id] = set()
        self._deletion_certificates[state_id] = {}
        return state

    @staticmethod
    def _projection_entry(entry: Mapping[str, Any]) -> str:
        return json.dumps(entry, sort_keys=True, separators=(",", ":"), default=str)

    @classmethod
    def _projection_digest(cls, projection: Any) -> str:
        payload = json.dumps(
            projection,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    async def certify_episode_deletion(
        self,
        state: StateHandle,
        episode_uuid: str,
        intended_fact_id: str,
        structural_index: StructuralIndex,
    ) -> GraphitiDeletionCertificate:
        """Issue one-use deletion evidence from certified fact provenance.

        Certification is conservative: the episode must map to one source whose
        certified fact inventory is explicitly complete and contains only the
        intended fact. Every observable edge attributed to the episode must map
        to that fact's canonical region.
        """

        if episode_uuid not in self._active_episodes.get(state.state_id, set()):
            raise ValueError("episode is not an active adapter-tracked target")
        provenance = tuple(
            sorted(self._episode_provenance[state.state_id].get(episode_uuid, set()))
        )
        if len(provenance) != 1:
            raise ValueError("episode deletion requires exactly one source provenance")

        facts = structural_index.fact_by_id()
        intended = facts.get(intended_fact_id)
        if intended is None:
            raise ValueError("intended fact is absent from the structural index")
        if intended.provenance_id != provenance[0]:
            raise ValueError("intended fact provenance does not match the episode")
        source_facts = structural_index.facts_for_source(provenance[0])
        if tuple(fact.fact_id for fact in source_facts) != (intended_fact_id,):
            raise ValueError("episode source does not contain exactly the intended fact")

        inventory_certified = any(
            certificate.status is CertificateStatus.CERTIFIED
            and certificate.field == "source_fact_inventory"
            and certificate.provenance_ids == provenance
            and tuple(certificate.details.get("fact_ids", ())) == (intended_fact_id,)
            for certificate in structural_index.certificates.values()
        )
        if not inventory_certified:
            raise ValueError("source fact inventory is not certified complete")

        before = await self._observable_state_raw(state)
        episode_entries = [
            entry
            for entry in before
            if episode_uuid in tuple(str(value) for value in entry.get("episodes", ()))
        ]
        if not episode_entries:
            raise ValueError("no observable edge is attributable to the target episode")

        mapper = FactLevelMapper(structural_index)
        intended_region = mapper.semantic_key(intended)
        for rank, entry in enumerate(episode_entries, start=1):
            mapping = mapper.map(
                RetrievalRecord(
                    backend="graphiti",
                    local_id=f"deletion-check:{episode_uuid}:{rank}",
                    text=str(entry.get("fact") or ""),
                    rank=rank,
                    score=None,
                    provenance_ids=provenance,
                    metadata={},
                    raw=entry,
                )
            )
            if (
                mapping.status is not MappingStatus.MAPPED
                or mapping.regions != frozenset({intended_region})
            ):
                raise ValueError(
                    "observable episode content cannot be mapped solely to the intended fact"
                )

        unaffected = tuple(
            sorted(
                self._projection_entry(entry)
                for entry in before
                if episode_uuid
                not in tuple(str(value) for value in entry.get("episodes", ()))
            )
        )
        certificate = GraphitiDeletionCertificate(
            token=uuid4().hex,
            state_id=state.state_id,
            episode_uuid=episode_uuid,
            intended_fact_id=intended_fact_id,
            provenance_ids=provenance,
            before_digest=self._projection_digest(before),
            unaffected_entries=unaffected,
        )
        self._deletion_certificates[state.state_id][certificate.token] = certificate
        return certificate

    @staticmethod
    def _reference_time(source: SourceUnit) -> datetime:
        value = source.timestamp
        if value:
            for format_string in (
                "%Y/%m/%d (%a) %H:%M",
                "%I:%M %p on %d %B, %Y",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S",
            ):
                try:
                    parsed = datetime.strptime(value, format_string)
                    return parsed.replace(tzinfo=parsed.tzinfo or UTC)
                except ValueError:
                    pass
        return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=source.ordinal)

    async def ingest(
        self, state: StateHandle, sources: Sequence[SourceUnit]
    ) -> tuple[OperationReceipt, ...]:
        receipts: list[OperationReceipt] = []
        for source in sources:
            episode_uuid = str(
                uuid5(NAMESPACE_URL, f"{state.state_id}:{source.provenance_id}")
            )
            before = await self.observable_state(state)
            raw = await state.backend_state.add_episode(
                name=source.source_id,
                episode_body=source.text,
                source_description=json.dumps(
                    {"ufuzz_provenance_id": source.provenance_id},
                    sort_keys=True,
                ),
                reference_time=self._reference_time(source),
                source=self._episode_type(),
                group_id=state.metadata["group_id"],
                uuid=episode_uuid,
            )
            self._episode_provenance[state.state_id][episode_uuid] = {
                source.provenance_id
            }
            self._active_episodes[state.state_id].add(episode_uuid)
            edge_ids = tuple(str(edge.uuid) for edge in getattr(raw, "edges", ()))
            receipts.append(
                OperationReceipt(
                    backend="graphiti",
                    state_id=state.state_id,
                    operation="ingest",
                    success=True,
                    target_local_id=episode_uuid,
                    affected_local_ids=(episode_uuid, *edge_ids),
                    provenance_ids=(source.provenance_id,),
                    before=before,
                    after=await self.observable_state(state),
                    raw=public_dataclass(raw),
                )
            )
        return tuple(receipts)

    async def retrieve(
        self, state: StateHandle, query: str, top_k: int
    ) -> tuple[RetrievalRecord, ...]:
        edges = await state.backend_state.search(
            query,
            group_ids=[state.metadata["group_id"]],
            num_results=top_k,
        )
        records: list[RetrievalRecord] = []
        for rank, edge in enumerate(edges, start=1):
            provenance = {
                provenance_id
                for episode_id in getattr(edge, "episodes", ())
                if str(episode_id) in self._episode_provenance[state.state_id]
                for provenance_id in self._episode_provenance[state.state_id][
                    str(episode_id)
                ]
            }
            records.append(
                RetrievalRecord(
                    backend="graphiti",
                    local_id=str(edge.uuid),
                    text=str(getattr(edge, "fact", "")),
                    rank=rank,
                    score=None,
                    provenance_ids=tuple(sorted(provenance)),
                    metadata={
                        "episode_ids": tuple(str(v) for v in getattr(edge, "episodes", ())),
                        "group_id": getattr(edge, "group_id", None),
                    },
                    raw=public_dataclass(edge),
                )
            )
        return tuple(records)

    async def update(
        self,
        state: StateHandle,
        target_local_id: str,
        new_text: str,
        *,
        provenance_ids: Sequence[str],
        context: Mapping[str, Any] | None = None,
    ) -> OperationReceipt:
        context = dict(context or {})
        reference_time = context.get("reference_time")
        if not isinstance(reference_time, datetime):
            reference_time = datetime(2100, 1, 1, tzinfo=UTC) + timedelta(
                seconds=len(self._active_episodes[state.state_id])
            )
        episode_uuid = str(uuid4())
        before = await self._observable_state_raw(state)
        raw = await state.backend_state.add_episode(
            name=f"update:{target_local_id}",
            episode_body=new_text,
            source_description=json.dumps(
                {"ufuzz_provenance_ids": list(provenance_ids)}, sort_keys=True
            ),
            reference_time=reference_time,
            source=self._episode_type(),
            group_id=state.metadata["group_id"],
            uuid=episode_uuid,
        )
        self._episode_provenance[state.state_id][episode_uuid] = set(provenance_ids)
        self._active_episodes[state.state_id].add(episode_uuid)
        edge_ids = tuple(str(edge.uuid) for edge in getattr(raw, "edges", ()))
        return OperationReceipt(
            backend="graphiti",
            state_id=state.state_id,
            operation="update",
            success=True,
            target_local_id=target_local_id,
            affected_local_ids=(episode_uuid, *edge_ids),
            provenance_ids=tuple(provenance_ids),
            before=before,
            after=await self.observable_state(state),
            raw=public_dataclass(raw),
        )

    async def delete(
        self,
        state: StateHandle,
        target_local_id: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> OperationReceipt:
        context = dict(context or {})
        certificate = context.get("deletion_certificate")
        if not isinstance(certificate, GraphitiDeletionCertificate):
            return OperationReceipt(
                backend="graphiti",
                state_id=state.state_id,
                operation="delete",
                success=False,
                target_local_id=target_local_id,
                affected_local_ids=(),
                provenance_ids=(),
                before=None,
                after=None,
                raw=None,
                reason=(
                    "Graphiti deletion requires an adapter-issued structural certificate"
                ),
            )
        registered = self._deletion_certificates.get(state.state_id, {}).pop(
            certificate.token,
            None,
        )
        episode_uuid = certificate.episode_uuid
        if registered != certificate or certificate.state_id != state.state_id:
            return OperationReceipt(
                backend="graphiti",
                state_id=state.state_id,
                operation="delete",
                success=False,
                target_local_id=target_local_id,
                affected_local_ids=(),
                provenance_ids=(),
                before=None,
                after=None,
                raw=None,
                reason="deletion certificate is unknown, stale, or belongs to another state",
            )
        if target_local_id != episode_uuid:
            return OperationReceipt(
                backend="graphiti",
                state_id=state.state_id,
                operation="delete",
                success=False,
                target_local_id=target_local_id,
                affected_local_ids=(),
                provenance_ids=(),
                before=None,
                after=None,
                raw=None,
                reason="deletion target does not match the certified episode",
            )
        if episode_uuid not in self._active_episodes[state.state_id]:
            return OperationReceipt(
                backend="graphiti",
                state_id=state.state_id,
                operation="delete",
                success=False,
                target_local_id=target_local_id,
                affected_local_ids=(),
                provenance_ids=(),
                before=None,
                after=None,
                raw=None,
                reason="episode is not an active adapter-tracked target",
            )
        before = await self._observable_state_raw(state)
        if self._projection_digest(before) != certificate.before_digest:
            return OperationReceipt(
                backend="graphiti",
                state_id=state.state_id,
                operation="delete",
                success=False,
                target_local_id=target_local_id,
                affected_local_ids=(),
                provenance_ids=certificate.provenance_ids,
                before=before,
                after=before,
                raw=None,
                reason="observable state changed after deletion certification",
            )
        await state.backend_state.remove_episode(episode_uuid)
        self._active_episodes[state.state_id].remove(episode_uuid)
        provenance = self._episode_provenance[state.state_id].pop(episode_uuid, set())
        after = await self._observable_state_raw(state)
        after_entries = {self._projection_entry(entry) for entry in after}
        unaffected_preserved = set(certificate.unaffected_entries) <= after_entries
        target_absent = all(
            episode_uuid
            not in tuple(str(value) for value in entry.get("episodes", ()))
            for entry in after
        )
        success = unaffected_preserved and target_absent
        return OperationReceipt(
            backend="graphiti",
            state_id=state.state_id,
            operation="delete",
            success=success,
            target_local_id=episode_uuid,
            affected_local_ids=(episode_uuid,),
            provenance_ids=tuple(sorted(provenance)),
            before=before,
            after=after,
            raw={
                "episode_uuid": episode_uuid,
                "certificate_token": certificate.token,
            },
            reason=(
                None
                if success
                else "post-deletion observation could not prove preservation of unrelated entries"
            ),
        )

    async def unrelated_existing_region_change(
        self,
        state: StateHandle,
        target_local_id: str,
        new_text: str,
        *,
        provenance_ids: Sequence[str],
        context: Mapping[str, Any] | None = None,
    ) -> OperationReceipt:
        receipt = await self.update(
            state,
            target_local_id,
            new_text,
            provenance_ids=provenance_ids,
            context=context,
        )
        return replace(receipt, operation="unrelated_existing_region_change")

    async def observable_state(self, state: StateHandle) -> Any:
        raw = await self._observable_state_raw(state)
        projection = []
        for entry in raw:
            provenance = {
                provenance_id
                for episode_id in entry["episodes"]
                for provenance_id in self._episode_provenance[state.state_id].get(
                    episode_id,
                    {f"unmapped-episode:{episode_id}"},
                )
            }
            projection.append(
                {
                    **entry,
                    "episodes": sorted(provenance),
                }
            )
        return sorted(projection, key=repr)

    async def _observable_state_raw(self, state: StateHandle) -> Any:
        episode_ids = sorted(self._active_episodes[state.state_id])
        if not episode_ids:
            return []
        result = await state.backend_state.get_nodes_and_edges_by_episode(episode_ids)
        edges = [
            {
                "fact": getattr(edge, "fact", None),
                "episodes": sorted(str(v) for v in getattr(edge, "episodes", ())),
                "valid_at": str(getattr(edge, "valid_at", None)),
                "invalid_at": str(getattr(edge, "invalid_at", None)),
                "expired_at": str(getattr(edge, "expired_at", None)),
            }
            for edge in result.edges
        ]
        return sorted(edges, key=repr)

    def _episode_type(self) -> Any:
        if util.find_spec("graphiti_core") is None:
            if self._graphiti is not None:
                return "message"
            raise RuntimeError(f"graphiti-core=={GRAPHITI_VERSION} is not installed")
        from graphiti_core.nodes import EpisodeType

        return EpisodeType.message

    async def clone_state(self, state: StateHandle) -> StateHandle:
        raise NotImplementedError("Graphiti has no public exact-clone API")

    async def replay_state(self, artifact: InitializationArtifact) -> StateHandle:
        state = await self.create_isolated_state(artifact)
        await self.ingest(state, artifact.sources)
        return state

    async def reset(self, state: StateHandle) -> None:
        for episode_uuid in tuple(self._active_episodes[state.state_id]):
            await state.backend_state.remove_episode(episode_uuid)
            self._active_episodes[state.state_id].remove(episode_uuid)
            self._episode_provenance[state.state_id].pop(episode_uuid, None)
        self._deletion_certificates[state.state_id].clear()

    async def teardown(self, state: StateHandle) -> None:
        try:
            await self.reset(state)
        finally:
            self._states.pop(state.state_id, None)
            self._active_episodes.pop(state.state_id, None)
            self._episode_provenance.pop(state.state_id, None)
            self._deletion_certificates.pop(state.state_id, None)
