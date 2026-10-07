# Architecture

## Scope

This repository starts with a deterministic local baseline so later agentic behavior can be compared against a stable control.

## Functional requirements

1. Ingest supported text documents with immutable checksum identity.
2. Chunk documents deterministically.
3. Retrieve relevant chunks with bounded top-k.
4. Return grounded answers with citation identifiers.
5. Expose health, readiness and version API endpoints.

## Non-functional requirements

Provider-independent local execution, explicit configuration, bounded work, testability, structured errors and evidence-driven claims.

## Non-goals for M1

No claim of production Qdrant/PostgreSQL persistence, semantic embeddings, reranking, LangGraph orchestration, MCP transport or OpenTelemetry completeness. Those are later milestone deliverables.

## Threat model

Documents and retrieved chunks are untrusted data. The system must not treat document text as system instructions. Inputs are size/type validated, filesystem paths are not accepted by the HTTP ingestion endpoint, and unsupported content is rejected.

## Acceptance criteria

G1 architecture documented; G2 reproducible environment; G3 deterministic ingestion; G4 retrieval functional; G7 baseline RAG functional; G8 citation objects returned; tests green. G5/G6/G9-G20 remain pending until implemented and verified.
