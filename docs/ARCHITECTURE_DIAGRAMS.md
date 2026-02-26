# Architecture and UI Diagrams

This document provides presentation-ready architecture diagrams for the HealthcareOps LLM workflow.

---

## 1) End-to-End Platform Architecture

```mermaid
flowchart LR
    U[User / Analyst] --> UI[Streamlit UI]
    U --> API[FastAPI /v1/answer]

    subgraph App Layer
      UI --> ORCH[Inference Orchestrator]
      API --> ORCH
      ORCH --> NORM[Response Normalizer + Schema Validation]
    end

    subgraph Retrieval Layer
      ORCH --> RET[Hybrid Retriever]
      RET --> BM25[BM25 Sparse Search]
      RET --> DENSE[Dense Search]
      DENSE --> FAISS[(FAISS Index)]
      RET --> RERANK[Optional Cross-Encoder Reranker]
      RET --> CITE[Citation Builder]
    end

    subgraph Model Layer
      ORCH --> RUNNER[Model Runner]
      RUNNER --> BASE[HF Base Model]
      RUNNER --> ADAPT[PEFT Adapters: baseline / sft / dpo]
    end

    subgraph Ops Layer
      ORCH --> CACHE[(DiskCache)]
      API --> OBS[Metrics + Tracing]
    end

    CITE --> NORM
    NORM --> RESP[Structured JSON Response]
    RESP --> UI
    RESP --> API
```

---

## 2) Training and Alignment Pipeline (SFT + DPO)

```mermaid
flowchart TB
    D1[SFT Dataset JSONL] --> PREP1[Formatting + Curriculum Ordering]
    D2[DPO Preference Pairs JSONL] --> PREP2[Chosen vs Rejected Preparation]

    PREP1 --> SFT[SFT Trainer]
    PREP2 --> DPO[DPO Trainer]

    BASE[Base Model from HF] --> SFT
    BASE --> DPO

    SFT --> AD1[SFT Adapter Artifacts]
    DPO --> AD2[DPO Adapter Artifacts]

    AD1 --> HUB1[HF Adapter Repo: SFT]
    AD2 --> HUB2[HF Adapter Repo: DPO]

    HUB1 --> INF[Inference Variant Routing]
    HUB2 --> INF
```

---

## 3) Inference Request Lifecycle

```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit/FastAPI
    participant Orch as Orchestrator
    participant Ret as Hybrid Retriever
    participant LLM as Base+Adapter
    participant Norm as Normalizer

    User->>UI: Submit query + variant
    UI->>Orch: Create request context
    Orch->>Ret: Retrieve top-k docs
    Ret-->>Orch: Docs + scores + citations
    Orch->>Orch: Context relevance check
    Orch->>LLM: Prompt (grounded or general mode)
    LLM-->>Orch: Raw generation
    Orch->>Norm: Parse/repair/validate JSON
    Norm-->>Orch: Canonical response
    Orch-->>UI: answer, citations, tool_calls, refusal, follow_up
    UI-->>User: Render variant output + latency
```

---

## 4) Deployment Topology (Cloud-Friendly)

```mermaid
flowchart LR
    GH[(GitHub Repo)] --> ST[Streamlit Community Cloud]
    ST --> APP[app.py]
    ST --> HF[(Hugging Face Hub)]

    subgraph HF Assets
      HF --> B[Base Model Repo]
      HF --> S[SFT Adapter Repo]
      HF --> D[DPO Adapter Repo]
      HF --> E[Embedding Model Repo]
    end

    subgraph Runtime
      APP --> IDX[(FAISS/BM25 Index)]
      APP --> RAM[CPU Memory]
      APP --> NET[HF Downloads via HF_TOKEN]
    end
```

---

## 5) Streamlit UI Architecture (Look-Feel + Behavior)

```mermaid
flowchart TB
    subgraph Header
      TITLE[Project Title + Variant Compare Banner]
      MODE[Mode Toggle: Local vs API]
    end

    subgraph Controls
      CFG1[Serving Config Path]
      CFG2[RAG Config Path]
      QUERY[Question Text Area]
      RUN[Run Button]
    end

    subgraph Compare Grid
      B1[Baseline Card]
      B2[SFT Card]
      B3[DPO Card]
    end

    subgraph Response Card Template
      JSON[Formatted JSON Panel]
      META[Latency / Trace Metadata]
      ERR[Error Surface]
    end

    TITLE --> QUERY
    MODE --> CFG1
    MODE --> CFG2
    RUN --> B1
    RUN --> B2
    RUN --> B3
    B1 --> JSON
    B2 --> JSON
    B3 --> JSON
    B1 --> META
    B2 --> META
    B3 --> META
    B1 --> ERR
    B2 --> ERR
    B3 --> ERR
```

---

## 6) UI Interaction and State Flow

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Configured: Set mode/config paths
    Configured --> QueryReady: Enter question
    QueryReady --> Running: Click Run
    Running --> VariantBaseline: Execute baseline
    Running --> VariantSFT: Execute sft
    Running --> VariantDPO: Execute dpo
    VariantBaseline --> Rendered
    VariantSFT --> Rendered
    VariantDPO --> Rendered
    Rendered --> QueryReady: Edit query
    Rendered --> Running: Re-run
    Running --> Error: Exception/timeout
    Error --> QueryReady: Retry
```

---

## 7) Reliability Controls (Output Quality)

```mermaid
flowchart TD
    GEN[Raw Model Output] --> PARSE[JSON Parse Attempt]
    PARSE -->|Fail| RETRY1[Strict JSON Retry Prompt]
    PARSE -->|Pass| VALID[Schema Validation]
    RETRY1 --> VALID
    VALID -->|Fail| REPAIR[Normalizer Fallback + Canonical Fields]
    VALID -->|Pass| CHECK[Schema-like Text Check]
    CHECK -->|Detected| RETRY2[Anti-Schema Repair Retry]
    CHECK -->|Clean| DONE[Return Response]
    RETRY2 --> DONE
    REPAIR --> DONE
```

---

## 8) Notes for Presentations

- Use Diagram 1 for overall architecture.
- Use Diagram 2 for model alignment story (SFT -> DPO).
- Use Diagram 3 when explaining runtime flow.
- Use Diagram 5 + 6 for UI/UX architecture discussion.
- Use Diagram 7 to explain robustness and why outputs remain structured.

