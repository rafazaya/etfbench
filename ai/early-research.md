# ETFBench Early Research: AI Evaluation Frameworks

> Research compiled November 2025, updated February 2026.
> This document consolidates framework evaluation for building ETFBench.

## Executive Summary

General-purpose benchmarks (MMLU, GSM8K) fail to capture domain-specific knowledge in finance. Public leaderboards suffer from data contamination and Goodhart's Law. ETFBench requires custom, industry-specific evaluation with evidence citation requirements.

**Decision**: Use DeepEval as primary framework for:
- Synthetic data generation from ETF documents
- Custom metrics (GEval) for citation requirements
- Pytest integration for CI/CD
- RAG pipeline evaluation

---

## 1. Benchmark Architecture

### Three Layers of Custom Benchmarking

| Layer | Purpose | Examples |
|-------|---------|----------|
| **Dataset** | Ground truth Q&A pairs + context | Curated questions, synthetic goldens, source documents |
| **Orchestration** | Evaluation infrastructure | Model loading, prompt formatting, inference execution |
| **Metric** | Success quantification | Exact match, faithfulness, citation accuracy |

### Knowledge Probing vs System Performance

- **Knowledge Probing**: "Does the model know this?" → EleutherAI Harness, LightEval
- **System Performance**: "Can the RAG pipeline retrieve and synthesize correctly?" → DeepEval, RAGAS, Inspect

ETFBench needs both: base model knowledge + RAG system evaluation.

---

## 2. Framework Comparison

### Base Model Evaluation

| Framework | Strengths | Best For |
|-----------|-----------|----------|
| **EleutherAI Harness** | Industry standard, decontamination tools, HPC support | Base model selection, knowledge probing |
| **HuggingFace LightEval** | Developer UX, caching, HF integration | Rapid iteration, custom models |
| **OpenCompass** | Scale, visual/math evaluators, leaderboards | Comprehensive benchmarking |

### RAG & Application Evaluation

| Framework | Strengths | Best For |
|-----------|-----------|----------|
| **DeepEval** | Pytest integration, synthetic data generation, 50+ metrics | Unit testing LLM apps, CI/CD |
| **RAGAS** | Reference-free evaluation, knowledge graph generation | RAG component scoring |
| **UK AISI Inspect** | Agentic evaluation, sandboxing, tool use | Agents with code execution |

### 2026 Landscape Updates

| Framework | Status (Feb 2026) |
|-----------|-------------------|
| EleutherAI Harness | v0.4.7+, added VLM support, chat templating |
| DeepEval | 50+ metrics, multimodal, red-teaming, agent evaluation |
| UK AISI Inspect | 100+ pre-built evals, broader adoption |
| Langfuse | Acquired by ClickHouse (Jan 2026) |
| Vals AI | New entrant with finance-specific benchmarks |

---

## 3. DeepEval Deep Dive

Selected for ETFBench due to synthetic generation + custom metrics.

### Key Capabilities

**Metrics**:
- Faithfulness (LLM-as-judge for hallucination detection)
- Answer Relevancy (semantic alignment with query)
- Contextual Precision/Recall (retriever evaluation)
- GEval (custom criteria with evaluation steps)

**Synthetic Data Generation**:
```python
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import EvolutionConfig, Evolution

synthesizer = Synthesizer(
    model="gpt-4.1",
    evolution_config=EvolutionConfig(
        evolutions={
            Evolution.REASONING: 0.25,
            Evolution.MULTICONTEXT: 0.20,
            Evolution.COMPARATIVE: 0.20,
        }
    )
)
goldens = synthesizer.generate_goldens_from_docs(
    document_paths=['document.pdf'],
    include_expected_output=True
)
```

**Custom Citation Metric** (GEval):
```python
from deepeval.metrics import GEval

citation_metric = GEval(
    name="EvidenceCitation",
    criteria="Answer must cite specific document, section, and quote",
    evaluation_steps=[
        "Identify citations in response",
        "Verify citations reference real documents",
        "Check cited info supports claims"
    ]
)
```

---

## 4. Domain Benchmark Case Studies

### FinanceBench (Most Relevant)

- Requires **evidence strings** (exact source sentences from 10-K filings)
- Measures **refusal rate** (models refusing to answer financial questions)
- Emphasizes **tabular data retrieval**
- Finding: Models often refuse or hallucinate on financial questions

**ETFBench should adopt**: Evidence citation requirements, refusal tracking.

### LegalBench

- Decomposes law into atomic cognitive tasks: Classification, Extraction, Rule QA
- Folder structure: each task has `tasks.py` + `helm_prompt_settings.jsonl`

**ETFBench should adopt**: Taxonomic decomposition by ETF knowledge domain.

### PubMedQA

- Includes "Maybe" as valid answer (uncertainty calibration)
- Distinguishes expert-labeled vs artificially-generated data

**ETFBench should adopt**: Difficulty tiers, expert review for synthetic data.

---

## 5. Strategic Recommendations

### Hybrid Architecture

1. **Synthetic Generation**: DeepEval Synthesizer + RAGAS for document-to-Q&A
2. **Base Model Probing**: EleutherAI Harness for multiple-choice knowledge tests
3. **RAG Evaluation**: DeepEval for unit testing pipelines
4. **Agentic** (if needed): UK AISI Inspect for tool-using agents

### ETFBench-Specific Requirements

| Requirement | Solution |
|-------------|----------|
| Evidence citations | GEval with citation criteria |
| Regulatory accuracy | Custom metric checking SEC rule references |
| Refusal tracking | Monitor model safety filter triggers |
| Document corpus | SEC EDGAR, comment letters, regulations |

### Decontamination

Use EleutherAI Harness `--check_contamination` to verify benchmark questions don't appear in training data.

---

## 6. References

### Frameworks
- [EleutherAI Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [DeepEval](https://github.com/confident-ai/deepeval)
- [RAGAS](https://docs.ragas.io/)
- [UK AISI Inspect](https://inspect.aisi.org.uk/)
- [Stanford HELM](https://crfm-helm.readthedocs.io/)

### Domain Benchmarks
- [FinanceBench](https://github.com/patronus-ai/financebench)
- [LegalBench](https://github.com/HazyResearch/legalbench)
- [Vals AI Finance](https://www.vals.ai/benchmarks)

### Papers
- [G-Eval Paper](https://arxiv.org/abs/2303.16634)
- [RAGAS Paper](https://arxiv.org/abs/2309.15217)

### Original Research Sources
See `ai/gemini-research-full.md` for the complete November 2025 research with 41 citations.
