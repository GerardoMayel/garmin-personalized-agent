# Specification 05: LangGraph Agent Graph Specification

## 1. Scope
Defines the stateful orchestration graph using LangGraph, including states, conditional edges, analytical tools, and model invocation.

## 2. Graph Definition
- **State (`AgentState`)**:
  - `user_query`: `str`
  - `date_range`: `Tuple[date, date]`
  - `retrieved_biometrics`: `Dict[str, Any]`
  - `ml_insights`: `Dict[str, Any]`
  - `rag_contexts`: `List[str]`
  - `messages`: `List[BaseMessage]`
  - `final_report`: `Optional[str]`

- **Nodes**:
  1. `QueryClassifier`: Determines whether query is purely factual, diagnostic, or longitudinal.
  2. `DataRetriever`: Fetches biometrics and time-series aggregates from local Parquet cache.
  3. `AnalyticsEngine`: Runs anomaly and causality algorithms if requested.
  4. `ClinicalRAG`: Retrieves clinical guidelines for anomalies.
  5. `PhysiologySynthesizer`: Synthesizes findings using domain-adapted LLM with Chain-of-Thought (CoT).
