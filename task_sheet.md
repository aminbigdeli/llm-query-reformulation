# Query Reformulation Task Assignment Sheet

Implement a query reformulation system that applies extracted patterns to reformulate queries using LLMs, then evaluates the effectiveness using oracle selection (best performing reformulation per query).

## Task Assignments

#### 1. Query Reformulation Script (`src/query_reformulator.py`)

Create a comprehensive script that:

**Input Parameters:**
- `--patterns_file`: Path to extracted patterns JSON file (e.g., `consolidated_patterns_on_7310_pairs.json`)
- `--dataset_name`: Dataset identifier (`trecdl2019`, `trecdl2020`, `dev_small`, `diamond_dataset`)
- `--llm_provider`: Either `openai` or `ollama`
- `--model`: Model name (e.g., `gpt-4o`, `qwen2.5:72b`, `qwq:latest`)
- `--output_dir`: Directory to save reformulation results
- `--batch_size`: Number of queries to process in parallel (default: 10)
- `--max_reformulations_per_query`: Limit reformulations per query (default: all patterns)

**Core Functionality:**
1. **Pattern Loading**: Load patterns from JSON file
2. **Query Loading**: Load original queries from dataset
3. **Reformulation Generation**: For each query, generate reformulations using each pattern
4. **LLM Integration**: Support both OpenAI and Ollama providers
5. **Batch Processing**: Process queries in configurable batches
6. **Output Generation**: Save reformulated queries in structured format

**Output Format:**
```json
{
  "query_id": "q1",
  "original_query": "machine learning algorithms",
  "reformulations": [
    {
      "pattern_name": "Semantic Clarification",
      "pattern_id": "pattern_1",
      "reformulated_query": "machine learning algorithms for classification and regression",
      "confidence_score": 0.85
    }
  ],
  "metadata": {
    "total_patterns_applied": 15,
    "processing_time": "2024-01-15T10:30:00Z"
  }
}
```

#### 2. BM25 Run File Generator (`src/bm25_runner.py`)

Create a system to generate BM25 run files for reformulated queries:

**Input Parameters:**
- `--reformulations_file`: Path to reformulation results JSON
- `--index_path`: Path to MS MARCO Lucene index
- `--output_dir`: Directory to save BM25 run files
- `--hits`: Number of documents to retrieve per query (default: 1000)
- `--threads`: Number of parallel threads (default: 50)

**Core Functionality:**
1. **Query Processing**: Extract all reformulated queries from reformulation results
2. **BM25 Retrieval**: Generate BM25 run files for each reformulated query
3. **Batch Processing**: Process queries in parallel for efficiency
4. **Output Organization**: Save run files with clear naming convention

**Output Structure:**
```
results/bm25_runs/
├── trecdl2019/
│   ├── q1_pattern_1.run
│   ├── q1_pattern_2.run
│   └── q2_pattern_1.run
└── dev_small/
    ├── q1_pattern_1.run
    └── q2_pattern_1.run
```

**Usage:**
```bash
python src/bm25_runner.py \
  --reformulations_file results/reformulations.json \
  --index_path indexes/msmarco-passage/lucene-index-msmarco \
  --output_dir results/bm25_runs \
  --hits 1000 \
  --threads 50
```

#### 3. Oracle Computation System (`src/oracle_calculator.py`)

Create a system to calculate metrics and select the best reformulation per query:

**Input Parameters:**
- `--bm25_runs_dir`: Directory containing BM25 run files
- `--qrels_file`: Path to relevance judgments file
- `--output_file`: Path to save oracle results

**Core Functionality:**
1. **Multi-Metric Calculation**: Calculate MRR@10, NDCG@10, and Recall@1000 for each reformulation using existing calculators
2. **Oracle Selection**: Select best reformulation per query based on MRR@10 score
3. **Analysis**: Generate comprehensive effectiveness analysis

**Output Format:**
```json
{
  "oracle_results": {
    "q1": {
      "best_reformulation": {
        "pattern_name": "Semantic Clarification",
        "reformulated_query": "machine learning algorithms for classification",
        "mrr_at_10": 0.85,
        "ndcg_at_10": 0.72,
        "recall_at_1000": 0.45
      },
      "all_reformulations": [
        {
          "pattern_name": "Semantic Clarification",
          "reformulated_query": "machine learning algorithms for classification",
          "mrr_at_10": 0.85,
          "ndcg_at_10": 0.72,
          "recall_at_1000": 0.45
        },
        {
          "pattern_name": "Query Expansion",
          "reformulated_query": "machine learning algorithms supervised learning",
          "mrr_at_10": 0.78,
          "ndcg_at_10": 0.68,
          "recall_at_1000": 0.52
        }
      ]
    }
  }
}
```

#### 4. Results Analysis and Reporting (`src/results_analyzer.py`)

Create comprehensive analysis tools to process oracle file and generate results:

**Features like:**
- Pattern effectiveness analysis
- Query type analysis  
- Performance improvement visualization
- Export results to CSV/Excel
- Generate summary reports
- Comparison with baseline performance

---

## Integration Points

### Data Flow
1. Patterns + Dataset → Reformulated Queries (Task 1)
2. Reformulated Queries → BM25 Run Files (Task 2)
3. BM25 Run Files + Qrels → Oracle Computation (Task 3)
4. Oracle Results → Analysis and Reporting (Task 4)

### File Dependencies
```
results/
├── patterns/
│   └── consolidated_patterns_on_7310_pairs.json
├── reformulations/
│   ├── trecdl2019_reformulations.json
│   └── dev_small_reformulations.json
├── bm25_runs/
│   ├── trecdl2019_bm25_runs/
│   └── dev_small_bm25_runs/
└── oracle_results/
    ├── trecdl2019_oracle.json
    └── dev_small_oracle.json
```

