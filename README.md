# LLM Query Reformulation

Our project focuses on understanding how queries can be reformulated to improve retrieval effectiveness in information retrieval systems. It uses LLMs to automatically identify a consolidated list of patterns in query reformulation pairs and applies these patterns to new unseen queries to enhance their retrieval effectiveness.

## Project Structure

```
llm_query_reformulation/
├── data/                           # Dataset directory
│   ├── chameleons/                # Chameleons dataset (Lesser, Pygmy, Veiled). These are hard queries that mainly receive MRR@10 of 0.
│   │   ├── Lesser(common6)/       # Lesser chameleon dataset variants
│   │   │   ├── Lesser_common6_0.1
│   │   │   ├── Lesser_common6_0.2
│   │   │   ├── Lesser_common6_0.3
│   │   │   ├── Lesser_common6_0.4
│   │   │   └── Lesser_common6_0.5
│   │   ├── Pygmy(common5)/        # Pygmy chameleon dataset variants
│   │   │   ├── Pygmy_common5_0.1
│   │   │   ├── Pygmy_common5_0.2
│   │   │   ├── Pygmy_common5_0.3
│   │   │   ├── Pygmy_common5_0.4
│   │   │   └── Pygmy_common5_0.5
│   │   └── Veiled(common4)/       # Veiled chameleon dataset variants
│   │       ├── Veiled_common4_0.1
│   │       ├── Veiled_common4_0.2
│   │       ├── Veiled_common4_0.3
│   │       ├── Veiled_common4_0.4
│   │       └── Veiled_common4_0.5
│   ├── dev_small/                 # Small development dataset of MS MARCO
│   │   ├── queries.dev.small.tsv
│   │   └── qrels.dev.small.tsv    # Relevance judgments
│   ├── diamond_dataset/           # Main diamond dataset from Matches Made in Heaven Dataset
│   │   ├── diamond_dataset.tsv
│   │   └── diamond_dataset_test_235631.tsv
│   ├── trecdl2019/               # TREC DL 2019 dataset
│   │   ├── original_queries.tsv
│   │   ├── bm25_run_original_queries.tsv
│   │   └── qrels.trec            # Relevance judgments
│   └── trecdl2020/               # TREC DL 2020 dataset
│       ├── original_queries.tsv
│       ├── bm25_run_original_queries.tsv
│       └── qrels.trec            # Relevance judgments
├── results/                       # Experiment results
│   └── consolied_reformulation_patterns_qwen2.5:72b/  # experiment results
│       ├── consolidated_patterns_on_7310_pairs.json
│       ├── extracted_patterns_from_1000_pairs.json
│       └── extracted_patterns_from_4200_pairs.json
├── src/                          # Source code
│   ├── iterative_pattern_extraction.py    # Main pattern extraction script
│   ├── query_reformulation_prompts.py     # Prompt templates
│   └── query_reformulation_all_prompts.py # Extended version of all prompt variations templates
├── MRR_calculator.py             # Mean Reciprocal Rank calculator
├── NDCG_calculator.py            # Normalized Discounted Cumulative Gain calculator
└── Recall_calculator.py          # Recall@K calculator
```

You can read more Chamaeonls paper and its datasets [HERE](https://www.academia.edu/download/75533636/3459637.pdf).
You can also read more about Matches Made in Heaven dataset [HERE](https://www.academia.edu/download/79261751/3459637.pdf).

## Key Components

### 1. Pattern Extraction (`src/iterative_pattern_extraction.py`)

The main script that performs iterative pattern extraction from query reformulation pairs. It supports:

- **Multiple LLM Providers**: OpenAI GPT models and Ollama local models
- **Iterative Learning**: Processes query pairs in batches and consolidates patterns
- **Pattern Consolidation**: Maintains a fixed-size set of high-quality patterns
- **Comprehensive Output**: Generates patterns, individual mappings, and detailed analysis

**Usage:**
```bash
cd src
python iterative_pattern_extraction.py
```

**Configuration:**
- Modify `llm_provider` variable to switch between "openai" and "ollama"
- Adjust `model`, `batch_size`, `max_patterns`, and `sample_size` parameters
- Set `OPENAI_API_KEY` environment variable for OpenAI models

### 2. Prompt Templates (`src/query_reformulation_prompts.py`)

Contains various prompt templates for different aspects of query reformulation:

- **Consolidated Pattern Prompts**: Maintains a fixed-size set of patterns, merging and refining as new evidence emerges
- **N-Pattern Extraction Prompts**: Extracts maximum patterns initially, then refines them iteratively
- Query reformulation prompts
- Iterative pattern learning prompts
- Pattern consolidation prompts

**Two Main Approaches:**
1. **Consolidated Approach**: Uses `create_patterns_only_iterative_prompt()` to start from zero patterns and incrementally consolidate patterns after each batch
2. **N-Pattern Approach**: Uses `create_iterative_pattern_prompt()` to extract and refine patterns dynamically

### 3. Evaluation Metrics

Three evaluation calculators for measuring retrieval effectiveness:

#### MRR Calculator (`MRR_calculator.py`)
```bash
python MRR_calculator.py -qrels <qrels_file> -run <run_file> -metric mrr_cut_10 -result <output_file>
```

#### NDCG Calculator (`NDCG_calculator.py`)
```bash
python NDCG_calculator.py -qrels <qrels_file> -run <run_file> -metric ndcg_cut_10 -output <output_file>
```

#### Recall Calculator (`Recall_calculator.py`)
```bash
python Recall_calculator.py -qrels <qrels_file> -run <run_file> -metric recall_cut_1000 -result <output_file>
```

## Datasets

### Diamond Dataset
The primary dataset containing query reformulation pairs with relevance judgments. Located in `data/diamond_dataset/diamond_dataset.tsv`.

### TREC DL Datasets
- **TREC DL 2019**: `data/trecdl2019/`
- **TREC DL 2020**: `data/trecdl2020/`

Each contains:
- `original_queries.tsv`: Original query set
- `bm25_run_original_queries.tsv`: BM25 retrieval results
- `qrels.trec`: Relevance judgments for evaluation

### Development Dataset
- **Dev Small**: `data/dev_small/`
  - `queries.dev.small.tsv`: Small development query set
  - `qrels.dev.small.tsv`: Relevance judgments

### Chameleons Dataset
Specialized dataset with different chameleon species for domain-specific analysis:
- **Lesser (common6)**: 5 variants (0.1, 0.2, 0.3, 0.4, 0.5)
- **Pygmy (common5)**: 5 variants (0.1, 0.2, 0.3, 0.4, 0.5)  
- **Veiled (common4)**: 5 variants (0.1, 0.2, 0.3, 0.4, 0.5)

Each variant represents different levels of commonality or specificity in the dataset.

## BM25 Run File Generation

To generate BM25 run files for evaluation, use the following command (adjust paths as needed):

```bash
sh target/appassembler/bin/SearchMsmarco -hits 1000 -threads 50 \
 -index indexes/msmarco-passage/lucene-index-msmarco \
 -queries data/trecdl2020/original_queries.tsv \
 -output data/trecdl2020/bm25_run_original_queries.tsv
```

**Parameters:**
- `-hits 1000`: Retrieve top 1000 documents per query
- `-threads 50`: Use 50 threads for parallel processing
- `-index`: Path to MS MARCO Lucene index
- `-queries`: Input query file (TSV format)
- `-output`: Output run file path

## Experiment Workflow

1. **Data Preparation**: Ensure datasets are in the correct `data/` subdirectories
2. **Pattern Extraction**: Run `iterative_pattern_extraction.py` to extract reformulation patterns
3. **Pattern Analysis**: Review generated patterns in `results/[experiment_folder]/`
4. **Query Reformulation**: Apply patterns to new queries using the prompt templates
5. **Evaluation**: Use BM25 to generate run files and evaluate with the metric calculators

## Output Structure

### Experiment Results Directory (`results/`)

The results directory contains timestamped experiment folders, each representing a complete pattern extraction run. Each experiment folder is named using the format: `experiment_YYYYMMDD_HHMMSS_[model_name]`.

#### Customizedted Results of Patterns Extracted: `consolied_reformulation_patterns_qwen2.5:72b/`

This directory contains results from pattern extraction experiments using the Qwen2.5:72b model, demonstrating two different methodological approaches:

- **`consolidated_patterns_on_7310_pairs.json`**: Patterns extracted using the **consolidated prompt approach** - starts from zero patterns and after each batch consolidates with existing patterns in the list, focusing on maintaining a fixed-size set of high-quality, comprehensive patterns that capture generalizable transformation strategies
- **`extracted_patterns_from_1000_pairs.json`**: Patterns extracted using the **N-pattern extraction approach** - initially extracts N patterns to fill the maximum pattern list, then refines them to consider more comprehensive patterns
- **`extracted_patterns_from_4200_pairs.json`**: Continuation of the N-pattern approach after processing more query pairs

**Methodological Differences:**
- **Consolidated Approach**: Starts from zero patterns and incrementally builds the pattern list by consolidating new patterns with existing ones after each batch, maintaining a fixed number of high-quality patterns
- **N-Pattern Approach**: Starts by extracting the maximum number of patterns, then iteratively refines them to create more comprehensive and generalizable patterns

These files demonstrate different strategies for pattern consolidation and refinement in query reformulation research.

#### Pattern File Format

Each pattern file contains an array of pattern objects with the following structure:

```json
[
  {
    "pattern_name": "Semantic Clarification",
    "description": "Clarifies ambiguous terms or actions to make the query more specific and understandable.",
    "transformation_rule": "Replace vague or ambiguous terms with more precise and contextually relevant terms.",
    "examples": [
      ["weather in prague in may", "temperature in prague in may"],
      ["ponds brand is which company", "who is pond's parent company"]
    ]
  }
]
```

**Pattern Fields:**
- `pattern_name`: Descriptive name for the transformation pattern
- `description`: Detailed explanation of how the pattern improves query effectiveness
- `transformation_rule`: Abstract rule describing how to apply the pattern
- `examples`: Array of [original_query, reformulated_query] pairs demonstrating the pattern

#### Complete Experiment Output Structure

Each experiment generates:
- `extracted_patterns_FINAL.json`: Consolidated reformulation patterns
- `individual_patterns_FINAL.json`: Pattern mappings for each query pair
- `individual_patterns_FINAL.csv`: CSV version for easy analysis
- `extraction_results_FINAL.json`: Iteration-by-iteration results
- `experiment_metadata_FINAL.json`: Experiment configuration
- `experiment_summary.txt`: Comprehensive analysis summary
- `individual_patterns_LIVE.json`: Real-time pattern updates during processing
- `extracted_patterns_[N]_queries.json`: Intermediate pattern snapshots
- `individual_patterns_[N]_queries.json`: Intermediate individual mappings
- `extraction_results_[N]_queries.json`: Intermediate iteration results

## Requirementnning

- Python 3.7+
- Required packages: `openai`, `ollama`, `pandas`, `numpy`, `tqdm`, `requests`
- For BM25: Anserini/Pyserini toolkit with MS MARCO index

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for OpenAI models

### Model Selection
- **OpenAI**: gpt-4o, gpt-4, gpt-3.5-turbo, gpt-4-turbo
- **Ollama**: llama2, mistral, codellama, qwen2.5:72b, qwq:latest

