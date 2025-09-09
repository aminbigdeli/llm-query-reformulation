#!/usr/bin/env python3
"""
Oracle Query Reformulation for TREC-DL Datasets
Applies every pattern to every query systematically (Oracle).
"""

import os, sys, json, logging, csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
from ollama import Client

# Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("oracle_trecdl")

THIS_FILE = Path(__file__).resolve()
PROJ_ROOT = THIS_FILE.parent.parent
sys.path.append(str(PROJ_ROOT / "src"))

from query_reformulation_all_prompts import (
    ReformulationPattern,
    create_oracle_single_pattern_prompt
)

# Config
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:72b")
client = Client(host=OLLAMA_HOST)

PATTERNS_FILE = PROJ_ROOT / "results/consolied_reformulation_patterns_qwen2.5:72b/consolidated_patterns_on_7310_pairs.json"
DATASETS = {
    "trecdl2019": {"path": PROJ_ROOT / "data/trecdl2019/original_queries.tsv", "has_header": False},
    "trecdl2020": {"path": PROJ_ROOT / "data/trecdl2020/original_queries.tsv", "has_header": True},
}
OUTPUT_DIR = PROJ_ROOT / "results" / "oracle_reformulations_trecdl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_patterns() -> List[ReformulationPattern]:
    """Load patterns from JSON."""
    with open(PATTERNS_FILE, "r") as f:
        data = json.load(f)
    return [ReformulationPattern(
        pattern_name=p["pattern_name"],
        description=p["description"],
        transformation_rule=p["transformation_rule"],
        examples=p.get("examples", [])
    ) for p in data]

def load_queries(dataset_name: str) -> pd.DataFrame:
    """Load TREC-DL queries."""
    cfg = DATASETS[dataset_name]
    if cfg["has_header"]:
        return pd.read_csv(cfg["path"], sep="\t")
    else:
        return pd.read_csv(cfg["path"], sep="\t", names=["qid", "query"])

def call_oracle(query: str, pattern: ReformulationPattern) -> Dict[str, Any]:
    """Call LLM with oracle single pattern prompt."""
    try:
        messages = create_oracle_single_pattern_prompt(query, pattern)
        
        # Convert to Ollama format
        system_content = messages[0]["content"] if messages[0]["role"] == "system" else ""
        user_content = messages[1]["content"] if len(messages) > 1 else messages[0]["content"]
        if system_content:
            user_content = f"{system_content}\n\n{user_content}"
        
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": user_content}],
            options={"temperature": 0, "num_predict": 1000}
        )
        
        # Extract JSON
        content = response["message"]["content"].strip()
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        
        if start_idx != -1 and end_idx > start_idx:
            return json.loads(content[start_idx:end_idx])
        else:
            return {
                "reformulated_query": content,
                "pattern_applied": pattern.pattern_name,
                "explanation": "Non-JSON response",
                "applicable": False,
                "confidence": "low"
            }
    except Exception as e:
        logger.warning(f"LLM failed for pattern {pattern.pattern_name}: {e}")
        return {
            "reformulated_query": query,
            "pattern_applied": pattern.pattern_name,
            "explanation": f"Error: {e}",
            "applicable": False,
            "confidence": "low"
        }

def run_oracle():
    """Run oracle reformulation: every query × every pattern."""
    patterns = load_patterns()
    logger.info(f"Loaded {len(patterns)} patterns")
    
    for dataset_name in DATASETS.keys():
        logger.info(f"Processing {dataset_name}")
        queries_df = load_queries(dataset_name)
        logger.info(f"Loaded {len(queries_df)} queries")
        
        for i, pattern in enumerate(patterns):
            logger.info(f"Pattern {i+1}/{len(patterns)}: {pattern.pattern_name}")
            
            results = []
            for _, row in queries_df.iterrows():
                qid = str(row["qid"])
                query = row["query"]
                
                result = call_oracle(query, pattern)
                results.append({
                    "qid": qid,
                    "original_query": query,
                    "reformulated_query": result["reformulated_query"],
                    "pattern_applied": result["pattern_applied"],
                    "explanation": result["explanation"],
                    "applicable": result["applicable"],
                    "confidence": result["confidence"]
                })
            
            # Save results for this dataset × pattern combination
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            pattern_safe = pattern.pattern_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            
            # JSON
            json_file = OUTPUT_DIR / f"{dataset_name}_{pattern_safe}_{ts}.json"
            with open(json_file, "w") as f:
                json.dump({
                    "dataset": dataset_name,
                    "pattern": pattern.pattern_name,
                    "timestamp": ts,
                    "total_queries": len(results),
                    "reformulations": results
                }, f, indent=2)
            
            # CSV
            csv_file = OUTPUT_DIR / f"{dataset_name}_{pattern_safe}_{ts}.csv"
            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["qid", "original_query", "reformulated_query", "pattern_applied", "explanation", "applicable", "confidence"])
                writer.writeheader()
                writer.writerows(results)
            
            applicable_count = sum(1 for r in results if r["applicable"])
            logger.info(f"Saved {dataset_name} × {pattern.pattern_name}: {applicable_count}/{len(results)} applicable")

if __name__ == "__main__":
    logger.info(f"Oracle reformulation: {OLLAMA_HOST} | {OLLAMA_MODEL}")
    run_oracle()
    logger.info(f"Done. Results in: {OUTPUT_DIR}")