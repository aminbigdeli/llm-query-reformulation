#!/usr/bin/env python3
"""
Oracle Query Reformulation for TREC-DL Datasets (Ollama)
"""

import os, sys, json, logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
from ollama import Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("trecdl_reformulator")

THIS_FILE = Path(__file__).resolve()
PROJ_ROOT = THIS_FILE.parent.parent  # script is in src/, project root is parent
sys.path.append(str(PROJ_ROOT / "src"))  # so local imports work regardless of CWD

from query_reformulation_all_prompts import (
    ReformulationPattern,
    create_pattern_application_prompt
)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:72b")  # default to your model
client = Client(host=OLLAMA_HOST)
logger.info(f"Ollama host: {OLLAMA_HOST} | model: {OLLAMA_MODEL}")

PATTERNS_FILE = PROJ_ROOT / "results/consolied_reformulation_patterns_qwen2.5:72b/consolidated_patterns_on_7310_pairs.json"
DATASETS = {
    "trecdl2019": {
        "path": PROJ_ROOT / "data/trecdl2019/original_queries.tsv",
        "columns": ["qid", "query"],
        "has_header": False,    # used if auto-detect fails
    },
    "trecdl2020": {
        "path": PROJ_ROOT / "data/trecdl2020/original_queries.tsv",
        "columns": ["qid", "query"],
        "has_header": True,     # used if auto-detect fails
    },
}
OUTPUT_DIR = PROJ_ROOT / "results" / "trecdl_reformulations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_patterns(path: Path) -> List[ReformulationPattern]:
    """Load patterns from JSON file with error handling."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        patterns = [
            ReformulationPattern(
                pattern_name=p["pattern_name"],
                description=p["description"],
                transformation_rule=p["transformation_rule"],
                examples=p.get("examples", []),
            ) for p in data
        ]
        logger.info(f"Loaded {len(patterns)} patterns from {path.name}")
        return patterns
    except Exception as e:
        logger.error(f"Error loading patterns from {path}: {e}")
        raise

def read_tsv_auto(path: Path, fallback_has_header: bool, columns: List[str]) -> pd.DataFrame:
    """
    Try header='infer' first; if it doesn't yield expected columns, fallback to names=columns.
    This prevents header/column mismatches from breaking the run.
    """
    try:
        df = pd.read_csv(path, sep="\t", header="infer")
        if set(columns).issubset(df.columns):
            logger.info(f"Auto-detected header in {path.name}")
            return df[columns]
        # Fallback to explicit names
        logger.info(f"Using explicit column names for {path.name}")
        df = pd.read_csv(path, sep="\t", header=None, names=columns)
        return df
    except Exception as e:
        logger.warning(f"TSV auto-detection failed for {path}, trying fallback: {e}")
        # Last-ditch fallback following dataset's declared has_header
        try:
            if fallback_has_header:
                return pd.read_csv(path, sep="\t")[columns]
            else:
                return pd.read_csv(path, sep="\t", header=None, names=columns)
        except Exception as e2:
            logger.error(f"All TSV reading methods failed for {path}: {e2}")
            raise

def to_ollama_messages(openai_style: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Flatten system into first user message for Ollama."""
    system = ""
    msgs = []
    for m in openai_style:
        if m["role"] == "system":
            system = system + ("\n\n" if system else "") + m["content"]
        else:
            msgs.append({"role": m["role"], "content": m["content"]})
    if system and msgs and msgs[0]["role"] == "user":
        msgs[0]["content"] = f"{system}\n\n{msgs[0]['content']}"
    return msgs or ([{"role": "user", "content": system}] if system else [])

def extract_json(text: str) -> Dict[str, Any]:
    """Robustly pull the outermost JSON object from a possibly-chatty response."""
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(text[s : e + 1])
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            pass
    # fallback: return raw text as a minimal structured record
    logger.warning("Using fallback response structure for non-JSON output")
    return {
        "reformulated_query": text.strip(),
        "applied_patterns": [],
        "explanation": "LLM returned non-JSON; using raw content",
        "confidence": "low",
        "pattern_justification": "raw"
    }

class TRECDLReformulator:
    def __init__(self, patterns_file: Path, model: str = OLLAMA_MODEL):
        self.patterns = load_patterns(patterns_file)
        self.model = model
        self.results: Dict[str, List[Dict[str, Any]]] = {}

    def _messages_for_query(self, q: str) -> List[Dict[str, str]]:
        return create_pattern_application_prompt(
            original_query=q,
            final_patterns=self.patterns,
            context_documents=None,
        )
    
    def _call_llm(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Call Ollama LLM with robust error handling."""
        try:
            om = to_ollama_messages(messages)
            resp = client.chat(
                model=self.model,
                messages=om,
                options={"temperature": 0, "num_predict": 1000}
            )
            return extract_json(resp["message"]["content"])
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            raise
    
    def reformulate(self, qid: str, q: str) -> Dict[str, Any]:
        """Reformulate a single query with full error handling."""
        try:
            response = self._call_llm(self._messages_for_query(q))
            # Normalize required fields
            return {
                "qid": qid,
                "original_query": q,
                "reformulated_query": response.get("reformulated_query", q),
                "applied_patterns": response.get("applied_patterns", []),
                "explanation": response.get("explanation", ""),
                "confidence": response.get("confidence", ""),
                "pattern_justification": response.get("pattern_justification", ""),
            }
        except Exception as e:
            logger.warning(f"LLM failed for qid={qid}: {e}")
            return {
                "qid": qid,
                "original_query": q,
                "reformulated_query": q,
                "applied_patterns": [],
                "explanation": f"Error: {e}",
                "confidence": "low",
                "pattern_justification": "error",
            }
    
    def run_dataset(self, name: str, cfg: Dict[str, Any]) -> None:
        """Process all queries in a dataset with progress logging."""
        logger.info(f"Processing {name}...")
        df = read_tsv_auto(cfg["path"], cfg["has_header"], cfg["columns"])
        
        out = []
        for i, (_, r) in enumerate(df.iterrows()):
            qid = str(r["qid"])
            query = r["query"]
            
            # Log progress every 10 queries
            if (i + 1) % 10 == 0:
                logger.info(f"{name}: Processing query {i+1}/{len(df)} (qid: {qid})")
            
            result = self.reformulate(qid, query)
            out.append(result)
            
            # Log reformulation result
            logger.debug(f"Query {qid}: '{query}' -> '{result['reformulated_query']}'")
        
        self.results[name] = out
        logger.info(f"{name}: processed {len(out)} queries")
    
    def save(self) -> None:
        """Save results with Zeus server compatible paths."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for name, out in self.results.items():
            # JSON
            jf = OUTPUT_DIR / f"{name}_reformulations_{ts}.json"
            with open(jf, "w") as f:
                json.dump({
                    "dataset": name,
                    "timestamp": ts,
                    "total_queries": len(out),
                    "patterns_used": len(self.patterns),
                    "patterns_file": str(PATTERNS_FILE),
                    "ollama_host": OLLAMA_HOST,
                    "ollama_model": OLLAMA_MODEL,
                    "reformulations": out
                }, f, indent=2)
            
            # CSV with robust handling
            import csv
            cf = OUTPUT_DIR / f"{name}_reformulations_{ts}.csv"
            cols = ["qid","original_query","reformulated_query","applied_patterns","explanation","confidence","pattern_justification"]
            with open(cf, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for row in out:
                    row = dict(row)
                    ap = row.get("applied_patterns", [])
                    if isinstance(ap, list): 
                        row["applied_patterns"] = "; ".join(ap)
                    w.writerow({k: row.get(k, "") for k in cols})
            
            logger.info(f"Saved {name}: {jf.name}, {cf.name}")
        
        # Summary file
        summary_file = OUTPUT_DIR / f"reformulation_summary_{ts}.json"
        summary = {
            "experiment_timestamp": ts,
            "patterns_file": str(PATTERNS_FILE),
            "total_patterns": len(self.patterns),
            "datasets_processed": list(self.results.keys()),
            "total_queries": sum(len(results) for results in self.results.values()),
            "pattern_names": [p.pattern_name for p in self.patterns],
            "ollama_host": OLLAMA_HOST,
            "ollama_model": OLLAMA_MODEL
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved experiment summary to {summary_file.name}")


def main():
    """Main function - Zeus server cluster compatible."""
    # Validate environment
    logger.info(f"Starting reformulation with Ollama host: {OLLAMA_HOST}, model: {OLLAMA_MODEL}")
    
    if not PATTERNS_FILE.exists():
        logger.error(f"Patterns file not found: {PATTERNS_FILE}")
        raise FileNotFoundError(f"Patterns file not found: {PATTERNS_FILE}")
    
    # Test Ollama connection
    try:
        models_response = client.list()
        available_models = [
            (m.get("name") or m.get("model") or m) for m in models_response.get("models", [])
        ]
        if OLLAMA_MODEL not in available_models:
            logger.warning(f"Model {OLLAMA_MODEL} not found. Available: {available_models}")
        else:
            logger.info(f"Confirmed model {OLLAMA_MODEL} is available")
    except Exception as e:
        logger.warning(f"Could not verify Ollama models: {e}")
    
    # Run reformulation
    try:
        tr = TRECDLReformulator(PATTERNS_FILE, model=OLLAMA_MODEL)
        for name, cfg in DATASETS.items():
            logger.info(f"Starting dataset: {name}")
            tr.run_dataset(name, cfg)
        tr.save()
        logger.info(f"Done. Outputs in: {OUTPUT_DIR}")
    except Exception as e:
        logger.error(f"Reformulation failed: {e}")
        raise


if __name__ == "__main__":
    main()
