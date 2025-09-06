from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class QueryPair:
    """Represents a pair of original and reformulated queries"""
    original_query: str
    reformulated_query: str
    query_id: str = ""

@dataclass
class ReformulationPattern:
    """Represents a reformulation pattern identified from query pairs"""
    pattern_name: str
    description: str
    transformation_rule: str
    examples: List[Tuple[str, str]]  # (original, reformulated) examples

def create_pattern_extraction_prompt(query_pairs: List[QueryPair], 
                                   existing_patterns: List[ReformulationPattern] = None,
                                   max_patterns: int = 15) -> List[Dict[str, str]]:
    """
    Creates a prompt for reformulation pattern extraction
    
    Args:
        query_pairs: List of query pairs to analyze
        existing_patterns: Previously identified patterns (for iterative updates)
        max_patterns: Maximum number of patterns to identify
        
    Returns:
        List of messages for the LLM
    """
    messages = [
        {
            "role": "system",
            "content": "You are QueryReformulationLLM, an intelligent assistant that can identify and extract patterns from query reformulation pairs. You analyze how original queries are transformed into reformulated queries to identify common patterns and transformation rules."
        },
        {
            "role": "user",
            "content": get_pattern_extraction_content(query_pairs, existing_patterns, max_patterns)
        }
    ]
    return messages

def get_pattern_extraction_content(query_pairs: List[QueryPair], 
                                 existing_patterns: List[ReformulationPattern] = None,
                                 max_patterns: int = 15) -> str:
    """
    Gets the content for the pattern extraction prompt
    
    Args:
        query_pairs: List of query pairs to analyze
        existing_patterns: Previously identified patterns
        max_patterns: Maximum number of patterns to identify
        
    Returns:
        Formatted prompt content
    """
    
    # Format query pairs
    query_pairs_text = "\n".join([
        f"[{i+1}] Original: \"{pair.original_query}\" ? Reformulated: \"{pair.reformulated_query}\""
        for i, pair in enumerate(query_pairs)
    ])
    
    # Format existing patterns if provided
    existing_patterns_text = ""
    if existing_patterns:
        existing_patterns_text = "\n".join([
            f"- {pattern.pattern_name}: {pattern.description} (Rule: {pattern.transformation_rule})"
            for pattern in existing_patterns
        ])
        existing_patterns_text = f"\nPreviously Identified Patterns:\n{existing_patterns_text}\n"
    
    return f"""Analyze the following query reformulation pairs to identify patterns that transform original queries into reformulated queries. 

Your task is to:
1. Identify common patterns in how queries are reformulated
2. Extract transformation rules that could be applied to new queries
3. Create a comprehensive list of reformulation patterns

Focus on patterns such as:
- Query expansion (adding synonyms, related terms)
- Query refinement (making queries more specific)
- Query generalization (making queries broader)
- Query restructuring (changing word order, grammar)
- Query disambiguation (clarifying ambiguous terms)
- Query optimization (improving search effectiveness)

{existing_patterns_text}
Query Pairs to Analyze:
{query_pairs_text}

Instructions:
- Identify at most {max_patterns} distinct reformulation patterns
- For each pattern, provide a clear name, description, and transformation rule
- Include specific examples from the provided pairs
- Order patterns by frequency and importance
- Ensure patterns are actionable and can be applied to new queries

Return the patterns in the following JSON format:
{{
    "patterns": [
        {{
            "pattern_name": "Pattern Name",
            "description": "Detailed description of the pattern",
            "transformation_rule": "How to apply this pattern",
            "examples": [
                ["original_query_1", "reformulated_query_1"],
                ["original_query_2", "reformulated_query_2"]
            ],
            "frequency": "number of occurrences in the dataset"
        }}
    ],
    "summary": "Brief summary of key findings"
}}

Pattern Analysis:"""




def create_iterative_pattern_prompt(query_pairs: List[QueryPair], 
                                 consolidated_patterns: List[ReformulationPattern] = None,
                                 creator_max_patterns: int = 20) -> List[Dict[str, str]]:
    """
    Creates a single iterative prompt for pattern extraction and consolidation.
    This updates the consolidated pattern list based on new query pairs.
    
    Args:
        query_pairs: List of query pairs to analyze
        consolidated_patterns: Previously consolidated patterns (if available)
        creator_max_patterns: Maximum number of patterns to keep
        
    Returns:
        List of messages for the LLM
    """
    messages = [
        {
            "role": "system",
            "content": "You are QueryReformulationLLM, an intelligent assistant that identifies and updates abstract patterns that describe how queries are reformulated to improve retrieval effectiveness. Your goal is to consolidate high-level transformation strategies that explain how and why a reformulation improves the query."
        },
        {
            "role": "user",
            "content": get_iterative_pattern_content(query_pairs, consolidated_patterns, creator_max_patterns)
        }
    ]
    return messages


def get_iterative_pattern_content(query_pairs: List[QueryPair], 
                               consolidated_patterns: List[ReformulationPattern] = None,
                               creator_max_patterns: int = 15) -> str:
    """
    Gets the content for the iterative pattern prompt for reformulation strategies.
    
    Args:
        query_pairs: List of query pairs to analyze
        consolidated_patterns: Previously consolidated patterns
        creator_max_patterns: Maximum number of patterns to keep
        
    Returns:
        Formatted prompt content
    """
    
    # Format query pairs with IDs for individual pattern extraction
    query_pairs_text = "\n".join([
        f"[{i+1}] Query ID: {pair.query_id} | Original: \"{pair.original_query}\" → Reformulated: \"{pair.reformulated_query}\""
        for i, pair in enumerate(query_pairs)
    ])
    
    # Format existing consolidated patterns if available
    consolidated_patterns_text = ""
    if consolidated_patterns:
        consolidated_patterns_text = "\n".join([
            f"- {pattern.pattern_name}: {pattern.description} (Rule: {pattern.transformation_rule})"
            for pattern in consolidated_patterns
        ])
        consolidated_patterns_text = f"\nCurrent Consolidated Patterns:\n{consolidated_patterns_text}\n"
    
    return f"""You are given query reformulation pairs and an optional list of existing abstract reformulation patterns.

Your task is to:
1. For each individual query pair, identify the actual pattern(s) applied to transform the original query into the reformulated query. Extract the specific transformation strategy used for each individual pair.
2. Consolidate the list of high-level reformulation patterns by incorporating any newly discovered patterns from task 1, merging semantically redundant ones, and refining names/descriptions so the final set captures generalizable strategies.

Each pattern must include:
- A short pattern_name describing the type of transformation (e.g., "Semantic Clarification")
- A description explaining how this pattern helps improve effectiveness
- A transformation_rule summarizing the abstract logic (e.g., "replace ambiguous action with contextual behavior")
- A few examples from the query pairs in the format [["original", "reformulated"]]

For each query pair, identify which pattern(s) were applied and provide a brief explanation.

Return the results in the following JSON format:
{{
    "consolidated_patterns": [
        {{
            "pattern_name": "Pattern Name",
            "description": "Description of the pattern",
            "transformation_rule": "How to apply this pattern",
            "examples": [["original_query", "reformulated_query"]]
        }}
    ],
    "individual_patterns": [
        {{
            "query_id": "actual_query_id",
            "original_query": "original query text",
            "reformulated_query": "reformulated query text",
            "applied_patterns": ["pattern_name_1", "pattern_name_2"],
            "explanation": "Brief explanation of what patterns were applied and why"
        }}
    ],
    "summary": "Brief summary of key findings"
}}

Query Reformulation Pairs:
{query_pairs_text}
{consolidated_patterns_text}
Initial Pattern List Length: {len(consolidated_patterns) if consolidated_patterns else 0}

Instructions:
- Return at most {creator_max_patterns} consolidated patterns
- For each query pair, identify the actual transformation pattern(s) applied, even if they're not yet in the consolidated list
- Use the actual query_id from the dataset (e.g., "qid_123") in the individual_patterns section
- Do not repeat semantically redundant patterns
- Avoid trivial lexical changes unless they contribute significantly to meaning or intent
- Order patterns by relevance or frequency across the provided query pairs
- If a query pair shows a new transformation strategy, extract and name that pattern for the individual entry

Return the results in the JSON format specified above:
"""




def create_patterns_only_iterative_prompt(
    query_pairs: List[QueryPair],
    consolidated_patterns: List[ReformulationPattern] = None,
    max_patterns: int = 10,
) -> List[Dict[str, str]]:
    """
    Creates a prompt that manages a fixed-size set of consolidated patterns only.

    Behavior rules enforced in the prompt:
    - Always return exactly `max_patterns` consolidated patterns.
    - If current consolidated patterns are fewer than `max_patterns`, expand by adding
      new non-redundant patterns from the query pairs until the count equals `max_patterns`.
    - Only consolidate (merge/replace) when the set is already at `max_patterns` and a
      new pattern is required by evidence from the query pairs.

    Args:
        query_pairs: List of query pairs to analyze
        consolidated_patterns: Previously consolidated patterns (if available)
        max_patterns: Fixed number of patterns to maintain and return (default 10)

    Returns:
        List of messages for the LLM
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are QueryReformulationLLM, an assistant that maintains a fixed-size, "
                "high-quality set of consolidated reformulation patterns that explain how "
                "and why query reformulations improve retrieval effectiveness."
            ),
        },
        {
            "role": "user",
            "content": get_patterns_only_iterative_content(
                query_pairs=query_pairs,
                consolidated_patterns_constrained=consolidated_patterns,
                max_patterns=max_patterns,
            ),
        },
    ]
    return messages


def get_patterns_only_iterative_content(
    query_pairs: List[QueryPair],
    consolidated_patterns_constrained: List[ReformulationPattern] = None,
    max_patterns: int = 10,
) -> str:
    """
    Builds the content for a patterns-only iterative prompt that enforces a fixed-size
    consolidated set.

    Rules encoded in the instructions:
    - Always output exactly `max_patterns` consolidated patterns.
    - If fewer than `max_patterns` exist, add new non-redundant patterns derived from
      the query pairs until the set size equals `max_patterns` (no consolidation yet).
    - If already at `max_patterns`, only consolidate when the query pairs provide
      evidence for a new pattern that is not currently captured; merge or replace the
      least generalizable/lowest-utility overlapping patterns to keep the size fixed.
    - Output only the consolidated patterns. No per-pair mapping, no summary.

    Args:
        query_pairs: List of query pairs to analyze
        consolidated_patterns_constrained: Current fixed-size consolidated patterns
        max_patterns: Fixed number of patterns to maintain and return (default 10)

    Returns:
        Formatted prompt content string
    """

    # Format query pairs with IDs (when available) for evidence/examples
    query_pairs_text = "\n".join(
        [
            f"[{i+1}] Query ID: {pair.query_id} | Original: \"{pair.original_query}\" → Reformulated: \"{pair.reformulated_query}\""
            for i, pair in enumerate(query_pairs)
        ]
    )

    # Format existing consolidated patterns, if provided
    consolidated_patterns_text = ""
    if consolidated_patterns_constrained:
        consolidated_patterns_text = "\n".join(
            [
                f"- {pattern.pattern_name}: {pattern.description} (Rule: {pattern.transformation_rule})"
                for pattern in consolidated_patterns_constrained
            ]
        )
        consolidated_patterns_text = (
            f"\nCurrent Consolidated Patterns (size={len(consolidated_patterns_constrained)}):\n"
            f"{consolidated_patterns_text}\n"
        )

    return f"""You are given query reformulation pairs and an optional current set of consolidated reformulation patterns.

Your objective is to maintain a fixed-size, high-quality set of consolidated patterns that explain how reformulations improve retrieval. The set must always contain exactly {max_patterns} patterns.

Decision rules:
- If the current set size is less than {max_patterns}, expand the set by adding new, non-redundant patterns evidenced by the pairs until the set reaches exactly {max_patterns}. Do not merge or drop existing patterns in this case.
- If the current set already has {max_patterns} patterns, determine whether the pairs reveal a new transformation strategy that is not captured by the current set.
  - If no new strategy is needed, keep the existing set (you may refine names/descriptions/rules for clarity) while preserving exactly {max_patterns} patterns.
  - If a new strategy is needed, integrate it by consolidating: merge semantically overlapping or low-utility patterns such that the final set remains exactly {max_patterns}. Prioritize patterns that are generalizable, non-redundant, and impactful.

Each consolidated pattern must include:
- pattern_name: concise, descriptive name (e.g., "Semantic Clarification")
- description: how this pattern improves effectiveness
- transformation_rule: abstract rule explaining how to apply it
- examples: a few pairs from the provided data in the form [["original", "reformulated"]]

Output requirements:
- Return a single JSON object with exactly one key: "consolidated_patterns".
- "consolidated_patterns" must contain exactly {max_patterns} items.
- Do not include any other top-level keys or extra commentary.

Query Reformulation Pairs:
{query_pairs_text}
{consolidated_patterns_text}
Max Pattern Count: {max_patterns}

Return JSON only in the following shape:
{{
  "consolidated_patterns": [
    {{
      "pattern_name": "Pattern Name",
      "description": "Description of the pattern",
      "transformation_rule": "How to apply this pattern",
      "examples": [["original_query", "reformulated_query"]]
    }}
  ]
}}
"""
