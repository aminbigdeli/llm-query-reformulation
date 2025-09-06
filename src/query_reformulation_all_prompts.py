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

def create_query_reformulation_prompt(original_query: str, 
                                   patterns: List[ReformulationPattern],
                                   context_documents: List[str] = None) -> List[Dict[str, str]]:
    """
    Creates a prompt for applying reformulation patterns to a new query
    
    Args:
        original_query: The query to reformulate
        patterns: List of reformulation patterns to apply
        context_documents: Optional context documents for informed reformulation
        
    Returns:
        List of messages for the LLM
    """
    messages = [
        {
            "role": "system",
            "content": "You are QueryReformulationLLM, an expert at reformulating search queries to improve retrieval effectiveness. You apply learned patterns to transform queries while maintaining their intent."
        },
        {
            "role": "user",
            "content": get_reformulation_content(original_query, patterns, context_documents)
        }
    ]
    return messages

def get_reformulation_content(original_query: str, 
                            patterns: List[ReformulationPattern],
                            context_documents: List[str] = None) -> str:
    """
    Gets the content for the query reformulation prompt
    
    Args:
        original_query: The query to reformulate
        patterns: List of reformulation patterns to apply
        context_documents: Optional context documents
        
    Returns:
        Formatted prompt content
    """
    
    # Format patterns
    patterns_text = "\n".join([
        f"- {pattern.pattern_name}: {pattern.description}\n  Rule: {pattern.transformation_rule}\n  Examples: {pattern.examples[:2]}"  # Show first 2 examples
        for pattern in patterns
    ])
    
    # Format context if provided
    context_text = ""
    if context_documents:
        context_text = "\n".join([
            f"[{i+1}] {doc}"
            for i, doc in enumerate(context_documents)
        ])
        context_text = f"\nContext Documents:\n{context_text}\n"
    
    return f"""Reformulate the following query using the provided reformulation patterns to improve its search effectiveness.

            Original Query: "{original_query}"

            Available Reformulation Patterns:
            {patterns_text}
            {context_text}
            Instructions:
            1. Analyze the original query and identify which patterns are most applicable
            2. Apply the most relevant patterns to create an improved reformulated query
            3. Ensure the reformulated query maintains the original intent
            4. Consider the context documents if provided for more informed reformulation
            5. Provide a brief explanation of which patterns were applied and why

            Return the reformulation in the following JSON format:
            {{
                "reformulated_query": "the improved query",
                "applied_patterns": ["pattern_name_1", "pattern_name_2"],
                "explanation": "brief explanation of the reformulation",
                "confidence": "high/medium/low"
            }}

            Query Reformulation:"""

def create_iterative_pattern_learning_prompt(query_pairs: List[QueryPair],
                                          current_patterns: List[ReformulationPattern],
                                          iteration_number: int) -> List[Dict[str, str]]:
    """
    Creates a prompt for iterative pattern learning
    
    Args:
        query_pairs: New batch of query pairs to analyze
        current_patterns: Patterns identified in previous iterations
        iteration_number: Current iteration number
        
    Returns:
        List of messages for the LLM
    """
    messages = [
        {
            "role": "system",
            "content": "You are QueryReformulationLLM, an intelligent assistant that iteratively learns and refines query reformulation patterns. You analyze new query pairs and update existing patterns or identify new ones."
        },
        {
            "role": "user",
            "content": get_iterative_learning_content(query_pairs, current_patterns, iteration_number)
        }
    ]
    return messages

def get_iterative_learning_content(query_pairs: List[QueryPair],
                                 current_patterns: List[ReformulationPattern],
                                 iteration_number: int) -> str:
    """
    Gets the content for iterative pattern learning
    
    Args:
        query_pairs: New batch of query pairs
        current_patterns: Existing patterns
        iteration_number: Current iteration
        
    Returns:
        Formatted prompt content
    """
    
    # Format current patterns
    current_patterns_text = "\n".join([
        f"- {pattern.pattern_name}: {pattern.description} (Rule: {pattern.transformation_rule})"
        for pattern in current_patterns
    ])
    
    # Format new query pairs
    new_pairs_text = "\n".join([
        f"[{i+1}] Original: \"{pair.original_query}\" ? Reformulated: \"{pair.reformulated_query}\""
        for i, pair in enumerate(query_pairs)
    ])
    
    return f"""This is iteration {iteration_number} of pattern learning. You have access to:
            1. Previously identified patterns: {len(current_patterns)} patterns
            2. New query pairs to analyze: {len(query_pairs)} pairs

            Current Patterns:
            {current_patterns_text}

            New Query Pairs to Analyze:
            {new_pairs_text}

            Your task is to:
            1. Analyze the new query pairs
            2. Determine if they fit existing patterns or reveal new patterns
            3. Update existing patterns if needed (add examples, refine rules)
            4. Identify any new patterns not covered by existing ones
            5. Provide an updated comprehensive pattern list

            Return the updated patterns in JSON format:
            {{
                "updated_patterns": [
                    {{
                        "pattern_name": "Pattern Name",
                        "description": "Updated description",
                        "transformation_rule": "Updated rule",
                        "examples": [["original", "reformulated"]],
                        "frequency": "updated count",
                        "is_new": "true/false"
                    }}
                ],
                "new_patterns_count": "number of new patterns identified",
                "updated_patterns_count": "number of existing patterns updated",
                "summary": "summary of changes in this iteration"
            }}

            Pattern Analysis:""" 

def create_final_consolidation_prompt(all_learned_patterns: List[ReformulationPattern], 
                                    target_pattern_count: int = 10) -> List[Dict[str, str]]:
    """
    Creates a prompt for final pattern consolidation to produce a fixed number of patterns.
    
    Args:
        all_learned_patterns: All patterns learned through iterative process
        target_pattern_count: Desired number of final consolidated patterns
        
    Returns:
        List of messages for the LLM
    """
    messages = [
        {
            "role": "system",
            "content": "You are QueryReformulationLLM, an expert at consolidating and refining query reformulation patterns. You analyze all learned patterns and create a final, comprehensive set of the most important and distinct patterns."
        },
        {
            "role": "user",
            "content": get_final_consolidation_content(all_learned_patterns, target_pattern_count)
        }
    ]
    return messages

def get_final_consolidation_content(all_learned_patterns: List[ReformulationPattern], 
                                  target_pattern_count: int = 10) -> str:
    """
    Gets the content for the final consolidation prompt.
    
    Args:
        all_learned_patterns: All patterns learned through iterative process
        target_pattern_count: Desired number of final patterns
        
    Returns:
        Formatted prompt content
    """
    
    # Format all learned patterns
    patterns_text = "\n".join([
        f"- {pattern.pattern_name}: {pattern.description}\n  Rule: {pattern.transformation_rule}\n  Examples: {pattern.examples[:3]}"  # Show first 3 examples
        for pattern in all_learned_patterns
    ])
    
    return f"""You have learned {len(all_learned_patterns)} patterns through iterative analysis of query reformulation pairs. Now consolidate these into exactly {target_pattern_count} final, comprehensive patterns.

All Learned Patterns:
{patterns_text}

Your task is to:
1. Analyze all learned patterns and identify overlapping or similar patterns
2. Merge related patterns into comprehensive, distinct patterns
3. Prioritize patterns by frequency, importance, and applicability
4. Create exactly {target_pattern_count} final patterns that cover the most important reformulation strategies
5. Ensure each final pattern is distinct and actionable

Consolidation Guidelines:
- Merge patterns that represent similar transformation strategies
- Keep the most comprehensive and well-exemplified patterns
- Ensure coverage of different types of reformulation (expansion, refinement, restructuring, etc.)
- Prioritize patterns with more examples and higher frequency
- Create patterns that can be easily applied to new queries

Return the consolidated patterns in the following JSON format:
{{
    "final_patterns": [
        {{
            "pattern_name": "Comprehensive Pattern Name",
            "description": "Detailed description covering all related strategies",
            "transformation_rule": "Comprehensive rule for applying this pattern",
            "examples": [
                ["original_query_1", "reformulated_query_1"],
                ["original_query_2", "reformulated_query_2"],
                ["original_query_3", "reformulated_query_3"]
            ],
            "frequency": "total frequency across all merged patterns",
            "merged_from": ["pattern_name_1", "pattern_name_2"]
        }}
    ],
    "consolidation_summary": "Summary of how patterns were merged and prioritized",
    "total_patterns_consolidated": "number of original patterns",
    "final_patterns_count": "{target_pattern_count}"
}}

Final Pattern Consolidation:"""

def create_pattern_application_prompt(original_query: str, 
                                   final_patterns: List[ReformulationPattern],
                                   context_documents: List[str] = None) -> List[Dict[str, str]]:
    """
    Creates a prompt for applying the final consolidated patterns to a new query.
    
    Args:
        original_query: The query to reformulate
        final_patterns: The final consolidated patterns to apply
        context_documents: Optional context documents
        
    Returns:
        List of messages for the LLM
    """
    messages = [
        {
            "role": "system",
            "content": "You are QueryReformulationLLM, an expert at applying consolidated reformulation patterns to improve search queries. You use the final, refined patterns to transform queries effectively."
        },
        {
            "role": "user",
            "content": get_pattern_application_content(original_query, final_patterns, context_documents)
        }
    ]
    return messages

def get_pattern_application_content(original_query: str, 
                                 final_patterns: List[ReformulationPattern],
                                 context_documents: List[str] = None) -> str:
    """
    Gets the content for the pattern application prompt.
    
    Args:
        original_query: The query to reformulate
        final_patterns: The final consolidated patterns
        context_documents: Optional context documents
        
    Returns:
        Formatted prompt content
    """
    
    # Format final patterns with better structure
    patterns_text = "\n".join([
        f"Pattern {i+1}: {pattern.pattern_name}\n"
        f"  Description: {pattern.description}\n"
        f"  Transformation Rule: {pattern.transformation_rule}\n"
        f"  Examples: {pattern.examples[:3] if pattern.examples else 'No examples'}\n"
        for i, pattern in enumerate(final_patterns)
    ])
    
    # Format context if provided
    context_text = ""
    if context_documents:
        context_text = "\n".join([
            f"[{i+1}] {doc}"
            for i, doc in enumerate(context_documents)
        ])
        context_text = f"\nContext Documents:\n{context_text}\n"
    
    return f"""You are an expert at query reformulation. Your task is to improve the given query using the available reformulation patterns.

Original Query: "{original_query}"

Available Reformulation Patterns ({len(final_patterns)} patterns):
{patterns_text}
{context_text}
Instructions:
1. Analyze the original query carefully
2. Identify which patterns are most applicable to this specific query
3. Apply the selected patterns to create an improved reformulated query
4. Ensure the reformulated query maintains the original intent while being more effective
5. Consider the transformation rules and examples provided for each pattern

Return ONLY a JSON object with the following structure:
{{
    "reformulated_query": "the improved query",
    "applied_patterns": ["pattern_name_1", "pattern_name_2"],
    "explanation": "brief explanation of what was changed and why",
    "confidence": "high/medium/low",
    "pattern_justification": "why these specific patterns were chosen for this query"
}}

Important:
- Return ONLY the JSON object, no additional text
- Use exact pattern names from the list above
- Be specific about what changes were made and why
- If no patterns are applicable, return the original query unchanged

Query Reformulation:""" 


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
    
    # Format query pairs
    query_pairs_text = "\n".join([
        f"[{i+1}] Original: \"{pair.original_query}\" → Reformulated: \"{pair.reformulated_query}\""
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

Your task is to update the list of high-level reformulation patterns that describe how the original queries are transformed into more effective ones. Focus on capturing generalizable transformation strategies such as semantic shifts, contextual additions, synonym substitutions, or intent clarifications.

Each pattern must include:
- A short pattern_name describing the type of transformation (e.g., "Semantic Clarification")
- A description explaining how this pattern helps improve effectiveness
- A transformation_rule summarizing the abstract logic (e.g., "replace ambiguous action with contextual behavior")
- A few examples from the query pairs in the format [["original", "reformulated"]]

Return the updated list of patterns only, as a Python list of dictionaries, with a maximum of {creator_max_patterns} patterns. Do not repeat semantically redundant patterns. Avoid trivial lexical changes unless they contribute significantly to meaning or intent.

Order the patterns by relevance or frequency across the provided query pairs.

Query Reformulation Pairs:
{query_pairs_text}
{consolidated_patterns_text}
Initial Pattern List Length: {len(consolidated_patterns) if consolidated_patterns else 0}

Only return the updated list of patterns in the following format with maximum of two examples per pattern:
[{{"pattern_name": "...", "description": "...", "transformation_rule": "...", "examples": [["...", "..."]]}}]

Updated Pattern List:
"""





# def create_iterative_pattern_prompt(query_pairs: List[QueryPair], 
#                                  consolidated_patterns: List[ReformulationPattern] = None,
#                                  creator_max_patterns: int = 20) -> List[Dict[str, str]]:
#     """
#     Creates a single iterative prompt for pattern extraction and consolidation.
#     Similar to get_nugget_prompt_content, this updates the consolidated pattern list
#     based on new query pairs.
    
#     Args:
#         query_pairs: List of query pairs to analyze
#         consolidated_patterns: Previously consolidated patterns (if available)
#         creator_max_patterns: Maximum number of patterns to keep
        
#     Returns:
#         List of messages for the LLM
#     """
#     messages = [
#         {
#             "role": "system",
#             "content": "You are QueryReformulationLLM, an intelligent assistant that can update a list of query reformulation patterns to best capture the transformation strategies from query pairs. You analyze how original queries are transformed into reformulated queries to identify and update patterns."
#         },
#         {
#             "role": "user",
#             "content": get_iterative_pattern_content(query_pairs, consolidated_patterns, creator_max_patterns)
#         }
#     ]
#     return messages

# def get_iterative_pattern_content(query_pairs: List[QueryPair], 
#                                consolidated_patterns: List[ReformulationPattern] = None,
#                                creator_max_patterns: int = 15) -> str:
#     """
#     Gets the content for the iterative pattern prompt.
#     Similar to get_nugget_prompt_content but for query reformulation patterns.
    
#     Args:
#         query_pairs: List of query pairs to analyze
#         consolidated_patterns: Previously consolidated patterns
#         creator_max_patterns: Maximum number of patterns to keep
        
#     Returns:
#         Formatted prompt content
#     """
    
#     # Format query pairs
#     query_pairs_text = "\n".join([
#         f"[{i+1}] Original: \"{pair.original_query}\" ? Reformulated: \"{pair.reformulated_query}\""
#         for i, pair in enumerate(query_pairs)
#     ])
    
#     # Format existing consolidated patterns if available
#     consolidated_patterns_text = ""
#     if consolidated_patterns:
#         consolidated_patterns_text = "\n".join([
#             f"- {pattern.pattern_name}: {pattern.description} (Rule: {pattern.transformation_rule})"
#             for pattern in consolidated_patterns
#         ])
#         consolidated_patterns_text = f"\nCurrent Consolidated Patterns:\n{consolidated_patterns_text}\n"
    
#     return f"""Update the list of query reformulation patterns, if needed, so they best capture the transformation strategies from the provided query pairs. Leverage only the initial list of patterns (if exists) and the provided query pairs (this is an iterative process). Return only the final list of all patterns in a Pythonic list format (even if no updates). Make sure there is no redundant information. Ensure the updated pattern list has at most {creator_max_patterns} patterns (can be less), keeping only the most frequent and applicable ones. Order them in decreasing order of importance. Prefer patterns that provide more comprehensive transformation strategies.

# Query Pairs to Analyze:
# {query_pairs_text}
# {consolidated_patterns_text}
# Initial Pattern List Length: {len(consolidated_patterns) if consolidated_patterns else 0}

# Only update the list of patterns (if needed, else return as is). Do not explain. Always answer in pattern format. List in the form [{{"pattern_name": "name", "description": "description", "transformation_rule": "rule", "examples": [["original", "reformulated"]]}}] where each pattern is a dictionary with no mention of ".

# Updated Pattern List:""" 