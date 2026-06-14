import logging

logger = logging.getLogger(__name__)

def generate_personalized_feedback(features: dict) -> dict:
    """
    Analyzes essay features and generates a structured feedback report.
    Returns:
        dict: {
            "strengths": [list of strings],
            "weaknesses": [list of strings],
            "suggestions": [list of strings],
            "subscores": {
                "grammar": float (1-10),
                "content": float (1-10),
                "coherence": float (1-10),
                "vocabulary": float (1-10)
            }
        }
    """
    strengths = []
    weaknesses = []
    suggestions = []
    
    # Extract feature metrics
    word_count = features.get("word_count", 0)
    spelling_errors = features.get("spelling_errors", 0)
    grammar_errors = features.get("grammar_errors", 0)
    error_density = features.get("error_density", 0.0)
    ttr = features.get("type_token_ratio", 0.0)
    lexical_diversity = features.get("lexical_diversity", 0.0)
    transition_count = features.get("transition_count", 0)
    logical_flow = features.get("logical_flow_score", 5.0)
    paragraph_count = features.get("paragraph_count", 1)
    flesch_ease = features.get("flesch_reading_ease", 60.0)
    tfidf_sim = features.get("tfidf_similarity_prompt", 0.0)
    advanced_vocab = features.get("advanced_vocab_ratio", 0.0)
    
    # ------------------------------------------
    # 1. Grammar & Spelling Assessment (Subscore)
    # ------------------------------------------
    # Errors density mapping to 1-10 scale
    # 0.0 errors -> 10.0 score. 0.1 errors -> 2.0 score.
    grammar_subscore = max(1.0, min(10.0, 10.0 - (error_density * 80.0)))
    
    if error_density < 0.02:
        strengths.append("Excellent mechanics! The essay demonstrates high grammatical accuracy and minimal spelling mistakes.")
    elif error_density > 0.06:
        weaknesses.append("Frequent spelling or grammatical errors distract from the readability of the essay.")
        suggestions.append("Perform a careful proofreading pass or use automated checkers to capture basic spelling mistakes.")
        suggestions.append("Focus on subject-verb agreement and proper capitalization rules.")
    else:
        strengths.append("Good baseline control over spelling and basic sentence structures, with only minor errors.")
        suggestions.append("Review punctuation and run-on sentences to polish structural flow.")

    # ------------------------------------------
    # 2. Vocabulary Richness Assessment (Subscore)
    # ------------------------------------------
    # Lexical diversity and TTR mapping to 1-10 scale
    # Typically, lexical diversity (Guiraud's) ranges from 3 to 10 for school essays
    vocab_subscore = max(1.0, min(10.0, (lexical_diversity - 2.0) * 1.25 + (advanced_vocab * 15.0)))
    
    if lexical_diversity > 6.0 and advanced_vocab > 0.15:
        strengths.append("Sophisticated vocabulary! The essay features a wide range of words and avoids repetitive phrasing.")
    elif lexical_diversity < 4.0:
        weaknesses.append("Repetitive vocabulary usage. The essay relies heavily on simple, high-frequency words.")
        suggestions.append("Incorporate synonyms and context-specific terminology to express your arguments more precisely.")
        suggestions.append("Vary your descriptor words; avoid repeating verbs and adjectives (e.g. use 'detrimental' instead of repeating 'bad').")
    else:
        strengths.append("Competent and clear vocabulary that adequately conveys the intended meanings.")
        suggestions.append("Try integrating a few more advanced academic vocabulary words to elevate the tone.")

    # ------------------------------------------
    # 3. Coherence & Structure Assessment (Subscore)
    # ------------------------------------------
    # Logical flow and paragraph structure mapping
    coherence_subscore = max(1.0, min(10.0, logical_flow + (paragraph_count * 0.5)))
    
    if paragraph_count < 3:
        weaknesses.append("Under-structured essay. The text needs to be organized into distinct introductory, body, and concluding paragraphs.")
        suggestions.append("Adopt a standard multi-paragraph structure (Introduction, Body Paragraphs, and Conclusion).")
        
    if transition_count < 3:
        weaknesses.append("Weak flow. Sentences are choppy and transition abruptly between arguments.")
        suggestions.append("Use transition words (e.g., 'however', 'consequently', 'furthermore') to link ideas logically.")
    elif transition_count > 6:
        strengths.append("Excellent logical scaffolding! Transition markers help guide the reader through the argumentation.")
        
    if 40.0 <= flesch_ease <= 70.0:
        strengths.append("Highly readable sentence structure. The complexity is well-calibrated for an academic essay.")
    elif flesch_ease < 30.0:
        weaknesses.append("Sentence complexity is excessively high, which can make the arguments hard to follow.")
        suggestions.append("Break down overly long or run-on sentences into shorter, punchy assertions.")

    # ------------------------------------------
    # 4. Content Relevance Assessment (Subscore)
    # ------------------------------------------
    # TF-IDF similarity mapping. Normal range is 0.1 to 0.6
    content_subscore = max(1.0, min(10.0, 3.0 + (tfidf_sim * 12.0) + (word_count / 150.0)))
    
    if word_count < 150:
        weaknesses.append("Insufficient development. The essay is too short to explore the topic thoroughly.")
        suggestions.append("Expand your arguments by providing concrete examples, statistics, or logical explanations.")
    elif word_count > 350:
        strengths.append("Rich content development. The essay shows detail and depth of thought.")
        
    if tfidf_sim > 0.35:
        strengths.append("Strong alignment with the prompt topic. Main arguments remain highly relevant.")
    elif tfidf_sim < 0.15 and tfidf_sim > 0.0:
        weaknesses.append("Lacks thematic alignment. The essay drifts off-topic or doesn't reference core concepts of the prompt.")
        suggestions.append("Incorporate keywords from the prompt directly into your thesis statements to preserve topical focus.")

    # Cap individual subscores to standard range 1.0 to 10.0
    scores = {
        "grammar": round(grammar_subscore, 1),
        "content": round(content_subscore, 1),
        "coherence": round(coherence_subscore, 1),
        "vocabulary": round(vocab_subscore, 1)
    }
    
    return {
        "strengths": strengths if strengths else ["Basic sentence construction is coherent and easy to read."],
        "weaknesses": weaknesses if weaknesses else ["No major mechanical or lexical weaknesses detected."],
        "suggestions": suggestions if suggestions else ["Continue writing essays of this length, practicing more complex rhetorical structures."],
        "subscores": scores
    }
