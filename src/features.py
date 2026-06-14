import re
import math
import logging
import numpy as np
import pandas as pd
from textblob import TextBlob
import textstat
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import joblib

logger = logging.getLogger(__name__)

# Standard English transition words list
TRANSITION_WORDS = {
    "however", "therefore", "furthermore", "moreover", "consequently", 
    "additionally", "instead", "likewise", "similarly", "subsequently", 
    "conversely", "meanwhile", "nevertheless", "nonetheless", "hence", 
    "thus", "otherwise", "besides", "further", "accordingly", "indeed", 
    "contrarily", "ultimately", "meantime", "specifically", "namely", 
    "instead", "finally", "initially", "primarily", "secondly", "thirdly"
}

class FeatureExtractor:
    def __init__(self, vectorizer_path=None, lda_path=None):
        self.vectorizer = None
        self.lda = None
        
        # Load pre-trained models if paths are provided
        if vectorizer_path:
            try:
                self.vectorizer = joblib.load(vectorizer_path)
            except Exception as e:
                logger.warning(f"Could not load vectorizer: {e}")
                
        if lda_path:
            try:
                self.lda = joblib.load(lda_path)
            except Exception as e:
                logger.warning(f"Could not load LDA model: {e}")

    def fit_tfidf_lda(self, texts: list, n_topics: int = 5):
        """Fits TF-IDF and LDA on a training text corpus."""
        logger.info("Fitting TF-IDF Vectorizer...")
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        logger.info("Fitting LDA Topic Model...")
        self.lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        self.lda.fit(tfidf_matrix)
        
        return self

    def save_models(self, vectorizer_path: str, lda_path: str):
        """Saves fitted models."""
        if self.vectorizer:
            joblib.dump(self.vectorizer, vectorizer_path)
        if self.lda:
            joblib.dump(self.lda, lda_path)

    def extract_features(self, text: str, prompt_text: str = None) -> dict:
        """
        Extracts all features for a single essay.
        Returns a dictionary of feature names and values.
        """
        features = {}
        
        # Guard clause for empty inputs
        if not text or len(text.strip()) == 0:
            text = "Placeholder text for empty essay input."
            
        # Basic text stats
        blob = TextBlob(text)
        words_raw = text.split()
        char_count = len(text)
        word_count = len(words_raw)
        
        # Paragraphs
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        paragraph_count = max(1, len(paragraphs))
        
        # Sentences
        sentences = [str(s) for s in blob.sentences]
        sentence_count = max(1, len(sentences))
        avg_sentence_len = word_count / sentence_count
        
        features["char_count"] = char_count
        features["word_count"] = word_count
        features["paragraph_count"] = paragraph_count
        features["sentence_count"] = sentence_count
        features["avg_sentence_length"] = avg_sentence_len
        
        # --- 1. Content Relevance Features ---
        features["tfidf_similarity_prompt"] = 0.0
        if prompt_text and self.vectorizer:
            try:
                vectors = self.vectorizer.transform([text, prompt_text])
                features["tfidf_similarity_prompt"] = cosine_similarity(vectors[0], vectors[1])[0][0]
            except Exception as e:
                logger.error(f"Error calculating TF-IDF prompt similarity: {e}")
                
        # LDA Topics
        for i in range(5):
            features[f"lda_topic_{i}"] = 0.0
            
        if self.vectorizer and self.lda:
            try:
                tfidf_vec = self.vectorizer.transform([text])
                lda_dist = self.lda.transform(tfidf_vec)[0]
                for i in range(min(len(lda_dist), 5)):
                    features[f"lda_topic_{i}"] = lda_dist[i]
            except Exception as e:
                logger.error(f"Error calculating LDA topics: {e}")
                
        # --- 2. Grammar & Spelling Features ---
        # Spelling errors estimation using TextBlob
        spell_errors = 0
        for word in blob.words:
            # Check length to avoid checking short noise or abbreviations
            if len(word) > 3:
                # TextBlob word spell check returns list of (word, confidence)
                # If confidence is 1.0, it is likely correct. Otherwise, we verify.
                suggestions = word.spellcheck()
                if suggestions and suggestions[0][0] != word and suggestions[0][1] > 0.8:
                    spell_errors += 1
                    
        # Simple Grammar heuristics (double modals, double negatives, subject-verb markers)
        grammar_errors = 0
        text_lower = text.lower()
        # Common grammar flaws check using regex
        grammar_patterns = [
            r"\b(can not|cannot)\b",
            r"\b(could of|would of|should of|might of)\b",
            r"\b(a other)\b",
            r"\b(had ran|had ate|had went|had wrote)\b",
            r"\b(more better|most best|more taller)\b",
            r"\b(double negative rules like: did not do nothing)\b", # illustrative check
            r"\b(dont|cant|wont|isnt|arent|wasnt|werent)\b",  # missing apostrophes
        ]
        for pattern in grammar_patterns:
            grammar_errors += len(re.findall(pattern, text_lower))
            
        error_density = (spell_errors + grammar_errors) / max(1, word_count)
        
        features["spelling_errors"] = spell_errors
        features["grammar_errors"] = grammar_errors
        features["error_density"] = error_density
        
        # --- 3. Coherence & Structure Features ---
        # Transition words
        transition_count = sum(1 for w in words_raw if w.lower() in TRANSITION_WORDS)
        transition_freq = transition_count / max(1, sentence_count)
        features["transition_count"] = transition_count
        features["transition_frequency"] = transition_freq
        
        # Readability Scores
        try:
            features["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
            features["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(text)
            features["gunning_fog"] = textstat.gunning_fog(text)
            features["dale_chall"] = textstat.dale_chall_readability_score(text)
        except Exception as e:
            logger.warning(f"Error calculating textstat readability: {e}")
            features["flesch_reading_ease"] = 60.0
            features["flesch_kincaid_grade"] = 8.0
            features["gunning_fog"] = 8.0
            features["dale_chall"] = 6.0
            
        # Logical Flow Score (Heuristic: Paragraph distribution equality, sentence transitions, sentence lengths variability)
        sent_lengths = [len(s.split()) for s in sentences]
        sent_length_std = np.std(sent_lengths) if len(sent_lengths) > 1 else 0.0
        
        para_lengths = [len(p.split()) for p in paragraphs]
        para_length_std = np.std(para_lengths) if len(para_lengths) > 1 else 0.0
        
        # Logic flow score based on variance of sentence lengths (engaging text varies sentence length)
        # and moderate transitions
        logical_flow = min(10.0, max(1.0, 3.0 + (sent_length_std / 3.0) + (transition_freq * 10.0) - (para_length_std / 100.0)))
        features["logical_flow_score"] = logical_flow
        
        # --- 4. Vocabulary Richness Features ---
        words_clean = [re.sub(r"[^\w]", "", w.lower()) for w in words_raw if re.sub(r"[^\w]", "", w)]
        unique_word_count = len(set(words_clean))
        ttr = unique_word_count / max(1, len(words_clean))
        
        # Lexical Diversity (using clean words)
        lexical_diversity = unique_word_count / max(1, math.sqrt(len(words_clean)))  # Guiraud's Index
        
        # Average word length
        avg_word_len = sum(len(w) for w in words_clean) / max(1, len(words_clean))
        
        # Advanced vocabulary (complex words with 3+ syllables)
        try:
            complex_words = textstat.difficult_words(text)
            advanced_vocab_ratio = complex_words / max(1, len(words_clean))
        except Exception:
            advanced_vocab_ratio = sum(1 for w in words_clean if len(w) > 7) / max(1, len(words_clean))
            
        features["unique_word_count"] = unique_word_count
        features["type_token_ratio"] = ttr
        features["lexical_diversity"] = lexical_diversity
        features["advanced_vocab_ratio"] = advanced_vocab_ratio
        features["avg_word_length"] = avg_word_len
        
        # --- 5. Additional Features ---
        # Sentiment
        features["sentiment_polarity"] = blob.sentiment.polarity
        features["sentiment_subjectivity"] = blob.sentiment.subjectivity
        
        # POS Tag distribution & Named Entities (using NLTK)
        try:
            pos_tags = nltk.pos_tag(words_raw)
            noun_count = sum(1 for w, t in pos_tags if t.startswith("NN"))
            verb_count = sum(1 for w, t in pos_tags if t.startswith("VB"))
            adj_count = sum(1 for w, t in pos_tags if t.startswith("JJ"))
            adv_count = sum(1 for w, t in pos_tags if t.startswith("RB"))
            
            features["noun_ratio"] = noun_count / max(1, word_count)
            features["verb_ratio"] = verb_count / max(1, word_count)
            features["adj_ratio"] = adj_count / max(1, word_count)
            features["adv_ratio"] = adv_count / max(1, word_count)
            
            # Proper nouns are a solid proxy for Named Entities
            named_entities = sum(1 for w, t in pos_tags if t in ("NNP", "NNPS"))
            features["named_entity_count"] = named_entities
        except Exception as e:
            logger.warning(f"Error computing POS tags: {e}")
            features["noun_ratio"] = 0.3
            features["verb_ratio"] = 0.15
            features["adj_ratio"] = 0.1
            features["adv_ratio"] = 0.05
            features["named_entity_count"] = sum(1 for w in words_raw if w and w[0].isupper() and not w.lower() in TRANSITION_WORDS)
            
        return features

    def extract_features_df(self, texts: list, prompts: list = None) -> pd.DataFrame:
        """Extracts features for a list of texts and returns a Pandas DataFrame."""
        rows = []
        for i, text in enumerate(texts):
            prompt = prompts[i] if prompts else None
            rows.append(self.extract_features(text, prompt))
        return pd.DataFrame(rows)
