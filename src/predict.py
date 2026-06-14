import os
import logging
import joblib
import numpy as np
import pandas as pd
from src.features import FeatureExtractor
from src.feedback import generate_personalized_feedback

logger = logging.getLogger(__name__)

# Check if Streamlit is running, to use st.cache_resource
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Function wrapper for cached resource loading
def _load_asset(path, loader_func):
    """Utility to cache asset loading if Streamlit is available."""
    if HAS_STREAMLIT:
        # Define a unique function name inside streamlit cache to avoid collision
        cached_loader = st.cache_resource(loader_func)
        return cached_loader(path)
    else:
        return loader_func(path)

def joblib_loader(path):
    return joblib.load(path)

class ScoringPipeline:
    def __init__(self, models_dir="saved_models"):
        self.models_dir = models_dir
        self.vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
        self.lda_path = os.path.join(models_dir, "lda_model.joblib")
        self.scaler_path = os.path.join(models_dir, "scaler.joblib")
        self.model_path = os.path.join(models_dir, "best_traditional_model.joblib")
        
        # Core assets
        self.fe = None
        self.scaler = None
        self.model = None
        self.fallback_mode = False
        
        self.load_pipeline()

    def load_pipeline(self):
        """Loads vectorizer, LDA, scaler, and regression model."""
        try:
            if os.path.exists(self.vectorizer_path) and os.path.exists(self.lda_path):
                self.fe = FeatureExtractor(self.vectorizer_path, self.lda_path)
            else:
                self.fe = FeatureExtractor() # Initialize empty, will use defaults
                
            if os.path.exists(self.scaler_path):
                self.scaler = _load_asset(self.scaler_path, joblib_loader)
            else:
                self.scaler = None
                
            if os.path.exists(self.model_path):
                self.model = _load_asset(self.model_path, joblib_loader)
            else:
                self.model = None
                self.fallback_mode = True
                logger.warning("No pre-trained model found! Running in Rule-Based Fallback Mode.")
        except Exception as e:
            logger.error(f"Error loading models: {e}. Defaulting to Fallback Mode.")
            self.model = None
            self.scaler = None
            self.fe = FeatureExtractor()
            self.fallback_mode = True

    def calculate_fallback_score(self, features: dict) -> float:
        """Calculates a robust heuristic score based on extracted features."""
        word_count = features.get("word_count", 0)
        spelling_errors = features.get("spelling_errors", 0)
        grammar_errors = features.get("grammar_errors", 0)
        transition_count = features.get("transition_count", 0)
        lexical_diversity = features.get("lexical_diversity", 0.0)
        advanced_vocab = features.get("advanced_vocab_ratio", 0.0)
        paragraph_count = features.get("paragraph_count", 1)
        tfidf_sim = features.get("tfidf_similarity_prompt", 0.0)
        
        # Baseline score starts at 5.0
        score = 5.0
        
        # Word count contribution
        if word_count < 100:
            score -= 1.5
        elif word_count > 300:
            score += 1.5
            
        # Spelling & Grammar penalties
        total_errors = spelling_errors + grammar_errors
        error_penalty = min(3.0, (total_errors / max(1, word_count)) * 25)
        score -= error_penalty
        
        # Structure & Coherence additions
        score += min(1.5, transition_count * 0.2)
        if paragraph_count >= 3:
            score += 0.5
            
        # Vocabulary additions
        score += min(1.5, (lexical_diversity - 3.0) * 0.4)
        score += min(1.0, advanced_vocab * 3.0)
        
        # Cosine prompt similarity addition
        if tfidf_sim > 0.0:
            score += min(1.0, tfidf_sim * 2.0)
            
        # Clamp score between 1.0 and 10.0
        return float(round(max(1.0, min(10.0, score)), 1))

    def predict(self, essay_text: str, prompt_text: str = None) -> dict:
        """
        Runs the full essay scoring process:
        1. Extracts features
        2. Scaler transformation
        3. Predicts overall score
        4. Compiles personalized feedback & subscores
        """
        # Ensure we have fresh/loaded structures
        if self.fe is None or (not self.fallback_mode and self.model is None):
            self.load_pipeline()
            
        # 1. Feature extraction
        features = self.fe.extract_features(essay_text, prompt_text)
        
        # 2. Model prediction
        if self.fallback_mode or self.model is None or self.scaler is None:
            final_score = self.calculate_fallback_score(features)
        else:
            try:
                # Convert features dict to DF matching training format
                df_feat = pd.DataFrame([features])
                
                # Scaler & Predict
                X_scaled = self.scaler.transform(df_feat)
                pred = self.model.predict(X_scaled)[0]
                
                # Clamp score in range 1.0 to 10.0
                final_score = float(round(max(1.0, min(10.0, pred)), 1))
            except Exception as e:
                logger.error(f"Prediction error: {e}. Using fallback score.")
                final_score = self.calculate_fallback_score(features)
                
        # 3. Generate feedback & subscores
        report = generate_personalized_feedback(features)
        
        return {
            "final_score": final_score,
            "category_scores": report["subscores"],
            "feedback": {
                "strengths": report["strengths"],
                "weaknesses": report["weaknesses"],
                "suggestions": report["suggestions"]
            },
            "features": features,
            "is_fallback": self.fallback_mode
        }

# Singleton pipeline instance
pipeline = ScoringPipeline()

def score_essay(essay_text: str, prompt_text: str = None) -> dict:
    return pipeline.predict(essay_text, prompt_text)
