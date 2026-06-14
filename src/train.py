import os
import sys
import argparse
import logging
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
# Try importing XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Try importing LightGBM
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock classes to prevent inheritance errors when PyTorch is not installed
    class nn:
        class Module:
            pass
    class Dataset:
        pass
    class torch:
        pass

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Try importing CatBoost
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available. It will be skipped in comparisons.")

# Create directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)


# ==========================================
# SYNTHETIC DATA GENERATOR
# ==========================================
def generate_synthetic_data(num_samples=200):
    """Generates synthetic essay dataset for immediate training out-of-the-box."""
    logger.info(f"Generating {num_samples} synthetic essays for training...")
    
    prompts = [
        "Should computers replace human teachers in class?",
        "Explain the impact of cell phones on modern communication.",
        "Discuss the importance of reading books in the digital era."
    ]
    
    # Vocabulary lists to build essays
    positive_adjectives = ["amazing", "excellent", "significant", "vital", "crucial", "essential", "advantageous", "dynamic", "profound", "outstanding", "meaningful"]
    negative_adjectives = ["bad", "poor", "detrimental", "harmful", "negative", "substandard", "faulty", "inefficient", "unfavorable"]
    academic_nouns = ["education", "paradigm", "infrastructure", "methodology", "cognitive development", "interpersonal skills", "social dynamics", "connectivity", "intellectual growth"]
    transitions = ["furthermore", "however", "consequently", "in addition", "therefore", "nevertheless", "on the other hand", "specifically", "as a result"]
    
    essays = []
    scores = []
    prompt_ids = []
    prompt_texts = []
    
    for i in range(num_samples):
        # Determine score class: 1 (Poor), 2 (Medium), 3 (Excellent)
        quality = random.choices(["poor", "medium", "excellent"], weights=[0.25, 0.50, 0.25])[0]
        p_idx = random.randint(0, len(prompts) - 1)
        prompt = prompts[p_idx]
        
        # Build paragraph structures
        paragraphs = []
        if quality == "poor":
            # 1 to 2 paragraphs, short, full of spelling mistakes, basic vocabulary
            num_paras = random.randint(1, 2)
            for _ in range(num_paras):
                sentences = [
                    f"Computers are {random.choice(negative_adjectives)} for learning in school.",
                    f"The teacher is {random.choice(positive_adjectives)} but computer is just a screen.",
                    "Many student dont like it because it is boring and hard to understand.",
                    "Also they make a lot of mistake and can not ask question to anyone.",
                    "Cell phones are bad too, people talk to much on them."
                ]
                # Introduce spelling/grammar errors
                raw_para = " ".join(sentences)
                raw_para = raw_para.replace("don't", "dont").replace("cannot", "can not").replace("question", "questin").replace("boring", "boreing")
                paragraphs.append(raw_para)
            score = random.uniform(2.0, 4.5)
            
        elif quality == "medium":
            # 3 paragraphs, medium length, some transitions, fewer errors
            num_paras = 3
            for p_num in range(num_paras):
                if p_num == 0:
                    sentences = [
                        f"In modern times, the role of technology is becoming more {random.choice(positive_adjectives)}.",
                        f"Some people think computers should replace teachers because of speed and efficiency.",
                        "However, I believe that human teachers are still needed in classrooms."
                    ]
                elif p_num == 1:
                    sentences = [
                        f"Computers can provide a lot of {random.choice(academic_nouns)} resources and information.",
                        f"They help students learn at their own pace, which is {random.choice(positive_adjectives)}.",
                        f"On the other hand, computers lack emotional connection and empathy."
                    ]
                else:
                    sentences = [
                        f"Therefore, a balanced approach is the {random.choice(positive_adjectives)} solution.",
                        "We should use technology to assist teachers, not replace them entirely.",
                        "In conclusion, human teachers are irreplaceable."
                    ]
                paragraphs.append(" ".join(sentences))
            score = random.uniform(5.0, 7.5)
            
        else:
            # 4 to 5 paragraphs, long, rich vocabulary, transition words, zero spelling errors, advanced readability
            num_paras = random.randint(4, 5)
            for p_num in range(num_paras):
                if p_num == 0:
                    sentences = [
                        f"The debate regarding whether artificial intelligence and computers can replace human educators is of {random.choice(positive_adjectives)} importance.",
                        f"While proponents argue that technology offers personalized {random.choice(academic_nouns)}, it is imperative to examine the cognitive limitations of machines.",
                        "This essay will explore why human connection remains the cornerstone of effective pedagogy."
                    ]
                elif p_num == 1:
                    sentences = [
                        f"{random.choice(transitions).capitalize()}, human teachers foster {random.choice(academic_nouns)} and critical thinking skills.",
                        "Education is not merely the transmission of factual data; rather, it is an interactive process of mentoring.",
                        "A computer algorithm cannot replicate the inspiration a dedicated teacher provides."
                    ]
                elif p_num == 2:
                    sentences = [
                        f"Furthermore, classroom learning teaches students essential social dynamics and collaborative cooperation.",
                        f"When students interact with peers under the guidance of a teacher, they develop empathy and {random.choice(academic_nouns)}.",
                        f"Consequently, replacing teachers with machines would lead to a {random.choice(negative_adjectives)} isolation of young learners."
                    ]
                else:
                    sentences = [
                        f"Ultimately, while technology is a {random.choice(positive_adjectives)} tool for research, the teacher's role is sacred.",
                        "The human element in classroom management, motivation, and ethical guidance cannot be automated.",
                        "To conclude, computers should remain supplementary tools to enrich the classroom rather than replacements."
                    ]
                paragraphs.append(" ".join(sentences))
            score = random.uniform(8.0, 10.0)
            
        essay_text = "\n\n".join(paragraphs)
        essays.append(essay_text)
        scores.append(round(score, 1))
        prompt_ids.append(p_idx + 1)
        prompt_texts.append(prompt)
        
    df = pd.DataFrame({
        "essay_id": list(range(1, num_samples + 1)),
        "essay_set": prompt_ids,
        "essay": essays,
        "prompt": prompt_texts,
        "domain1_score": scores  # Score out of 10
    })
    
    df.to_csv("data/synthetic_essays.csv", index=False)
    logger.info("Saved synthetic essays to data/synthetic_essays.csv")
    return df


# ==========================================
# PYTORCH DEEP LEARNING MODEL DEFINITION
# ==========================================
class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, model_type="LSTM"):
        super(SimpleRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if model_type == "BiLSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_dim * 2, 1)
        elif model_type == "GRU":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=False)
            self.fc = nn.Linear(hidden_dim, 1)
        else: # LSTM
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=False)
            self.fc = nn.Linear(hidden_dim, 1)
            
    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.rnn(embedded)
        # Average pooling over sequence length
        out = torch.mean(out, dim=1)
        out = self.fc(out)
        return out.squeeze()


class EssayDataset(Dataset):
    def __init__(self, texts, scores, word_to_idx, max_len=150):
        self.texts = texts
        self.scores = scores
        self.word_to_idx = word_to_idx
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = text.lower().split()
        
        # Numericalize and pad
        indices = [self.word_to_idx.get(t, 1) for t in tokens] # 1 is UNK
        if len(indices) < self.max_len:
            indices = indices + [0] * (self.max_len - len(indices)) # 0 is PAD
        else:
            indices = indices[:self.max_len]
            
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.scores[idx], dtype=torch.float)


# ==========================================
# TRAINING & EVALUATION PIPELINE
# ==========================================
def train_pipeline(generate_synthetic_flag=False):
    # Load dataset
    if generate_synthetic_flag or not os.path.exists("data/synthetic_essays.csv"):
        df = generate_synthetic_data(250)
    else:
        df = pd.read_csv("data/synthetic_essays.csv")
        logger.info(f"Loaded {len(df)} essays from data/synthetic_essays.csv")
        
    # Pre-fit TF-IDF and LDA Vectorizers using src/features
    from src.features import FeatureExtractor
    
    # Initialize extractor and fit
    logger.info("Initializing Feature Extractor and fitting text structures...")
    fe = FeatureExtractor()
    fe.fit_tfidf_lda(df["essay"].tolist())
    
    # Save vectorizer and LDA to saved_models/
    fe.save_models("saved_models/tfidf_vectorizer.joblib", "saved_models/lda_model.joblib")
    logger.info("Saved vectorizer and LDA model to saved_models/")
    
    # Extract dense features
    logger.info("Extracting handcrafted NLP features from essays...")
    X_features = fe.extract_features_df(df["essay"].tolist(), df["prompt"].tolist())
    y = df["domain1_score"].values
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, "saved_models/scaler.joblib")
    logger.info("Saved feature scaler to saved_models/scaler.joblib")
    
    # ------------------------------------------
    # Traditional Machine Learning models training
    # ------------------------------------------
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(random_state=42),
    }
    
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(random_state=42)
    else:
        logger.warning("XGBoost not available. Skipping in comparisons.")
        
    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMRegressor(random_state=42, verbose=-1)
    else:
        logger.warning("LightGBM not available. Skipping in comparisons.")
        
    if CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostRegressor(random_state=42, verbose=0)
        
    results = {}
    best_model_name = ""
    best_rmse = float("inf")
    best_model = None
    
    logger.info("Starting traditional ML model comparison and cross validation...")
    for name, model in models.items():
        # Hyperparameter tuning for Random Forest / XGBoost as a demonstration
        if name == "Random Forest":
            param_dist = {
                "n_estimators": [50, 100, 150],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5]
            }
            search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=5, cv=3, random_state=42, n_jobs=-1)
            search.fit(X_train_scaled, y_train)
            trained_model = search.best_estimator_
        elif name == "XGBoost":
            param_dist = {
                "n_estimators": [50, 100],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.2]
            }
            search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=5, cv=3, random_state=42, n_jobs=-1)
            search.fit(X_train_scaled, y_train)
            trained_model = search.best_estimator_
        else:
            model.fit(X_train_scaled, y_train)
            trained_model = model
            
        preds = trained_model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
        logger.info(f"{name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model = trained_model
            
    logger.info(f"Selected Best Traditional Model: {best_model_name} (RMSE: {best_rmse:.4f})")
    joblib.dump(best_model, "saved_models/best_traditional_model.joblib")
    logger.info("Saved best traditional model to saved_models/best_traditional_model.joblib")
    
    # ------------------------------------------
    # Deep Learning (LSTM/BiLSTM/GRU) training
    # ------------------------------------------
    dl_results = {}
    if TORCH_AVAILABLE:
        logger.info("Setting up PyTorch Deep Learning pipelines...")
        
        # Build vocabulary from essay corpus
        all_words = " ".join(df["essay"].tolist()).lower().split()
        vocab = sorted(list(set(all_words)))
        word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        for i, w in enumerate(vocab):
            word_to_idx[w] = i + 2
            
        vocab_size = len(word_to_idx)
        embedding_dim = 50
        hidden_dim = 64
        
        joblib.dump(word_to_idx, "saved_models/word_to_idx.joblib")
        
        # Split for deep learning
        train_texts, val_texts, train_scores, val_scores = train_test_split(
            df["essay"].tolist(), df["domain1_score"].tolist(), test_size=0.2, random_state=42
        )
        
        train_dataset = EssayDataset(train_texts, train_scores, word_to_idx)
        val_dataset = EssayDataset(val_texts, val_scores, word_to_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        dl_models = ["LSTM", "BiLSTM", "GRU"]
        
        for m_type in dl_models:
            logger.info(f"Training PyTorch Deep Learning Model: {m_type}...")
            net = SimpleRNNModel(vocab_size, embedding_dim, hidden_dim, model_type=m_type)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(net.parameters(), lr=0.005)
            
            # Training loop
            net.train()
            for epoch in range(10):  # 10 epochs for demonstration
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
            # Evaluation
            net.eval()
            val_preds = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = net(inputs)
                    val_preds.extend(outputs.tolist())
                    
            dl_rmse = np.sqrt(mean_squared_error(val_scores, val_preds))
            dl_mae = mean_absolute_error(val_scores, val_preds)
            dl_r2 = r2_score(val_scores, val_preds)
            dl_results[m_type] = {"RMSE": dl_rmse, "MAE": dl_mae, "R2": dl_r2}
            logger.info(f"{m_type} Evaluation -> RMSE: {dl_rmse:.4f}, MAE: {dl_mae:.4f}, R2: {dl_r2:.4f}")
            
            # Save PyTorch weights
            torch.save(net.state_dict(), f"saved_models/{m_type.lower()}_model.pth")
    else:
        logger.warning("PyTorch not installed. Skipping deep learning recurrent models (LSTM, GRU, BiLSTM) training.")
        
    # DistilBERT Fine-tuning outline documentation in code
    logger.info("DistilBERT model training is outlined in code. Standard weights can be loaded from HuggingFace.")
    
    # Save a configuration summary file
    summary = {
        "best_traditional_model": best_model_name,
        "traditional_results": results,
        "deep_learning_results": dl_results
    }
    joblib.dump(summary, "saved_models/training_summary.joblib")
    logger.info("Saved training summary report to saved_models/training_summary.joblib")
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Automated Essay Scoring Models.")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic data.")
    args = parser.parse_args()
    
    train_pipeline(generate_synthetic_flag=args.generate_synthetic)
