import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Safely download NLTK data
def _init_nltk():
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng"
    }
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True)

_init_nltk()

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def preprocess(self, text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
        """
        Cleans and normalizes text according to the following operations:
        - Lowercasing
        - Whitespace normalization
        - Special character, punctuation, and number removal
        - Tokenization
        - Stopword removal (optional)
        - Lemmatization (optional)
        """
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # 1. Lowercasing & Whitespace Normalization
            text = text.lower().strip()
            text = re.sub(r"\s+", " ", text)
            
            # 2. Number & Special Character / Punctuation Removal
            # Retain only alphabetic characters and spaces
            text = re.sub(r"[^a-zA-Z\s]", "", text)
            
            # 3. Tokenization
            words = word_tokenize(text)
            
            # 4. Stopword Removal
            if remove_stopwords:
                words = [w for w in words if w not in self.stop_words]
                
            # 5. Lemmatization
            if lemmatize:
                words = [self.lemmatizer.lemmatize(w) for w in words]
                
            return " ".join(words)
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            return ""

    def get_sentences(self, text: str) -> list:
        """Splits raw text into a list of sentences."""
        if not text or not isinstance(text, str):
            return []
        try:
            return sent_tokenize(text)
        except Exception as e:
            logger.error(f"Error during sentence tokenization: {str(e)}")
            return [text]

    def get_words(self, text: str) -> list:
        """Tokenizes raw text into words without removing punctuation or stop words."""
        if not text or not isinstance(text, str):
            return []
        try:
            # Basic cleaning of multiple spaces
            text = re.sub(r"\s+", " ", text.strip())
            return word_tokenize(text)
        except Exception as e:
            logger.error(f"Error during word tokenization: {str(e)}")
            return text.split()

# Create a singleton instance for import
preprocessor = TextPreprocessor()

def clean_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    return preprocessor.preprocess(text, remove_stopwords, lemmatize)

def tokenize_sentences(text: str) -> list:
    return preprocessor.get_sentences(text)

def tokenize_words(text: str) -> list:
    return preprocessor.get_words(text)
