 # Utility functions (e.g., tokenization, padding)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_tokenizer(texts, num_words):
    """Create a tokenizer for a given set of texts."""
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    return tokenizer

def pad_text_sequences(sequences, max_len):
    """Pad sequences to a fixed length."""
    return pad_sequences(sequences, maxlen=max_len, padding='post')
