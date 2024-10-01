import numpy as np
import pickle
import os
from src.data_preprocessing import load_data, preprocess_data
from src.model import build_lstm_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def save_tokenizers(input_tokenizer, target_tokenizer):
    """Save the input and target tokenizers to disk."""
    os.makedirs('saved_models', exist_ok=True)  # Ensure directory exists
    with open('saved_models/input_tokenizer.pkl', 'wb') as f:
        pickle.dump(input_tokenizer, f)
    with open('saved_models/target_tokenizer.pkl', 'wb') as f:
        pickle.dump(target_tokenizer, f)

def train_model():
    """Train the LSTM model on the dataset."""
    # Load and preprocess data
    input_texts, target_texts = load_data('data/dataset.csv')
    max_input_len = max([len(text.split()) for text in input_texts])
    max_target_len = max([len(text.split()) for text in target_texts])

    # Preprocess data
    input_sequences, target_sequences, input_tokenizer, target_tokenizer = preprocess_data(
        input_texts, target_texts, num_words=20000, max_len=max_input_len)

    # Save tokenizers after preprocessing
    save_tokenizers(input_tokenizer, target_tokenizer)

    # Shift the target sequences for teacher forcing
    target_sequences = np.expand_dims(target_sequences, axis=-1)

    # Build model
    input_vocab_size = len(input_tokenizer.word_index) + 1
    target_vocab_size = len(target_tokenizer.word_index) + 1
    model = build_lstm_model(input_vocab_size, target_vocab_size, max_input_len, max_target_len)

    # Checkpoint and EarlyStopping
    checkpoint = ModelCheckpoint('saved_models/lstm_model.keras', monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    # Train the model
    history = model.fit(
        [input_sequences, target_sequences[:, :-1]], target_sequences[:, 1:],
        batch_size=64,
        epochs=10,
        validation_split=0.1,
        callbacks=[checkpoint, early_stopping]
    )

if __name__ == '__main__':
    train_model()
