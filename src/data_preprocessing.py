import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(file_path):
    """
    Load the CSV file into input and target sentence pairs,
    ensuring that each line only has two columns (Input and Target).
    """
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Remove rows with more than two columns
    data = remove_extra_columns(data)
    
    return data['Input'].values, data['Target'].values

def preprocess_data(input_texts, target_texts, num_words=20000, max_len=100):
    """Tokenize and pad the input and target sentences."""
    # Tokenizer for input sentences
    input_tokenizer = Tokenizer(num_words=num_words)
    input_tokenizer.fit_on_texts(input_texts)
    input_sequences = input_tokenizer.texts_to_sequences(input_texts)

    # Tokenizer for target sentences
    target_tokenizer = Tokenizer(num_words=num_words)
    target_tokenizer.fit_on_texts(target_texts)
    target_sequences = target_tokenizer.texts_to_sequences(target_texts)

    # Pad the sequences
    input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='post')
    target_sequences = pad_sequences(target_sequences, maxlen=max_len, padding='post')

    return input_sequences, target_sequences, input_tokenizer, target_tokenizer

def get_max_sequence_length(texts):
    """Calculate the maximum length of any sentence in the dataset."""
    return max([len(seq.split()) for seq in texts])

def remove_extra_columns(data):
    """
    Remove any rows that have more than two columns.
    This function ensures that only the 'Input' and 'Target' columns are kept.
    """
    # Count the number of columns in each row
    data_filtered = data.loc[:, data.columns[:2]]  # Keep only the first two columns
    data_filtered = data_filtered.dropna()  # Drop rows where any of the two columns has NaN values
    return data_filtered
