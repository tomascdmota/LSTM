import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_trained_model():
    """Load the trained LSTM model from disk."""
    model = load_model('saved_models/lstm_model.keras')
    return model

def load_tokenizers():
    """Load the saved input and target tokenizers."""
    with open('saved_models/input_tokenizer.pkl', 'rb') as f:
        input_tokenizer = pickle.load(f)
    with open('saved_models/target_tokenizer.pkl', 'rb') as f:
        target_tokenizer = pickle.load(f)
    return input_tokenizer, target_tokenizer

def generate_output(input_sentence, model, input_tokenizer, target_tokenizer, max_input_len, max_target_len):
    """Generate output sentence from input sentence using the trained model."""
    input_sequence = input_tokenizer.texts_to_sequences([input_sentence])
    input_sequence = pad_sequences(input_sequence, maxlen=max_input_len, padding='post')

    # Initialize decoder input
    target_seq = np.zeros((1, 1))
    decoded_sentence = ''
    
    # Encode input sentence and generate predictions token by token
    states_value = model.predict(input_sequence)
    stop_condition = False

    while not stop_condition:
        output_tokens, h, c = model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = target_tokenizer.index_word.get(sampled_token_index, '')

        decoded_sentence += ' ' + sampled_word

        if len(decoded_sentence.split()) >= max_target_len or sampled_word == '':
            stop_condition = True

    return decoded_sentence.strip()
