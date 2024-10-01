# Defines the LSTM model architecture
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

def build_lstm_model(input_vocab_size, target_vocab_size, max_input_len, max_target_len, latent_dim=256):
    """Build the LSTM model for sequence-to-sequence learning."""
    # Encoder
    encoder_inputs = Input(shape=(max_input_len,))
    encoder_embedding = Embedding(input_vocab_size, latent_dim)(encoder_inputs)
    encoder_lstm, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(max_target_len,))
    decoder_embedding = Embedding(target_vocab_size, latent_dim)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(target_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
