from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def fine_tune_model():
    # Load the pre-trained model
    model = load_model('saved_models/lstm_model.keras')

    # Optionally freeze some layers if needed
    # for layer in model.layers[:n]:  # Freeze the first 'n' layers
    #     layer.trainable = False

    # Load and preprocess the new data for fine-tuning
    input_texts, target_texts = load_data('data/dataset.csv')
    max_input_len = max([len(text.split()) for text in input_texts])
    max_target_len = max([len(text.split()) for text in target_texts])

    input_sequences, target_sequences, input_tokenizer, target_tokenizer = preprocess_data(
        input_texts, target_texts, num_words=20000, max_len=max_input_len)

    target_sequences = np.expand_dims(target_sequences, axis=-1)

    # Define a new checkpoint for the fine-tuned model
    checkpoint = ModelCheckpoint('saved_models/lstm_model_finetuned.keras', monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    # Fine-tune the model
    history = model.fit(
        [input_sequences, target_sequences[:, :-1]], target_sequences[:, 1:],
        batch_size=64,
        epochs=5,  # Typically, you use fewer epochs for fine-tuning
        validation_split=0.1,
        callbacks=[checkpoint, early_stopping]
    )

if __name__ == '__main__':
    fine_tune_model()
