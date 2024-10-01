from src.train import train_model
from src.inference import generate_output, load_trained_model, load_tokenizers

if __name__ == '__main__':
    # Uncomment the following line to train the model
    train_model()

    # For inference
    # Load the trained model and tokenizers
    model = load_trained_model()  # Load the trained model
    input_tokenizer, target_tokenizer = load_tokenizers()  # Load tokenizers

    # Define the maximum input and target lengths (set based on your dataset)
    max_input_len = 120  # Example value, you should set this based on your data
    max_target_len = 50   # Example value, set this based on your data

    # Provide an input sentence for inference
    input_sentence = "Pesquisar documentos de Luis de Camoes, desde 1990"
    
    # Generate and print output
    output_sentence = generate_output(input_sentence, model, input_tokenizer, target_tokenizer, max_input_len, max_target_len)
    print(f"Generated Output: {output_sentence}")
