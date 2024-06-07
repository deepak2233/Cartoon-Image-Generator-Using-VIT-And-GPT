import argparse
import yaml
import tensorflow as tf
from data.load_data import load_caption_data
from eda.eda import plot_caption_length_distribution, plot_common_words, plot_top_ngrams, plot_wordcloud, display_sample_images
from utils.text_processing import preprocess_text
from utils.image_processing import preprocess_and_extract_features
from models.build_model import build_model
from models.train_model import train_model
from models.evaluate_model import evaluate_model

def main(config, model_type):
    # Enable GPU dynamic memory allocation
    print('##############Cartoon Caption Project#############')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('############gpus######################')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
      print("No GPU was found")

    print()
    print('############### Load the Datasets ####################')
    df_train, df_val, df_test = load_caption_data()
    print(df_train.head(2))

    # EDA
    print('############### Exploratory Data Analysis ####################')
    plot_caption_length_distribution(df_train)
    plot_common_words(df_train)
    plot_top_ngrams(df_train)
    plot_wordcloud(df_train)
    display_sample_images(df_train)
    print('############### End of Exploratory Data Analysis ####################')

    print('############### Text Preprocessing ####################')
    # Preprocess text
    tokenizer, train_sequences, val_sequences, test_sequences, vocab_size, max_length = preprocess_text(df_train, df_val, df_test)

    print('############### Images Preprocessing ####################')
    # Preprocess images
    train_features = preprocess_and_extract_features(df_train['image'])
    val_features = preprocess_and_extract_features(df_val['image'])
    test_features = preprocess_and_extract_features(df_test['image'])

    print('############### Model Building ####################')
    # Build model
    model = build_model(max_length, vocab_size, config['embedding_dim'], config['units'], train_features.shape[1], model_type)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    print('############### Model Training ####################')
    # Train model
    history = train_model(model, train_features, train_sequences, val_features, val_sequences, vocab_size, config['batch_size'], config['epochs'], model_type)

    print('############### Model Evaluation ####################')
    # Evaluate model
    bleu_score, rouge_score = evaluate_model(model, test_features, df_test, tokenizer, max_length)

    # Save model
    model.save(config['model_save_path'])
    print(f'Model saved at {config["model_save_path"]}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config.yaml")
    parser.add_argument('--model_type', type=str, required=True, help="Model type: lstm_cnn or vit_gpt2")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(config, args.model_type)

