import argparse
import yaml
import os
import warnings
import tensorflow as tf
import tf_keras  # Ensure compatible version of tf-keras is installed

from data.load_data import load_caption_data
from eda.eda import plot_caption_length_distribution, plot_common_words, plot_top_ngrams, plot_wordcloud, display_sample_images
from utils.text_processing import preprocess_text
from utils.image_processing import preprocess_and_extract_features, preprocess_and_extract_features_vit
from models.build_model import build_model
from models.train_model import train_model, data_generator
from models.evaluate_model import evaluate_model

def main(config, model_type):
    # Suppress TensorFlow info and warning messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Suppress all warnings
    warnings.filterwarnings("ignore")

    # Define the mirrored strategy
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Load dataset
    print('###############Load the Datasets####################')
    df_train, df_val, df_test = load_caption_data()
    
    # EDA
    print('###############Exploratory Data Analysis####################')
    plot_caption_length_distribution(df_train)
    plot_common_words(df_train)
    plot_top_ngrams(df_train)
    plot_wordcloud(df_train)
    display_sample_images(df_train)
    print('###############Text Preprocessing####################')
    
    # Preprocess text
    tokenizer, train_sequences, val_sequences, test_sequences, vocab_size, max_length = preprocess_text(df_train, df_val, df_test)
    
    print('############### Images Preprocessing####################')
    # Preprocess images
    if model_type == 'vit_gpt2':
        train_features = preprocess_and_extract_features_vit(df_train['image'])
        val_features = preprocess_and_extract_features_vit(df_val['image'])
        test_features = preprocess_and_extract_features_vit(df_test['image'])
    else:
        train_features = preprocess_and_extract_features(df_train['image'])
        val_features = preprocess_and_extract_features(df_val['image'])
        test_features = preprocess_and_extract_features(df_test['image'])

    train_generator = data_generator(train_features, train_sequences, config['batch_size'], vocab_size)
    val_generator = data_generator(val_features, val_sequences, config['batch_size'], vocab_size)

    # Build and compile model within the strategy scope
    with strategy.scope():
        print('###############Model Building####################')
        model = build_model(max_length, vocab_size, config['embedding_dim'], config['units'], train_features.shape[1], model_type)
        print('######################MOdel Summary###################')
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam')

    print('###############Model Training####################')
    # Train model
    history = train_model(model, train_generator, val_generator, vocab_size, config['batch_size'], config['epochs'], model_type)

    # Evaluate model
    print('###############Model Evaluating####################')
    bleu_score, rouge_score = evaluate_model(model, test_features, df_test, tokenizer, max_length, model_type)

    print('###############Save the Model####################')
    # Save model
    model.save(config['model_save_path'])
    print(f'Model saved at {config["model_save_path"]}')

if __name__ == "__main__":
    print('###############Cartoon Captioning Generator Project####################')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config.yaml")
    parser.add_argument('--model_type', type=str, required=True, help="Model type: lstm_cnn or vit_gpt2")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(config, args.model_type)
