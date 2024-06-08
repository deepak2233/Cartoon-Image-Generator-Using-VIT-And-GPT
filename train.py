import argparse
import yaml
import os
import warnings
import tensorflow as tf
import numpy as np
from data.load_data import load_caption_data
from eda.eda import plot_caption_length_distribution, plot_common_words, plot_top_ngrams, plot_wordcloud, display_sample_images
from utils.text_processing import preprocess_text
from utils.image_processing import preprocess_and_extract_features, preprocess_and_extract_features_vit
from models.build_model import build_model
from models.train_model import train_model, data_generator
from models.evaluate_model import evaluate_model

def main(config, model_type):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings("ignore")
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    df_train, df_val, df_test = load_caption_data()
    plot_caption_length_distribution(df_train)
    plot_common_words(df_train)
    plot_top_ngrams(df_train)
    plot_wordcloud(df_train)
    display_sample_images(df_train)

    tokenizer, train_sequences, val_sequences, test_sequences, vocab_size, max_length = preprocess_text(df_train, df_val, df_test)

    if model_type == 'vit_gpt2':
        train_features = preprocess_and_extract_features_vit(df_train['image'])
        val_features = preprocess_and_extract_features_vit(df_val['image'])
        test_features = preprocess_and_extract_features_vit(df_test['image'])
    else:
        train_features = preprocess_and_extract_features(df_train['image'])
        val_features = preprocess_and_extract_features(df_val['image'])
        test_features = preprocess_and_extract_features(df_test['image'])

    batch_size = config['batch_size']

    with strategy.scope():
        model = build_model(max_length, vocab_size, config['embedding_dim'], config['units'], train_features.shape[1], model_type)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam')

    train_dataset = tf.data.Dataset.from_generator(
        data_generator,
        args=[train_features, train_sequences, batch_size, vocab_size],
        output_signature=(
            (
                tf.TensorSpec(shape=(None, train_features.shape[1]), dtype=tf.float32),
                tf.TensorSpec(shape=(None, max_length), dtype=tf.int32)
            ),
            tf.TensorSpec(shape=(None, max_length, vocab_size), dtype=tf.float32)
        )
    )
    val_dataset = tf.data.Dataset.from_generator(
        data_generator,
        args=[val_features, val_sequences, batch_size, vocab_size],
        output_signature=(
            (
                tf.TensorSpec(shape=(None, val_features.shape[1]), dtype=tf.float32),
                tf.TensorSpec(shape=(None, max_length), dtype=tf.int32)
            ),
            tf.TensorSpec(shape=(None, max_length, vocab_size), dtype=tf.float32)
        )
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config['epochs']
    )

    bleu_score, rouge_score = evaluate_model(model, test_features, df_test, tokenizer, max_length, model_type)

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
