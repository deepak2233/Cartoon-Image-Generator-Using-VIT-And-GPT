import tensorflow as tf
from transformers import AdamW
import torch
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf

def data_generator(features, sequences, batch_size, vocab_size):
    batch_features = []
    batch_labels = []
    while True:
        for i in range(len(features)):
            batch_features.append(features[i])
            batch_labels.append(tf.keras.utils.to_categorical(sequences[i], num_classes=vocab_size))
            if len(batch_features) == batch_size:
                yield ([np.array(batch_features), np.array(sequences[i:i+batch_size])], np.array(batch_labels))
                batch_features = []
                batch_labels = []
        if batch_features:  # Yield remaining data if not empty
            yield ([np.array(batch_features), np.array(sequences[len(features)-len(batch_features):])], np.array(batch_labels))


def create_tf_data_generator(features, sequences, batch_size, vocab_size, max_length):
    output_signature = (
        (tf.TensorSpec(shape=(None, features.shape[1]), dtype=tf.float32),
         tf.TensorSpec(shape=(None, max_length), dtype=tf.int32)),
        tf.TensorSpec(shape=(None, max_length, vocab_size), dtype=tf.float32)
    )
    generator = lambda: data_generator(features, sequences, batch_size, vocab_size, max_length)
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)

def train_lstm_cnn_model(model, train_generator, val_generator, vocab_size, batch_size, epochs):
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )
    return history

def train_vit_gpt2_model(train_loader, val_loader, gpt_model, tokenizer, epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt_model.to(device)
    optimizer = AdamW(gpt_model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        gpt_model.train()
        train_loss = 0

        for image_features, captions, masks in train_loader:
            image_features = image_features.to(device)
            captions = captions.to(device)
            masks = masks.to(device)

            outputs = gpt_model(input_ids=captions, attention_mask=masks, labels=captions)
            loss = outputs.loss
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader)}")

        # Validation
        gpt_model.eval()
        val_loss = 0

        with torch.no_grad():
            for image_features, captions, masks in val_loader:
                image_features = image_features.to(device)
                captions = captions.to(device)
                masks = masks.to(device)

                outputs = gpt_model(input_ids=captions, attention_mask=masks, labels=captions)
                val_loss += outputs.loss.item()

        print(f"Validation Loss: {val_loss / len(val_loader)}")

def train_model(model, train_generator, val_generator, vocab_size, batch_size, epochs, model_type, **kwargs):
    if model_type == 'lstm_cnn':
        return train_lstm_cnn_model(model, train_generator, val_generator, vocab_size, batch_size, epochs)
    elif model_type == 'vit_gpt2':
        return train_vit_gpt2_model(
            kwargs['train_loader'], 
            kwargs['val_loader'], 
            model, 
            kwargs['tokenizer'], 
            epochs, 
            kwargs['learning_rate']
        )
    else:
        raise ValueError("Unknown model type")
