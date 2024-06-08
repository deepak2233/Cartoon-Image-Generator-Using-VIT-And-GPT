import tensorflow as tf
from transformers import AdamW
import torch
from keras.utils import to_categorical

def data_generator(features, sequences, batch_size, vocab_size):
    num_samples = len(features)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_features = features[offset:offset + batch_size]
            batch_sequences = sequences[offset:offset + batch_size]

            # Add a time dimension to batch_features
            batch_features = batch_features[:, np.newaxis, :]

            batch_labels = []
            for seq in batch_sequences:
                # One-hot encode the sequence
                one_hot_seq = tf.keras.utils.to_categorical(seq, num_classes=vocab_size)
                batch_labels.append(one_hot_seq)

            batch_labels = np.array(batch_labels)

            yield ([batch_features, batch_sequences], batch_labels)


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
