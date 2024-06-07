from transformers import AdamW
import torch

def train_lstm_cnn_model(model, train_features, train_sequences, val_features, val_sequences, vocab_size, batch_size, epochs):
    history = model.fit(
        [train_features, train_sequences], 
        tf.keras.utils.to_categorical(train_sequences, num_classes=vocab_size),
        validation_data=([val_features, val_sequences], tf.keras.utils.to_categorical(val_sequences, num_classes=vocab_size)),
        batch_size=batch_size,
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

def train_model(model, train_features, train_sequences, val_features, val_sequences, vocab_size, batch_size, epochs, model_type):
    if model_type == 'lstm_cnn':
        return train_lstm_cnn_model(model, train_features, train_sequences, val_features, val_sequences, vocab_size, batch_size, epochs)
    elif model_type == 'vit_gpt2':
        # Assuming DataLoader objects `train_loader` and `val_loader` are created elsewhere for ViT-GPT2 training
        return train_vit_gpt2_model(train_loader, val_loader, model, tokenizer, epochs, learning_rate)
    else:
        raise ValueError("Unknown model type")
