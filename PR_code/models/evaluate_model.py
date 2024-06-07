from sklearn.metrics import accuracy_score, f1_score
import torch

def evaluate_lstm_cnn_model(model, test_features, df_test, tokenizer, max_length):
    predictions = model.predict(test_features)
    predicted_sequences = tf.argmax(predictions, axis=-1).numpy()
    true_sequences = df_test['image_description']

    # Convert sequences back to text for BLEU and ROUGE score calculation
    predicted_texts = tokenizer.sequences_to_texts(predicted_sequences)
    true_texts = tokenizer.sequences_to_texts(true_sequences)

    # Calculate BLEU and ROUGE scores
    bleu_score = calculate_bleu_score(true_texts, predicted_texts)
    rouge_score = calculate_rouge_score(true_texts, predicted_texts)

    return bleu_score, rouge_score

def evaluate_vit_gpt2_model(gpt_model, test_features, df_test, tokenizer, max_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt_model.to(device)
    gpt_model.eval()

    predictions = []

    with torch.no_grad():
        for image_features in test_features:
            image_features = image_features.to(device)
            input_ids = torch.tensor(tokenizer.encode("<BOS>")).unsqueeze(0).to(device)

            output = gpt_model.generate(input_ids=input_ids, max_length=max_length, num_beams=5, early_stopping=True)
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)
            predictions.append(prediction)

    true_texts = df_test['image_description']

    # Calculate BLEU and ROUGE scores
    bleu_score = calculate_bleu_score(true_texts, predictions)
    rouge_score = calculate_rouge_score(true_texts, predictions)

    return bleu_score, rouge_score

def evaluate_model(model, test_features, df_test, tokenizer, max_length, model_type):
    if model_type == 'lstm_cnn':
        return evaluate_lstm_cnn_model(model, test_features, df_test, tokenizer, max_length)
    elif model_type == 'vit_gpt2':
        return evaluate_vit_gpt2_model(model, test_features, df_test, tokenizer, max_length)
    else:
        raise ValueError("Unknown model type")
