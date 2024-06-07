from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from transformers import TFGPT2LMHeadModel
from tensorflow.keras.layers import Concatenate, LSTM, Dense, Embedding, Input, Dropout, Reshape, RepeatVector

import tensorflow as tf


def build_model(max_length, vocab_size, embedding_dim, units, feature_shape, model_type):

    if model_type =='lstm_cnn':
        # Image input
        image_input = Input(shape=(feature_shape,))
        
        # Text input
        text_input = Input(shape=(max_length,))
        
        # Reshape image features to match the shape of text embeddings
        image_features = Reshape((1, feature_shape))(image_input)
        
        # Repeat image features to match the sequence length
        image_features = LSTM(units, return_sequences=False)(image_features)
        image_features = RepeatVector(max_length)(image_features)
        
        # Text embedding
        text_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(text_input)
        text_embedding = LSTM(units, return_sequences=True)(text_embedding)
        
        # Concatenate image and text features
        combined_features = Concatenate(axis=-1)([image_features, text_embedding])
        
        # LSTM layer
        lstm_out = LSTM(units, return_sequences=True)(combined_features)
        
        # Dropout for regularization
        lstm_out = Dropout(0.5)(lstm_out)
        
        # Output layer
        output = Dense(vocab_size, activation='softmax')(lstm_out)
        
        # Define the model
        model = Model(inputs=[image_input, text_input], outputs=output)
        
        return model

        
    elif model_type == 'vit_gpt2':
        # Example ViT-GPT2 model
        gpt2_model = TFGPT2LMHeadModel.from_pretrained('gpt2')
        
        image_input = Input(shape=(feature_shape,))
        
        caption_input = Input(shape=(max_length,))
        caption_embedding = gpt2_model.transformer.wte(caption_input)
        
        gpt2_output = gpt2_model(inputs_embeds=tf.concat([tf.expand_dims(image_input, 1), caption_embedding], axis=1))
        
        decoder = Dense(vocab_size, activation='softmax')(gpt2_output.logits)
        model = Model(inputs=[image_input, caption_input], outputs=decoder)
        
    return model




