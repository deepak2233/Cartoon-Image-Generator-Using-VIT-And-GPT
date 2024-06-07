from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from transformers import TFGPT2LMHeadModel
import tensorflow as tf

def build_model(max_length, vocab_size, embedding_dim, units, feature_shape, model_type):
    if model_type == 'lstm_cnn':
        # Example LSTM model
        image_input = Input(shape=(feature_shape,))
        image_dense = Dense(embedding_dim, activation='relu')(image_input)
        
        caption_input = Input(shape=(max_length,))
        caption_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(caption_input)
        caption_lstm = LSTM(units)(caption_embedding)
        
        decoder = Dense(vocab_size, activation='softmax')([image_dense, caption_lstm])
        model = Model(inputs=[image_input, caption_input], outputs=decoder)
        
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




