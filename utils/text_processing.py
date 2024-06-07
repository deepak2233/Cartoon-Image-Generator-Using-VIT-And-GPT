import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


def preprocess_text(df_train, df_val, df_test):
    all_captions = df_train['image_description'].tolist() + df_val['image_description'].tolist() + df_test['image_description'].tolist()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    
    def captions_to_sequences(captions):
        sequences = tokenizer.texts_to_sequences(captions)
        return pad_sequences(sequences, padding='post')
    
    train_sequences = captions_to_sequences(df_train['image_description'])
    val_sequences = captions_to_sequences(df_val['image_description'])
    test_sequences = captions_to_sequences(df_test['image_description'])
    
    max_length = max(len(seq) for seq in train_sequences)
    return tokenizer, train_sequences, val_sequences, test_sequences, vocab_size, max_length



def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def generate_caption(model, tokenizer, image, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, '')
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    final_caption = in_text.split(' ', 1)[1].rsplit(' ', 1)[0]
    return final_caption


