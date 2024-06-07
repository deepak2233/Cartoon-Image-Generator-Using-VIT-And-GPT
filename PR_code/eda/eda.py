import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from collections import Counter
import numpy as np
from PIL import Image
import os

def plot_caption_length_distribution(df, save_path='eda/plots/caption_length_distribution.png'):
    plt.figure(figsize=(10,6))
    df['length'] = df['image_description'].apply(lambda x: len(x.split()))
    sns.histplot(df['length'], kde=False, bins=30)
    plt.title('Caption Length Distribution')
    plt.xlabel('Caption Length')
    plt.ylabel('Frequency')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_common_words(df, save_path='eda/plots/common_words.png'):
    all_words = ' '.join([text for text in df['image_description']])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
    plt.figure(figsize=(10,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_top_ngrams(df, n=2, top_k=20, save_path='eda/plots/top_ngrams.png'):
    from sklearn.feature_extraction.text import CountVectorizer
    
    def get_top_ngrams(corpus, n=None, top_k=20):
        vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:top_k]
    
    corpus = df['image_description']
    top_ngrams = get_top_ngrams(corpus, n=n, top_k=top_k)
    ngrams, counts = zip(*top_ngrams)
    
    plt.figure(figsize=(10,6))
    sns.barplot(x=list(counts), y=list(ngrams))
    plt.title(f'Top {top_k} {n}-grams')
    plt.xlabel('Frequency')
    plt.ylabel(f'{n}-grams')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_wordcloud(df, save_path='eda/plots/wordcloud.png'):
    all_words = ' '.join([text for text in df['image_description']])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
    plt.figure(figsize=(10,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def display_sample_images(df, sample_size=5, save_path='eda/plots/sample_images.png'):
    sample_df = df.sample(n=sample_size)
    plt.figure(figsize=(15, 15))
    for i, (_, row) in enumerate(sample_df.iterrows()):
        img = row['image']
        plt.subplot(1, sample_size, i+1)
        plt.imshow(img)
        plt.title(row['image_description'])
        plt.axis('off')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

