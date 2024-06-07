from datasets import load_dataset
import pandas as pd

def load_caption_data():
    data = load_dataset("jmhessel/newyorker_caption_contest", "explanation_4")
    df_train = pd.DataFrame(data['train'])
    df_val = pd.DataFrame(data['validation'])
    df_test = pd.DataFrame(data['test'])
    return df_train, df_val, df_test