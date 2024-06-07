import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNet50
import torch
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel

def preprocess_and_extract_features(image_objects):
    base_model = ResNet50(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

    features_list = [] 
    for img in image_objects:
        try:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            image = img.resize((224, 224))
            img_array = np.array(image)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            features = model.predict(img_array)
            features_list.append(features.flatten())
        except Exception as e:
            print(f"Error processing image: {e}")
            features_list.append(None)
    return np.array(features_list)



def preprocess_and_extract_features_vit(image_objects):
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224')

    features_list = []
    for img in image_objects:
        try:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            image = img.resize((224, 224))
            inputs = feature_extractor(images=image, return_tensors="pt")
            with torch.no_grad():
                features = model(**inputs).last_hidden_state
            features_list.append(features.numpy().flatten())
        except Exception as e:
            print(f"Error processing image: {e}")
            features_list.append(None)
    return np.array(features_list)

