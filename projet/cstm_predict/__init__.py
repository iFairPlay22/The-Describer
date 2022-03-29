import os
from tqdm import tqdm
import nltk
import pickle
import numpy as np
from PIL import Image
from collections import Counter
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
 
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence

import cstm_model as cstm_model

def load_image(input_image_file_path, device, transform=None):
    img = Image.open(input_image_file_path).convert('RGB')
    img = img.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        img = transform(img).unsqueeze(0)
    
    return img.to(device)

def test(input_image_file_path, device, image_shape, transform):

    # Load vocabulary wrapper
    with open('data_dir/vocabulary.pkl', 'rb') as f:
        vocabulary = pickle.load(f)

    # Get the models
    fullModel = cstm_model.FullModel(device, image_shape, vocabulary)
    fullModel.load()

    # Prepare an image
    img_tensor = load_image(input_image_file_path, device, transform)

    # Generate an caption from the image
    sampled_indices = fullModel.sample(img_tensor)
    sampled_indices = sampled_indices[0].cpu().numpy() 

    # Convert word_ids to words
    predicted_caption = []
    for token_index in sampled_indices:
        word = vocabulary.i2w[token_index]
        predicted_caption.append(word)
        if word == '<end>':
            break
    
    predicted_sentence = ' '.join(predicted_caption)
    return predicted_sentence

