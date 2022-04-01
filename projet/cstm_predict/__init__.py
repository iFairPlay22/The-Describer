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
import cstm_load as cstm_load

def predict(img_tensor, vocabulary, fullModel):
    
    # We don't train the models
    fullModel.eval()

    predicted_sentence = torch.tensor([])

    with torch.no_grad():

        # Generate an caption from the image
        sampled_indices = fullModel.sample(img_tensor)
        sampled_indices = sampled_indices[0].cpu().numpy() 

        # Convert word_ids to words
        predicted_caption = []
        for token_index in sampled_indices:
            word = vocabulary.getToken(token_index)
            predicted_caption.append(word)
            if word == '<end>':
                break
        
        predicted_sentence = ' '.join(predicted_caption)

    # We train the models
    fullModel.train()

    return predicted_sentence
