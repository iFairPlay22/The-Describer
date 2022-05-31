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

    res = dict()

    with torch.no_grad():

        # Generate an caption from the image
        n_sampled_indices = fullModel.sample(img_tensor)

        # Convert word_ids to words
        n_predicted_caption = []
        for sampled_indices in n_sampled_indices:
            predicted_caption = vocabulary.translate(sampled_indices.cpu().numpy())
            n_predicted_caption.append(' '.join(predicted_caption))

        res["indices"] = n_sampled_indices
        res["words"] = n_predicted_caption

    # We train the models
    fullModel.train()

    return res
