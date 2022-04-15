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

class Vocab(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        # Word to index
        self.__w2i = {}

        # Index to word
        self.__i2w = {}

        # Current index
        self.__index = 0
 
    # Get the token, of <unk> if not exists
    def __call__(self, token):
        if not token in self.__w2i:
            return self.__w2i['<unk>']

        return self.__w2i[token]

    def getToken(self, index):
        return self.__i2w[index]
 
    # Get the length of the vocabulary
    def __len__(self):
        return len(self.__w2i)

    # Add a word to the vocabulary
    def add_token(self, token):
        if not token in self.__w2i:
            self.__w2i[token] = self.__index
            self.__i2w[self.__index] = token
            self.__index += 1

    def translate(self, sentence):

        words = []
        for word in sentence:
            word = self.getToken(word)
            words.append(word)
            if word == '<end>':
                break

        return " ".join(words)

    def save(self, output):
        
        print("\n\nSave the vocabulary wrapper to '{}'".format(output))
        with open(output, 'wb') as f:
            pickle.dump(self, f)

    def load(output):
    
        print("\n\nLoad the vocabulary wrapper from '{}'".format(output))
        with open(output, 'rb') as f:
            return pickle.load(f)

# Step 1: Build the vocabulary wrapper and save it to disk.

def build_vocabulary(json, threshold):

    # Load JSON
    coco = COCO(json)

    # Loading captions
    ids = coco.anns.keys()
    counter = Counter()
    
    # For every caption ids
    
    for id in tqdm(ids):

        # Get the caption
        caption = str(coco.anns[id]['caption'])

        # Separate every words of the caption
        token = nltk.tokenize.word_tokenize(caption.lower())

        # Count the number of occurrences of every word
        counter.update(token)
 
    # Create a vocab wrapper and add some special tokens.
    vocab = Vocab()
    vocab.add_token('<pad>')
    vocab.add_token('<start>')
    vocab.add_token('<end>')
    vocab.add_token('<unk>')

    # Keep the topens that appears more than threshold 
    for token, cnt in counter.items():
        if threshold <= cnt:
            vocab.add_token(token)
        
    return vocab

def store_vocabulary(input, output):

    print("\n\n==> store_vocabulary()")

    print("Load the data from '{}'".format(input))
    vocab = build_vocabulary(json=input, threshold=4)
    print("Total vocabulary size: {}".format(len(vocab)))

    vocab.save(output)

# Step 2: Resize the images

def load_image(input_image_file_path, device, transform=None):
    img = Image.open(input_image_file_path)
    img = img.resize([224, 224], Image.LANCZOS).convert('RGB')
    
    if transform is not None:
        img = transform(img).unsqueeze(0)
    
    return img.to(device)

def reshape_image(image, shape):
    # Resize an image to the given shape
    return image.resize(shape, Image.ANTIALIAS)
 
def reshape_images(io, shape):

    print("\n\n==> reshape_images()")
    
    # Load the images
    for obj in io:

        input = obj["input"]
        output = obj["output"]

        # We create the output directory if it does not exist
        if not os.path.exists(output):
            os.makedirs(output)
            
        print("Resize images in path '{}'".format(output))
    
        images = os.listdir(input)
        for im in tqdm(images):

            with open(os.path.join(input, im), 'r+b') as f:
                with Image.open(f) as image:

                    # Reshape them
                    image = reshape_image(image, shape)

                    # Save them
                    image.save(os.path.join(output, im), image.format)
