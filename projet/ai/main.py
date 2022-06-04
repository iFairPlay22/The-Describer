import spacy 
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

from cstm_load import Vocab
import cstm_load as cstm_load
import cstm_train as cstm_train
import cstm_predict as cstm_predict
import cstm_model as cstm_model
import cstm_plot as cstm_plot

def download():
    
    print("\n\n==> Download (nltk punkt)...")
    nltk.download('punkt')

    print("\n\n==> Download (spacy en_core_web_sm)...")
    spacyEn = spacy.load('en_core_web_sm')

    return spacyEn

if __name__ == "__main__":

    # Programm constants
    totalEpochs = 20

    step = 0.0001

    spacyEn = download()

    images_path = [
        { "input" : './data_dir/train2014/', "output" : './data_dir/resized_images/test2014' },
        { "input" : './data_dir/val2014/',   "output" : './data_dir/resized_images/val2014' },
    ]

    captions_path = [
        './data_dir/annotations/captions_train2014.json',
        './data_dir/annotations/captions_val2014.json'
    ]

    output_vocabulary_path = './data_dir/vocabulary.pkl'
    output_models_path = './models_dir/'

    input_image_to_test_path = './sample.jpg'

    image_shape = [256, 256]

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(
            (0.485, 0.456, 0.406), 
            (0.229, 0.224, 0.225)
        )
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    todo = [ 
        # "install", 
        "train",
        # "test"
        # "predict"
    ]

    # Step 1 : Build the vocabulary & Build the images folder
    if "install" in todo:

        # Load the captions from "input_train_annotations_captions_train_path", generate the vocabulary and save it to "output_vocabulary_path"
        cstm_load.store_vocabulary(captions_path[0], output_vocabulary_path)

        # Load the images from "input_train_images_path", resize them to "image_shape" dimentions and save them in "output_train_resized_images_path"
        cstm_load.reshape_images(images_path, image_shape)


    if "train" in todo or "test" in todo or "predict" in todo:

        # Load the vocabulary
        vocabulary = Vocab.load(output_vocabulary_path)

        # Get the models
        fullModel = cstm_model.FullModel(device, image_shape, vocabulary)

        # Load the weights of the previous training
        fullModel.load()

        # Step 2 : Train the model
        if "train" in todo:
            
            cstm_train.train(totalEpochs, step, vocabulary, fullModel, images_path, captions_path, output_models_path, device, transform, spacyEn)

        # Step 3 : Test the model 
        if "test" in todo:

            cstm_train.testAll(vocabulary, fullModel, images_path, captions_path, device, transform, spacyEn)

        # Step 3 : Test the model 
        if "predict" in todo:

            # Prepare an image
            img_tensor = cstm_load.load_image(input_image_to_test_path, device, transform)

            # Predict the caption
            predicted_sentence = cstm_predict.predict(img_tensor, vocabulary, fullModel)
        
            # Print out the generated caption
            print(predicted_sentence)
            
    
            