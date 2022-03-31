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

print("\n\n==> Download nlkt...")
nltk.download('punkt')

if __name__ == "__main__":

    # Programm constants
    input_annotations_captions_train_path   = './data_dir/annotations/captions_train2014.json'
    output_vocabulary_path                  = './data_dir/vocabulary.pkl'

    input_images_path                       = './data_dir/train2014/'
    output_resized_images_path              = './data_dir/resized_images/'

    output_models_path                      = './models_dir/'

    input_image_to_test_path                = './sample.jpg'

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
        # "train"
        "test" 
    ]

    # Step 1 : Build the vocabulary & Build the images folder
    if "install" in todo:

        # Load the captions from "input_annotations_captions_train_path", generate the vocabulary and save it to "output_vocabulary_path"
        cstm_load.load_store_vocabulary(input_annotations_captions_train_path, output_vocabulary_path)

        # Load the images from "input_images_path", resize them to "image_shape" dimentions and save them in "output_resized_images_path"
        cstm_load.reshape_images(input_images_path, output_resized_images_path, image_shape)

    # Step 2 : Train the model
    if "train" in todo:
        cstm_train.train(output_resized_images_path, input_annotations_captions_train_path, output_models_path, output_vocabulary_path, device, image_shape, transform)
        
    if "test" in todo:
        predicted_sentence = cstm_predict.test(input_image_to_test_path, device, image_shape, transform)
            
        # Print out the image and the generated caption

        img = Image.open(input_image_to_test_path)
        plt.imshow(np.asarray(img))
        
        print (predicted_sentence)
    
            