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

class CustomCocoDataset(data.Dataset):
    
    def __init__(self, output_rezized_image_folder_path, input_annotations_captions_train_path, vocabulary, transform=None):

        print("\n\n==> Intializating CustomCocoDataset()")

        self.__output_rezized_image_folder_path = output_rezized_image_folder_path
        self.__data = COCO(input_annotations_captions_train_path)
        self.__indices = list(self.__data.anns.keys())
        self.__vocabulary = vocabulary
        self.__transform = transform
 
    #Returns one data pair (image, caption)
    def __getitem__(self, idx):

        annotation_id = self.__indices[idx]


        # Step 1 => Get the resized image and apply a transform to it 

        # Get the image id
        image_id = self.__data.anns[annotation_id]['image_id']

        # Get the resized image caption
        image_path = self.__data.loadImgs(image_id)[0]['file_name']

        # Get the resized image
        image = Image.open(os.path.join(self.__output_rezized_image_folder_path, image_path)).convert('RGB')

        # Apply the given transformation to the image
        if self.__transform is not None:
            image = self.__transform(image)

        # Step 2 => Get the caption and transform it to a list of integers 

        # Get the image caption
        caption = self.__data.anns[annotation_id]['caption']

        # Convert caption (string) to word ids.
        words = nltk.tokenize.word_tokenize(str(caption).lower())

        caption_ids = []
        caption_ids.append(self.__vocabulary('<start>'))
        caption_ids.extend([self.__vocabulary(token) for token in words])
        caption_ids.append(self.__vocabulary('<end>'))

        # Transform to a Torch tensor
        caption_ids_torch = torch.Tensor(caption_ids)       

        return image, caption_ids_torch
 
    def __len__(self):
        return len(self.__indices)
 
#  Creates mini-batch tensors from the list of tuples (image, caption)
def collate_function(data_batch):

    # Sort a data list by caption length (descending order).
    data_batch.sort(key=lambda d: len(d[1]), reverse=True)
    images, captions = zip(*data_batch)
 
    # Convert a list of 3D tensors (list(tensor(3, 256, 256))) to a 4D tensor (tensor(<batch_size>, 3, 256, 256))
    images = torch.stack(images, 0)
 
    # Merge captions (from list of 1D tensors to 2D tensor), similar to merging of images donw above.

    # Get the length of each captions
    cap_lens = [len(cap) for cap in captions]

    # Pad each caption to the longest caption in the batch (with 0)
    tgts = torch.zeros(len(captions), max(cap_lens)).long()

    # Complete the tensor with the captions ids
    for i, cap in enumerate(captions):

        # The end index is the size of the caption
        end = cap_lens[i]

        # Replace the 0 (in [0, end]) with the captions ids
        tgts[i, :end] = cap[:end]        

    # Returns:
    #         images: torch tensor of shape (batch_size, 3, 256, 256).
    #         tgts: torch tensor of shape (batch_size, padded_length).
    #         cap_lens: list; valid length for each padded caption.

    return images, tgts, cap_lens
 
# Returns torch.utils.data.DataLoader for custom coco dataset.
# This will return (images, captions, lengths) for each iteration.
#   images: a tensor of shape (batch_size, 3, 224, 224).
#   captions: a tensor of shape (batch_size, padded_length).
#   lengths: a list indicating valid length for each caption. length is (batch_size).
def get_loader(data_path, input_annotations_captions_train_path, vocabulary, transform, batch_size, shuffle, num_workers):

    # COCO caption dataset
    coco_dataset = CustomCocoDataset(data_path, input_annotations_captions_train_path, vocabulary, transform)
    
    # Data loader for COCO dataset
    custom_data_loader = torch.utils.data.DataLoader(dataset=coco_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_function)
    
    return custom_data_loader


def train(output_resized_images_path, input_annotations_captions_train_path, output_models_path, output_vocabulary_path, device, image_shape, transform):

    # Create model directory
    if not os.path.exists(output_models_path):
        os.makedirs(output_models_path)

    # Load vocabulary wrapper
    with open(output_vocabulary_path, 'rb') as f:
        vocabulary = pickle.load(f)

    # Build data loader
    custom_data_loader = get_loader(output_resized_images_path, input_annotations_captions_train_path, vocabulary, transform, 128, shuffle=True, num_workers=2) 

    # Build the models
    fullModel = cstm_model.FullModel(device, image_shape, vocabulary, True)

    # Optimizers
    optimizer = torch.optim.Adam(fullModel.getAllParameters(), lr=0.001)

    # Train the models
    print("\n\n==> Train the models...")
    
    total_num_steps = len(custom_data_loader)
    for epoch in tqdm(range(5)):

        print("\n\n==> Epoch " + str(epoch) + "...", end="")

        i = 0
        for images, captions, lens in tqdm(custom_data_loader):
    
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            tgts = pack_padded_sequence(captions, lens, batch_first=True)[0]

            # Forward, backward and optimize
            optimizer.zero_grad()

            # We make predictions
            outputs = fullModel.forward(images, captions, lens)

            # We get the total error
            loss    = fullModel.loss(outputs, tgts)

            # fullModel.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % 250 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                    .format(epoch, 5, i, total_num_steps, loss.item(), np.exp(loss.item()))) 
    
            # Save the model checkpoints
            if (i+1) % 1000 == 0:
                fullModel.save(output_models_path, epoch, i)
            
            i += 1

        fullModel.save(output_models_path, epoch, i)