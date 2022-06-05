import spacy 
import difflib
import re
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
import cstm_predict as cstm_predict
import cstm_plot as cstm_plot

class CustomCocoDataset(data.Dataset):
    
    def __init__(self, output_rezized_image_folder_path, input_annotations_captions_train_path, vocabulary, transform=None):

        print("\n\n==> InItializating CustomCocoDataset()")

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
    custom_training_data_loader = torch.utils.data.DataLoader(dataset=coco_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_function)
    
    return custom_training_data_loader

def train_learn(custom_training_data_loader, device, optimizer, fullModel, epoch, totalEpochs, totalBatch, output_models_path, learnPlot):

    print("\n\n==> train_learn()")
    
    batchNb = 0

    # Learn
    allLoss = []
    for images, captions, lens in tqdm(custom_training_data_loader):

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
        allLoss.append(loss.item())

        
        .backward()
        optimizer.step()

        batchNb += 1

        # Print log info
        if batchNb % 250 == 0:
            print(' [LEARNING] : Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, totalEpochs, batchNb, totalBatch, loss.item(), np.exp(loss.item()))) 

        # Save the model checkpoints
        if (batchNb+1) % 1000 == 0:
            fullModel.save(output_models_path, epoch, batchNb)

    # We save the weights of the models
    learnPlot.addPoint("Total loss", "red", sum(allLoss))
    # learnPlot.addPoint("Average loss", "green", sum(allLoss) / len(allLoss))
    fullModel.save(output_models_path, epoch, totalBatch)

def train_test(custom_testing_data_loader, device, vocabulary, fullModel, batch_size, epoch, totalEpochs, totalBatch, spacyEn, testPlot):

    print("\n\n==> train_test()")
    
    batchNb = 0
    imageNb = 0

    ratios = [
        { "min": 20, "color": "red",    "sum": 0 },
        { "min": 40, "color": 'orange', "sum": 0 },
        { "min": 60, "color": 'blue',   "sum": 0 },
        { "min": 80, "color": 'cyan',   "sum": 0 },
        { "min": 90, "color": 'green',  "sum": 0 }
    ]

    # Test
    for images, captions, lens in tqdm(custom_testing_data_loader):
        
        images = images.to(device)
        captions = captions.to(device)
        predictions = cstm_predict.predict(images, vocabulary, fullModel)
        imagesNg = len(images)

        for j in range(imagesNg):

            # Get the caption / prediction
            prediction = vocabulary.translate(predictions["indices"][j].cpu().numpy())
            caption = vocabulary.translate(captions[j].cpu().numpy())[1:-1]
            
            # Remove stop words from the string & punctuation
            predictedSentance = re.sub(r'[^\w\s]', '', prediction).split(" ")
            goodSentance = re.sub(r'[^\w\s]', '', caption).split(" ")

            # Words to integer list
            sw = spacyEn.Defaults.stop_words
            predictedKeywords = list(set( w for w in predictedSentance if w and not w in sw ))
            goodKeywords = list(set( w for w in goodSentance if w and not w in sw ))

            # Update ratios
            currentRatio = difflib.SequenceMatcher(None, predictedKeywords, goodKeywords).ratio()
            for ratio in ratios:
                if ratio["min"] <= 100 * currentRatio:
                    ratio["sum"] += 1
            
            imageNb += 1

        batchNb += 1

        # Print log info
        if batchNb % 250 == 0:
            print(' [TESTING] : Epoch [{}/{}], Step [{}/{}]'.format(epoch, totalEpochs, batchNb, totalBatch))
            for ratio in ratios:
                print(' Good predictions for ratio {} : {}% ({}/{})'.format(ratio["min"], ratio["sum"] * 100 / imageNb, ratio["sum"], imageNb))

    print(' [TESTING] : Epoch [{}/{}], Step [{}/{}]'.format(epoch, totalEpochs, batchNb, totalBatch))
    for ratio in ratios:
        ratio["avg"] = ratio["sum"] * 100 / imageNb
        print(' Good predictions for ratio {} : {}% ({}/{})'.format(ratio["min"], ratio["avg"], ratio["sum"], imageNb))
        testPlot.addPoint(ratio["min"], ratio["color"], ratio["avg"])
    
def train(totalEpochs, step, vocabulary, fullModel, images_path, captions_path, output_models_path, device, transform, spacyEn):

    # Create model directory
    if not os.path.exists(output_models_path):
        os.makedirs(output_models_path)

    # Build data loader
    batch_size = 128

    custom_training_data_loader = get_loader(images_path[0]["output"], captions_path[0], vocabulary, transform, batch_size, shuffle=True, num_workers=2) 
    training_total_num_steps = len(custom_training_data_loader) // batch_size
    
    # custom_testing_data_loader = get_loader(images_path[1]["output"], captions_path[1], vocabulary, transform, batch_size, shuffle=True, num_workers=2) 
    # testing_total_num_steps = len(custom_testing_data_loader) // batch_size

    # Train the models
    fullModel.train()

    # Optimizers
    optimizer = torch.optim.Adam(fullModel.getAllParameters(), lr=step)

    # Train the models
    print("\n\n==> Train the models...")

    # Display the plot
    
    learnPlot = cstm_plot.SmartPlot("Training", "Epochs", "Loss")
    testPlot = cstm_plot.SmartPlot("Test", "Epochs", "Ratios")

    for epoch in tqdm(range(totalEpochs)):

        print("\n\n==> Epoch " + str(epoch) + "...", end="")

        train_learn(custom_training_data_loader, device, optimizer, fullModel, epoch, totalEpochs, training_total_num_steps, output_models_path, learnPlot)

        # Save the plot
        learnPlot.build()

        # train_test(custom_testing_data_loader, device, vocabulary, fullModel, batch_size, epoch, totalEpochs, testing_total_num_steps, spacyEn, testPlot)

        # Save the plot
        testPlot.build()

    # Save the plot
    learnPlot.build()
    testPlot.build()

def testAll(vocabulary, fullModel, images_path, captions_path, device, transform, spacyEn):

    # Build data loader
    batch_size = 128
    
    custom_testing_data_loader = get_loader(images_path[1]["output"], captions_path[1], vocabulary, transform, batch_size, shuffle=True, num_workers=2) 
    testing_total_num_steps = len(custom_testing_data_loader) // batch_size

    # Train the models
    fullModel.test()

    # Train the models
    print("\n\n==> Test the model...")
    
    totalEpochs = 5

    # Display the plot
    plot = cstm_plot.cstm_plot.SmartPlot("[Test] : Ratios", "Epochs", "Ratios")

    for epoch in tqdm(range(totalEpochs)):

        print("\n\n==> Epoch " + str(epoch) + "...", end="")

        train_test(custom_testing_data_loader, device, vocabulary, fullModel, batch_size, epoch, totalEpochs, testing_total_num_steps, spacyEn, plot)

        # Save the plot
        plot.build()

    # Save the plot
    plot.build()
