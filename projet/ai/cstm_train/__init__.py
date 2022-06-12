import difflib
import re
import os
from tqdm import tqdm
import nltk
from PIL import Image
from pycocotools.coco import COCO
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pack_padded_sequence

import cstm_load as cstm_load
import cstm_model as cstm_model
import cstm_predict as cstm_predict
import cstm_plot as cstm_plot
import cstm_util

class CustomCocoDataset(data.Dataset):
    """ Custom Coco dataset that can represent a subset of the original dataset for training, test or evaluation. """
    
    def __init__(self, output_rezized_image_folder_path : str, input_annotations_captions_train_path : str, vocabulary : cstm_load.Vocab, transform = None):
        """ Initialize the dataset. """

        print("\n\n==> InItializating CustomCocoDataset()")

        self.__output_rezized_image_folder_path = output_rezized_image_folder_path
        self.__data = COCO(input_annotations_captions_train_path)
        self.__indices = list(self.__data.anns.keys())
        self.__vocabulary = vocabulary
        self.__transform = transform
 
    def __getitem__(self, idx : int):
        """ Returns one data pair (image, caption) """

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
        """ Return the size of the dataset """
        return len(self.__indices)
 
def collate_function(data_batch : int):
    """ Creates mini-batch tensors from the list of tuples (image, caption) """

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

def get_loader(data_path : str, input_annotations_captions_train_path : str, vocabulary : cstm_load.Vocab, transform, batch_size : int, shuffle, num_workers):

    """
    Returns torch.utils.data.DataLoader for custom coco dataset.
    This will return (images, captions, lengths) for each iteration.
      images: a tensor of shape (batch_size, 3, 224, 224).
      captions: a tensor of shape (batch_size, padded_length).
      lengths: a list indicating valid length for each caption. length is (batch_size).
    """

    # COCO caption dataset
    coco_dataset = CustomCocoDataset(data_path, input_annotations_captions_train_path, vocabulary, transform)
    
    # Data loader for COCO dataset
    custom_training_data_loader = torch.utils.data.DataLoader(dataset=coco_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_function)
    
    return custom_training_data_loader

def learn(custom_training_data_loader : CustomCocoDataset, device, optimizer, fullModel : cstm_model.FullModel, epoch : int, output_models_path : str, learnPlot : cstm_plot.SmartPlot = None , learnPlot2 : cstm_plot.SmartPlot = None):
    """ Learn 1 epoch of the model and save the loss in the plots. """

    print("\n\n==> learn()")

    # Learn
    batchNb = 0
    allLoss = []

    for images, captions, lens in tqdm(custom_training_data_loader):

        # Get the bach data
        images = images.to(device)
        captions = captions.to(device)
        tgts = pack_padded_sequence(captions, lens, batch_first=True)[0]

        # Reset the gradient
        optimizer.zero_grad()

        # Make predictions
        outputs = fullModel.forward(images, captions, lens)

        # Compute the total error
        loss    = fullModel.loss(outputs, tgts)
        allLoss.append(loss.item())
        
        # Backward and step
        loss.backward()
        optimizer.step()

        batchNb += 1

    # We save the data loss for each epoch
    print()
    print(' [LEARNING] : Epoch [{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch+1, sum(allLoss), sum(allLoss) / len(allLoss))) 

    if learnPlot:
        learnPlot.addPoint("Total loss", "red", sum(allLoss))

    if learnPlot2:
        learnPlot2.addPoint("Average loss", "red", sum(allLoss) / len(allLoss))

    # We save the model
    fullModel.save(output_models_path, epoch)

def eval(custom_testing_data_loader : CustomCocoDataset, device, vocabulary : cstm_load.Vocab, fullModel : cstm_model.FullModel, epoch, spacyEn, testPlot):
    """ Test 1 epoch of the model and save the accuracy in the plots. """

    print("\n\n==> eval()")
    
    batchNb = 0
    imageNb = 0

    ratios = [
        { "min": 0.1, "color": "orange", "sum": 0 },
        { "min": 0.2, "color": "blue",   "sum": 0 },
        { "min": 0.3, "color": 'green',  "sum": 0 },
        { "min": 0.4, "color": 'brown',  "sum": 0 },
        { "min": 0.5, "color": 'grey',   "sum": 0 },
        { "min": 0.6, "color": 'red',    "sum": 0 },
        { "min": 0.7, "color": 'pink',   "sum": 0 },
        { "min": 0.8, "color": 'yellow', "sum": 0 },
        { "min": 0.9, "color": 'aqua',   "sum": 0 }
    ]
    detailRatios = {}
    sw = spacyEn.Defaults.stop_words

    # Test
    for images, captions, lens in tqdm(custom_testing_data_loader):
        
        images = images.to(device)
        captions = captions.to(device)
        predictions = cstm_predict.predict(images, vocabulary, fullModel)

        for j in range(len(images)):

            # Get the caption / prediction
            prediction = vocabulary.translate(predictions["indices"][j].cpu().numpy()).split()[1:-1]
            caption = vocabulary.translate(captions[j].cpu().numpy()).split()[1:-1]

            # Remove punctuation
            prediction = re.sub(r'[^\w\s]', '', " ".join(prediction)).split(" ")
            caption = re.sub(r'[^\w\s]', '', " ".join(caption)).split(" ")

            # Remove stop words 
            predictionWithoutSw = set( w for w in prediction  if not w in sw )

            # Check if we can find sense similarities
            commonWordsWithSynonyms = [
                syn
                for predictionWordWithoutSw in predictionWithoutSw 
                for syn in cstm_util.getSynonyms(predictionWordWithoutSw)
                if any(caption[idx : idx + len(syn)] == syn for idx in range(len(caption) - len(syn) + 1))
            ]

            # Update ratios
            totalCommonWords = len(commonWordsWithSynonyms)
            if totalCommonWords in detailRatios:
                detailRatios[totalCommonWords] += 1
            else:
                detailRatios[totalCommonWords] = 1

            currentRatio = totalCommonWords / len(predictionWithoutSw)
            for ratio in ratios:
                if ratio["min"] <= currentRatio:
                    ratio["sum"] += 1
            
            imageNb += 1

        batchNb += 1

    print()
    print(' [TESTING] : Epoch [{}]'.format(epoch+1)) 

    print('>> Ratios')
    for ratio in ratios:
        ratio["avg"] = ratio["sum"] * 100 / imageNb
        print(' Good predictions for ratio {} : {}% ({}/{})'.format(ratio["min"], ratio["avg"], ratio["sum"], imageNb))

        if testPlot:
            testPlot.addPoint(ratio["min"], ratio["color"], ratio["avg"])
    
    print('>> Detail')
    for commonWords, number in sorted(detailRatios.items(), key=lambda item: item[0]):
        ratio = number * 100 / imageNb
        print("Common words {} : {}% ({}/{})".format(commonWords, ratio, number, imageNb))

def train(totalEpochs : int, batch_size : int, step : float, vocabulary : cstm_load.Vocab, fullModel : cstm_model.FullModel, images_path : str, captions_path : str, output_models_path : str, output_plot_path : str, device, transform, spacyEn, withTestDataset=False):
    """ Train and test the model with many epochs """

    # Create the model directory if not exists
    if not os.path.exists(output_models_path):
        os.makedirs(output_models_path)

    # Build the data loader for the training set
    custom_training_data_loader = get_loader(images_path[0]["output"], captions_path[0], vocabulary, transform, batch_size, shuffle=True, num_workers=2) 

    if withTestDataset:    
        # Build the data loader for the testing set
        custom_testing_data_loader = get_loader(images_path[1]["output"], captions_path[1], vocabulary, transform, batch_size, shuffle=True, num_workers=2) 

    # Train the models
    print("\n\n==> Train the models...")
    fullModel.trainMode()

    # Use Adam optimizer
    optimizer = torch.optim.Adam(fullModel.getAllParameters(), lr=step)

    # Display the plot
    learnPlot = cstm_plot.SmartPlot("Training", "Epochs", "Loss", output_plot_path)
    learnPlot2 = cstm_plot.SmartPlot("Training", "Epochs", "Loss", output_plot_path)
    testPlot = cstm_plot.SmartPlot("Test", "Epochs", "Ratios", output_plot_path)

    # For each epoch
    for epoch in tqdm(range(totalEpochs)):

        print("\n\n==> Epoch " + str(epoch) + "...", end="")

        # Learn
        learn(custom_training_data_loader, device, optimizer, fullModel, epoch, output_models_path, learnPlot, learnPlot2)

        if withTestDataset:
            # test
            eval(custom_testing_data_loader, device, vocabulary, fullModel, epoch, spacyEn, testPlot)

    # Save the plots
    learnPlot.build()
    learnPlot2.build()

    if withTestDataset:
        testPlot.build()

def testAll(vocabulary : cstm_load.Vocab, fullModel : cstm_model.FullModel, batch_size, images_path, captions_path : str , device, transform, spacyEn, withTestDataset=True, withTrainDataset=True):
    """ Test the test dataset or the training dataset with the model """

    # Train the models
    fullModel.evalMode()

    if withTrainDataset:
        # Build the data loader for the training set

        print("\n\n==> Evaluating the train model...")
        custom_training_data_loader = get_loader(images_path[0]["output"], captions_path[0], vocabulary, transform, batch_size, shuffle=True, num_workers=2) 
        eval(custom_training_data_loader, device, vocabulary, fullModel, 0, spacyEn, None)
        print()

    if withTestDataset:    
        # Build the data loader for the testing set

        print("\n\n==> Evaluating the test model...")
        custom_testing_data_loader = get_loader(images_path[1]["output"], captions_path[1], vocabulary, transform, batch_size, shuffle=True, num_workers=2) 
        eval(custom_testing_data_loader, device, vocabulary, fullModel, 0, spacyEn, None)
        print()
