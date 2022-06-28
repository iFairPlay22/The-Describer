import cstm_vars as v
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
import cstm_accuracy as cstm_accuracy

class CustomCocoDataset(data.Dataset):
    """ Custom Coco dataset that can represent a subset of the original dataset for training, test or evaluation. """
    
    def __init__(self, output_rezized_image_folder_path : str, input_annotations_captions_train_path : str, vocabulary : cstm_load.Vocab, transform = None):
        """ Initialize the dataset. """

        print("\n\n==> Itializating CustomCocoDataset()")

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
        # return 100 # to set a max quantity of images to load (util for testing)
 
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

def learn(scaler, custom_training_data_loader : CustomCocoDataset, vocabulary : cstm_load.Vocab, fullModel : cstm_model.FullModel, optimizer : torch.optim.Adam, epoch : int, costPlot : cstm_plot.SmartPlot = None , lossPlot : cstm_plot.SmartPlot = None, accuracyAveragePlot : cstm_plot.SmartPlot = None, detailedAccuracyPlots : cstm_plot.SmartPlot = None):
    """ Learn 1 epoch of the model and save the loss in the plots. """

    print("\n\n==> learn()")
    accurracyTool = cstm_accuracy.AccuracyBasedOnSynonyms()
    imageNb = 0
    allLoss = []

    # Foreach batch
    for images, captions, lens in tqdm(custom_training_data_loader):

        # Get the batch data
        images = images.to(v.DEVICE)
        captions = captions.to(v.DEVICE)
        tgts = pack_padded_sequence(captions, lens, batch_first=True)[0]

        # Reset the gradient
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            # Make predictions
            outputs = fullModel.forward(images, captions, lens)

            # Compute the total error
            loss    = fullModel.loss(outputs, tgts)
            allLoss.append(loss.item())
            

        

        
        
        
        
        # Backward and step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            # Make predictions
            predictions = cstm_predict.predict(images, vocabulary, fullModel)

            # Foreach image
            for j in range(len(images)):

                predictionSentance = vocabulary.translate(predictions["indices"][j].cpu().numpy()).split()[1:-1]
                targetSentance     = vocabulary.translate(captions[j].cpu().numpy()).split()[1:-1]
                accurracyTool.calculateAccuracy(predictionSentance, targetSentance)
                imageNb += 1

    # Data loss
    print()
    print(' [LEARNING] : Epoch [{}], Cost : {:.4f}, Loss: {:.4f}'.format(epoch+1, sum(allLoss), sum(allLoss) / len(allLoss))) 
    if costPlot:
        costPlot.addPoint("Learning cost", "red", sum(allLoss))
    if lossPlot:
        lossPlot.addPoint("Learning loss", "red", sum(allLoss) / len(allLoss))

    # Ratio average
    print()
    print('>> Ratio average')
    accuracyAverage = accurracyTool.getRatioAverage()
    print("Learning accuracy : {:.4f}% (good key words used)".format(accuracyAverage))
    if accuracyAveragePlot:
        accuracyAveragePlot.addPoint("Learning accuracy", "red", accuracyAverage)

    # Ratio detail
    print()
    print('>> Ratio details')
    detailedRatios, cummulatedDetailedRatios = accurracyTool.getDetailedRatios()
    for i in range(len(detailedRatios)):
        detailedRatio = detailedRatios[i]
        cummulatedDetailedRatio = cummulatedDetailedRatios[i]
        print(' Learning good predictions for ratio {:.2f} : {}% ({}/{})'.format(detailedRatio["min"], detailedRatio["sum"] * 100 / imageNb, detailedRatio["sum"], imageNb))
        if detailedAccuracyPlots:
            detailedAccuracyPlots[cummulatedDetailedRatio["min"]].addPoint("Learning accuracy", "red", cummulatedDetailedRatio["sum"] * 100 / imageNb)
    print()

    # We save the model
    fullModel.save(epoch)

def eval(custom_testing_data_loader : CustomCocoDataset, vocabulary : cstm_load.Vocab, fullModel : cstm_model.FullModel, epoch, costPlot : cstm_plot.SmartPlot = None , lossPlot : cstm_plot.SmartPlot = None, accuracyAveragePlot : cstm_plot.SmartPlot = None, detailedAccuracyPlots : cstm_plot.SmartPlot = None):
    """ Test 1 epoch of the model and save the accuracy in the plots. """

    with torch.no_grad():

        print("\n\n==> eval()")
        accurracyTool = cstm_accuracy.AccuracyBasedOnSynonyms()
        imageNb = 0
        allLoss = []

        # Foreach batch
        for images, captions, lens in tqdm(custom_testing_data_loader):
            
            # Get the batch data
            images = images.to(v.DEVICE)
            captions = captions.to(v.DEVICE)
            tgts = pack_padded_sequence(captions, lens, batch_first=True)[0]

            # Make predictions
            outputs = fullModel.forward(images, captions, lens)

            # Compute the total error
            loss    = fullModel.loss(outputs, tgts)
            allLoss.append(loss.item())

            # Make predictions
            predictions = cstm_predict.predict(images, vocabulary, fullModel)

            # Foreach image
            for j in range(len(images)):

                predictionSentance = vocabulary.translate(predictions["indices"][j].cpu().numpy()).split()[1:-1]
                targetSentance     = vocabulary.translate(captions[j].cpu().numpy()).split()[1:-1]
                accurracyTool.calculateAccuracy(predictionSentance, targetSentance)
                imageNb += 1


        # Data loss
        print()
        print(' [TESTING] : Epoch [{}], Cost : {:.4f}, Loss: {:.4f}'.format(epoch+1, sum(allLoss), sum(allLoss) / len(allLoss))) 
        if costPlot:
            costPlot.addPoint("Validation cost", "green", sum(allLoss))
        if lossPlot:
            lossPlot.addPoint("Validation loss", "green", sum(allLoss) / len(allLoss))

        # Ratio average
        print()
        print('>> Ratio average')
        accuracyAverage = accurracyTool.getRatioAverage()
        print("Validation accuracy : {:.4f}% (good key words used)".format(accuracyAverage))
        if accuracyAveragePlot:
            accuracyAveragePlot.addPoint("Validation accuracy", "green", accuracyAverage)

        # Ratio detail
        print()
        print('>> Ratio details')
        detailedRatios, cummulatedDetailedRatios = accurracyTool.getDetailedRatios()
        for i in range(len(detailedRatios)):
            detailedRatio = detailedRatios[i]
            cummulatedDetailedRatio = cummulatedDetailedRatios[i]
            print(' Validation good predictions for ratio {:.2f} : {}% ({}/{})'.format(detailedRatio["min"], detailedRatio["sum"] * 100 / imageNb, detailedRatio["sum"], imageNb))
            if detailedAccuracyPlots:
                detailedAccuracyPlots[cummulatedDetailedRatio["min"]].addPoint("Validation accuracy", "green", cummulatedDetailedRatio["sum"] * 100 / imageNb)
        print()

def train(vocabulary : cstm_load.Vocab, fullModel : cstm_model.FullModel, withTestDataset=False):
    """ Train and test the model with many epochs """

    # Build the data loader for the training set
    custom_training_data_loader = get_loader(v.IMAGES_PATH[0]["output"], v.CAPTIONS_PATH[0], vocabulary, v.TRANSFORM, v.BATCH_SIZE, shuffle=True, num_workers=2) 

    if withTestDataset:    
        # Build the data loader for the testing set
        custom_testing_data_loader = get_loader(v.IMAGES_PATH[1]["output"], v.CAPTIONS_PATH[1], vocabulary, v.TRANSFORM, v.BATCH_SIZE, shuffle=True, num_workers=2) 

    # Train the models
    print("\n\n==> Train the models...")
    fullModel.trainMode()

    # Use Adam optimizer
    optimizer = torch.optim.Adam(fullModel.getAllParameters(), lr=v.STEP)

    # Display the plot
    costPlot = cstm_plot.SmartPlot("Cost", "Epochs", "Cost (total loss)", v.OUTPUT_PLOTS_PATH)
    lossPlot = cstm_plot.SmartPlot("Loss", "Epochs", "Loss", v.OUTPUT_PLOTS_PATH)
    accuracyPlot = cstm_plot.SmartPlot("Accuracy", "Epochs", "Accuracy", v.OUTPUT_PLOTS_PATH)
    detailedAccuracyPlots = {
        ratio["min"] : cstm_plot.SmartPlot("Accuracy plot for ratio {:.2f}".format(ratio["min"]), "Epoch", "Common key words ratio", v.OUTPUT_PLOTS_PATH)
        for ratio in cstm_accuracy.AccuracyBasedOnSynonyms.getRatios()
    }
    scaler = torch.cuda.amp.GradScaler()
    # For each epoch
    for epoch in tqdm(range(v.TOTAL_EPOCHS)):

        print("\n\n==> Epoch " + str(epoch) + "...", end="")

        # Learn
        learn(scaler, custom_training_data_loader, vocabulary, fullModel, optimizer, epoch, costPlot, lossPlot, accuracyPlot, detailedAccuracyPlots)

        if withTestDataset:
            # test
            eval(custom_testing_data_loader, vocabulary, fullModel, epoch, costPlot, lossPlot, accuracyPlot, detailedAccuracyPlots)

    # Save the plots
    for plot in list(detailedAccuracyPlots.values()) + [ costPlot, lossPlot, accuracyPlot]:
        plot.build()

def test(vocabulary : cstm_load.Vocab, fullModel : cstm_model.FullModel, withTestDataset=True, withTrainDataset=True):
    """ Test the test dataset or the training dataset with the model """

    # Train the models
    fullModel.evalMode()

    if withTrainDataset:
        # Build the data loader for the training set

        print("\n\n==> Evaluating the train model...")
        custom_training_data_loader = get_loader(v.IMAGES_PATH[0]["output"], v.CAPTIONS_PATH[0], vocabulary, v.TRANSFORM, v.BATCH_SIZE, shuffle=True, num_workers=2) 
        eval(custom_training_data_loader, vocabulary, fullModel, 0)
        print()

    if withTestDataset:    
        # Build the data loader for the testing set

        print("\n\n==> Evaluating the test model...")
        custom_testing_data_loader = get_loader(v.IMAGES_PATH[1]["output"], v.CAPTIONS_PATH[1], vocabulary, v.TRANSFORM, v.BATCH_SIZE, shuffle=True, num_workers=2) 
        eval(custom_testing_data_loader, vocabulary, fullModel, 0)
        print()
