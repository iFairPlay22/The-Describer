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

# Step 1 : CNN

class CNNModel(nn.Module):

    def __init__(self, embedding_size):
        super(CNNModel, self).__init__()

        print("\n\n==> Intializating CNNModel()")

        # We use a pre-trained CNN model available under the PyTorch models repository: the ResNet 152 architecture
        # We remove the last layer of this pre-trained ResNet model 
        resnet = models.resnet152(pretrained=True)      
        module_list = list(resnet.children())[:-1] 
        self.__resnet_module = nn.Sequential(*module_list)

        # Replace it with a fully connected layer
        self.__linear_layer = nn.Linear(resnet.fc.in_features, embedding_size)

        # Followed by a batch normalization layer
        self.__batch_norm = nn.BatchNorm1d(embedding_size, momentum=0.01)

        print(embedding_size)
        
    def forward(self, input_images):
        """Extract feature vectors from input images."""

        # We don't train the ResNet model because it is pretrained
        #   The output of the ResNet model is K x l000-dimensional, assuming K number of neurons in the penultimate layer

        with torch.no_grad():
            resnet_features = self.__resnet_module(input_images)

        # Reshape the output of the ResNet model to (batch_size, resnet.fc.in_features)
        resnet_features = resnet_features.reshape(resnet_features.size(0), -1)

        # We train the 2 other layers (our own)

        # We apply the fully connected layer
        #   The output of the fully connected layer is (batch_size, embedding_size)
        linear_features = self.__linear_layer(resnet_features)

        # We apply the batch normalization layer
        #   The batch normalization layer normalizes the fully connected layer outputs with a mean
        #   of 0 and a standard deviation of 1 across the entire batch.
        final_features = self.__batch_norm(linear_features)

        return final_features

    def getAllParameters(self):
        return list(self.__linear_layer.parameters()) + list(self.__batch_norm.parameters())

# Step 2 : LSTM

class LSTMModel(nn.Module):

    def __init__(self, embedding_size, hidden_layer_size, vocabulary_size, num_layers, max_seq_len=20):
        super(LSTMModel, self).__init__()

        print("\n\n==> Intializating LSTMModel()")

        # Embedding layer 
        self.__embedding_layer = nn.Embedding(vocabulary_size, embedding_size)
        
        # LSTM layer
        self.__lstm_layer = nn.LSTM(embedding_size, hidden_layer_size, num_layers, batch_first=True)
        
        # Fully connected linear layer
        self.__linear_layer = nn.Linear(hidden_layer_size, vocabulary_size)
        
        # Max length of the predited caption
        self.__max_seq_len = max_seq_len
        
    def forward(self, input_features, captions, lens):
        """Decode image feature vectors and generates captions."""

        # We apply the embedding layer
        embeddings = self.__embedding_layer(captions)
        embeddings = torch.cat((input_features.unsqueeze(1), embeddings), 1)
        lstm_input = pack_padded_sequence(embeddings, lens, batch_first=True) 

        # We apply the LSTM layer
        hidden_variables, _ = self.__lstm_layer(lstm_input)

        # We apply the fully connected layer
        model_outputs = self.__linear_layer(hidden_variables[0])

        return model_outputs
    
    def sample(self, input_features, lstm_states=None):
        """Generate captions for given image features using greedy search."""

        sampled_indices = []

        # Insert 1 dimention to the input features
        lstm_inputs = input_features.unsqueeze(1)

        # Generate a caption with max self.__max_seq_len size
        for i in range(self.__max_seq_len):

            hidden_variables, lstm_states = self.__lstm_layer(lstm_inputs, lstm_states) # hiddens: (batch_size, 1, hidden_size)
            model_outputs = self.__linear_layer(hidden_variables.squeeze(1))            # outputs: (batch_size, vocab_size)
            _, predicted_outputs = model_outputs.max(1)                                 # predicted: (batch_size)
            sampled_indices.append(predicted_outputs)

            lstm_inputs = self.__embedding_layer(predicted_outputs)                     # inputs: (batch_size, embed_size)
            lstm_inputs = lstm_inputs.unsqueeze(1)                                      # inputs: (batch_size, 1, embed_size)

        sampled_indices = torch.stack(sampled_indices, 1)                               # sampled_ids: (batch_size, max_seq_length)
        return sampled_indices

    def getAllParameters(self):
        return list(self.parameters())

# Step 3 : CNN & LSTM

class FullModel(nn.Module):

    def __init__(self, device, image_shape, vocabulary):
        """Combine the CNN and LSTM models."""
        super(FullModel, self).__init__()

        # Build the models
        self.__encoder_model = CNNModel(image_shape[0]).to(device)
        self.__decoder_model = LSTMModel(image_shape[0], image_shape[0] + image_shape[1], len(vocabulary), 1).to(device)        
            
        # Loss and optimizer
        self.__loss_criterion = nn.CrossEntropyLoss()

    def forward(self, images, captions, lens):

        feats   = self.__encoder_model(images)
        outputs = self.__decoder_model(feats, captions, lens)

        return outputs

    def sample(self, image):

        feats = self.__encoder_model(image)
        return self.__decoder_model.sample(feats)

    def loss(self, outputs, tgts):

        # outputs = F.softmax(outputs, dim=1)
        return self.__loss_criterion(outputs, tgts)

    # def zero_grad(self):
    #     self.__decoder_model.zero_grad()
    #     self.__encoder_model.zero_grad()

    def getAllParameters(self):
        return self.__decoder_model.getAllParameters() + self.__encoder_model.getAllParameters()

    def save(self, output_models_path, epoch, batch_id):

        torch.save(
            self.__decoder_model.state_dict(), 
            os.path.join(output_models_path, 'decoder-{}-{}.ckpt'.format(epoch+1, batch_id + 1))
        )

        torch.save(
            self.__encoder_model.state_dict(), 
            os.path.join(output_models_path, 'encoder-{}-{}.ckpt'.format(epoch+1, batch_id + 1))
        )

    def load(self):

        decoderFile = ''
        encoderFile = ''

        files = os.listdir('models_dir')
        
        for filename in files:
            if(filename.startswith('decoder')):
                decoderFile = filename
            if(filename.startswith('encoder')):
                encoderFile = filename
        
        self.__encoder_model.load_state_dict(torch.load('models_dir/' + encoderFile))
        self.__decoder_model.load_state_dict(torch.load('models_dir/' + decoderFile))


    def trainMode(self):
        self.__encoder_model.train()
        self.__decoder_model.train()

    def testMode(self):
        self.__encoder_model.eval()
        self.__decoder_model.eval()