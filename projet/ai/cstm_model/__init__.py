import cstm_vars as v
import os
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

# Step 1 : CNN

class CNNModel(nn.Module):
    """ CNN Model """

    def __init__(self, embedding_size):
        """ Initialize the model """

        super(CNNModel, self).__init__()

        print("\n\n==> InItializating CNNModel()")

        # We use a pre-trained CNN model available under the PyTorch models repository: the ResNet 152 architecture
        # We remove the last layer of this pre-trained ResNet model 
        resnet = models.resnet152(pretrained=True)      
        module_list = list(resnet.children())[:-1] 
        self.__resnet_module = nn.Sequential(*module_list)

        # Replace it with a fully connected layer
        self.__linear_layer = nn.Linear(resnet.fc.in_features, embedding_size)

        # Followed by a batch normalization layer
        self.__batch_norm = nn.BatchNorm1d(embedding_size, momentum=0.01)
        
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
        """Return all the parameters of the model (used for Adam optimization)"""

        return list(self.__linear_layer.parameters()) + list(self.__batch_norm.parameters())

# Step 2 : LSTM

class LSTMModel(nn.Module):
    """ LSTM Model """

    def __init__(self, embedding_size, hidden_layer_size, vocabulary_size, num_layers, max_seq_len=20):
        """ Initialize the model """

        super(LSTMModel, self).__init__()

        print("\n\n==> InItializating LSTMModel()")

        # Embedding layer 
        self.__embedding_layer = nn.Embedding(vocabulary_size, embedding_size)
        
        # LSTM layer
        self.__lstm_layer = nn.LSTM(embedding_size, hidden_layer_size, num_layers, batch_first=True)
        # self.__lstm_layer = nn.GRU(embedding_size, hidden_layer_size, num_layers=2, dropout=0.8, bidirectional=True)

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
        """ Return all the parameters of the model (used for Adam optimization) """
        
        return list(self.parameters())

# Step 3 : CNN & LSTM

class FullModel(nn.Module):
    """ Full Model : CNN & LSTM """

    def __init__(self, vocabulary):
        """Combine the CNN and LSTM models."""
        
        super(FullModel, self).__init__()

        # Build the models
        self.__encoder_model = CNNModel(v.IMAGE_SHAPE[0]).to(v.DEVICE)
        self.__decoder_model = LSTMModel(v.IMAGE_SHAPE[0], v.IMAGE_SHAPE[0] + v.IMAGE_SHAPE[1], len(vocabulary), 1).to(v.DEVICE)        
            
        # Loss and optimizer
        self.__loss_criterion = nn.CrossEntropyLoss()

    def forward(self, images, captions, lens):
        """ Forward pass through the network. """

        feats   = self.__encoder_model(images)
        outputs = self.__decoder_model(feats, captions, lens)

        return outputs

    def sample(self, image):
        """ Make a prediction for a single image. """

        feats = self.__encoder_model(image)
        return self.__decoder_model.sample(feats)

    def loss(self, outputs, tgts):
        """ Compute the loss. """

        # outputs = F.softmax(outputs, dim=1)
        return self.__loss_criterion(outputs, tgts)

    # def zero_grad(self):
    #     """ Zero the gradients for the model. """
    #
    #     self.__decoder_model.zero_grad()
    #     self.__encoder_model.zero_grad()

    def getAllParameters(self):
        """  Return all the parameters of the model (used for Adam optimization) """

        return self.__decoder_model.getAllParameters() + self.__encoder_model.getAllParameters()

    def save(self, epoch : int):
        """ Save the values of the model : encoder + decoder. """

        # Create the model directory if not exists
        if not os.path.exists(v.OUTPUT_MODELS_PATH):
            os.makedirs(v.OUTPUT_MODELS_PATH)

        # Get the current time
        dateString = str(datetime.now())[0:19].replace("-", "_").replace(":", "_").replace(" ", "_") + "_"

        # Save the encoder model
        torch.save(
            self.__decoder_model.state_dict(), 
            os.path.join(v.OUTPUT_MODELS_PATH, 'decoder_{}_{}.ckpt'.format(dateString, epoch + 1))
        )

        # Save the decoder model
        torch.save(
            self.__encoder_model.state_dict(), 
            os.path.join(v.OUTPUT_MODELS_PATH, 'encoder_{}_{}.ckpt'.format(dateString, epoch + 1))
        )

    def load(self):
        """ Load the values of the model : encoder + decoder. """

        # Create the model directory if not exists
        if not os.path.exists(v.OUTPUT_MODELS_PATH):
            os.makedirs(v.OUTPUT_MODELS_PATH)

        # Find the latest model
        files = os.listdir(v.OUTPUT_MODELS_PATH)
        decoderFile = None
        encoderFile = None

        for filename in sorted(files):
            if filename.endswith('.ckpt'):
                if filename.startswith('decoder'):
                    decoderFile = filename
                if filename.startswith('encoder'):
                    encoderFile = filename

        if encoderFile and decoderFile:
            # Load the encoder model and the decoder model
            self.__encoder_model.load_state_dict(torch.load(v.OUTPUT_MODELS_PATH + encoderFile))
            self.__decoder_model.load_state_dict(torch.load(v.OUTPUT_MODELS_PATH + decoderFile))
            print("\n\n==> Loading models via FullModel.load()")
            print("Loaded models from {} and {}".format(encoderFile, decoderFile))

    def trainMode(self):
        """ Set the model in training mode : use batch_size """

        self.__encoder_model.train()
        self.__decoder_model.train()

    def evalMode(self):
        """ Set the model in training mode : image per image """

        self.__encoder_model.eval()
        self.__decoder_model.eval()