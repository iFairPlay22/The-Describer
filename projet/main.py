import os
import nltk
import pickle
import numpy as np
from PIL import Image
from collections import Counter
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence

print("\n\n==> Download nlkt...")
nltk.download('punkt')

annotations_captions_train_path = 'data_dir/annotations/captions_train2014.json'
vocabulary_path = './data_dir/vocabulary.pkl'

input_images_path = './data_dir/train2014/'
output_resized_images_path = './data_dir/resized_images/'
image_shape = [256, 256]

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
 
    # Get the length of the vocabulary
    def __len__(self):
        return len(self.__w2i)

    # Add a word to the vocabulary
    def add_token(self, token):
        if not token in self.__w2i:
            self.__w2i[token] = self.__index
            self.__i2w[self.__index] = token
            self.__index += 1

# Step 1: Build the vocabulary wrapper and save it to disk.

def build_vocabulary(json, threshold):

    # Load JSON
    coco = COCO(json)

    # Loading captions
    ids = coco.anns.keys()
    counter = Counter()
    
    # For every caption ids
    for i, id in tqdm(enumerate(ids)):

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

def load_store_vocabulary(input, output):

    print("\n\n==> load_store_vocabulary()")

    print("Load the data from '{}'".format(input))
    vocab = build_vocabulary(json=input, threshold=4)
    print("Total vocabulary size: {}".format(len(vocab)))

    print("Save the vocabulary wrapper to '{}'".format(output))
    with open(output, 'wb') as f:
        pickle.dump(vocab, f)

# Step 2: Resize the images

def reshape_image(image, shape):
    # Resize an image to the given shape
    return image.resize(shape, Image.ANTIALIAS)
 
def reshape_images(input, output, shape):

    print("\n\n==> reshape_images()")

    # We create the output directory if it does not exist
    if not os.path.exists(output):
        os.makedirs(output)
    print("Resize images in path '{}'".format(output))
 
    # Load the images
    images = os.listdir(input)
    for i, im in tqdm(enumerate(images)):
        with open(os.path.join(input, im), 'r+b') as f:
            with Image.open(f) as image:

                # Reshape them
                image = reshape_image(image, shape)

                # Save them
                image.save(os.path.join(output, im), image.format)

# Step 3 : Instanciate data loader

class CustomCocoDataset(data.Dataset):
    
    def __init__(self, rezized_image_folder_path, coco_json_path, vocabulary, transform=None):

        self.__rezized_image_folder_path = rezized_image_folder_path
        self.__data = COCO(coco_json_path)
        self.__indices = list(self.__coco_data.anns.keys())
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
        image = Image.open(os.path.join(self.__rezized_image_folder_path, image_path)).convert('RGB')

        # Apply the given transformation to the image
        if self.__transform is not None:
            image = self.__transform(image)

        # Step 2 => Get the caption and transform it to a list of integers 

        # Get the image caption
        caption = self.__data.anns[annotation_id]['caption']

        # Convert caption (string) to word ids.
        word_tokens = nltk.tokenize.word_tokenize(str(caption).lower())

        caption = []
        caption.append(self.__vocabulary('<start>'))
        caption.extend([self.__vocabulary(token) for token in word_tokens])
        caption.append(self.__vocabulary('<end>'))

        # Transform to a Torch tensor
        caption_torch = torch.Tensor(caption)       

        return image, caption_torch
 
    def __len__(self):
        return len(self.__indices)
 
#  Creates mini-batch tensors from the list of tuples (image, caption)
def collate_function(data_batch):

    # Sort a data list by caption length (descending order).
    data_batch.sort(key=lambda d: len(d[1]), reverse=True)
    imgs, caps = zip(*data_batch)
 
    # Convert a list of 3D tensors (list(tensor(3, 256, 256))) to a 4D tensor (tensor(<batch_size>, 3, 256, 256))
    imgs = torch.stack(imgs, 0)
 
    # Merge captions (from list of 1D tensors to 2D tensor), similar to merging of images donw above.

    # Get the length of each captions
    cap_lens = [len(cap) for cap in caps]

    # Pad each caption to the longest caption in the batch (with 0)
    tgts = torch.zeros(len(caps), max(cap_lens)).long()

    # Complete the tensor with the captions ids
    for i, cap in enumerate(caps):

        # The end index is the size of the caption
        end = cap_lens[i]

        # Replace the 0 (in [0, end]) with the captions ids
        tgts[i, :end] = cap[:end]        

    # Returns:
    #         imgs: torch tensor of shape (batch_size, 3, 256, 256).
    #         tgts: torch tensor of shape (batch_size, padded_length).
    #         cap_lens: list; valid length for each padded caption.

    return imgs, tgts, cap_lens
 
# Returns torch.utils.data.DataLoader for custom coco dataset.
# This will return (images, captions, lengths) for each iteration.
#   images: a tensor of shape (batch_size, 3, 224, 224).
#   captions: a tensor of shape (batch_size, padded_length).
#   lengths: a list indicating valid length for each caption. length is (batch_size).
def get_loader(data_path, coco_json_path, vocabulary, transform, batch_size, shuffle, num_workers):

    # COCO caption dataset
    coco_dataser = CustomCocoDataset(data_path, coco_json_path, vocabulary, transform)
    
    # Data loader for COCO dataset
    custom_data_loader = torch.utils.data.DataLoader(dataset=coco_dataser, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_function)
    
    return custom_data_loader

# Step 4 : Combining CNNs and LSTMs

class CNNModel(nn.Module):

    def __init__(self, embedding_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(CNNModel, self).__init__()

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

# The LSTM model consists of an LSTM layer followed by a fully connected linear layer
class LSTMModel(nn.Module):

    def __init__(self, embedding_size, hidden_layer_size, vocabulary_size, num_layers, max_seq_len=20):
        """Set the hyper-parameters and build the layers."""
        super(LSTMModel, self).__init__()

        # Embedding layer 
        self.__embedding_layer = nn.Embedding(vocabulary_size, embedding_size)
        
        # LSTM layer
        self.__lstm_layer = nn.LSTM(embedding_size, hidden_layer_size, num_layers, batch_first=True)
        
        # Fully connected linear layer
        self.__linear_layer = nn.Linear(hidden_layer_size, vocabulary_size)
        
        # Max length of the predited caption
        self.__max_seq_len = max_seq_len
        
    def forward(self, input_features, caps, lens):
        """Decode image feature vectors and generates captions."""

        # We apply the embedding layer
        embeddings = self.__embedding_layer(caps)
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

if __name__ == "__main__":

    # Load the captions from "annotations_captions_train_path", generate the vocabulary and save it to "vocabulary_path"
    load_store_vocabulary(annotations_captions_train_path, vocabulary_path)

    # Load the images from "input_images_path", resize them to "image_shape" dimentions and save them in "output_resized_images_path"
    reshape_images(input_images_path, output_resized_images_path, image_shape)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Create model directory
    if not os.path.exists('models_dir/'):
        os.makedirs('models_dir/')

        
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                            (0.229, 0.224, 0.225))])


    # Load vocabulary wrapper
    with open('data_dir/vocabulary.pkl', 'rb') as f:
        vocabulary = pickle.load(f)

        
    # Build data loader
    custom_data_loader = get_loader('data_dir/resized_images', 'data_dir/annotations/captions_train2014.json', vocabulary, 
                            transform, 128,
                            shuffle=True, num_workers=2) 


    # Build the models
    encoder_model = CNNModel(256).to(device)
    decoder_model = LSTMModel(256, 512, len(vocabulary), 1).to(device)
    
        
    # Loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    parameters = list(decoder_model.parameters()) + list(encoder_model.linear_layer.parameters()) + list(encoder_model.batch_norm.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.001)


    # Train the models
    total_num_steps = len(custom_data_loader)
    for epoch in tqdm(range(5)):
        for i, (imgs, caps, lens) in enumerate(custom_data_loader):
    
            # Set mini-batch dataset
            imgs = imgs.to(device)
            caps = caps.to(device)
            tgts = pack_padded_sequence(caps, lens, batch_first=True)[0]
    
            # Forward, backward and optimize
            feats = encoder_model(imgs)
            outputs = decoder_model(feats, caps, lens)
            loss = loss_criterion(outputs, tgts)
            decoder_model.zero_grad()
            encoder_model.zero_grad()
            loss.backward()
            optimizer.step()
    
            # Print log info
            if i % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                    .format(epoch, 5, i, total_num_steps, loss.item(), np.exp(loss.item()))) 
    
            # Save the model checkpoints
            if (i+1) % 1000 == 0:
                torch.save(decoder_model.state_dict(), os.path.join(
                    'models_dir/', 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder_model.state_dict(), os.path.join(
                    'models_dir/', 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
        