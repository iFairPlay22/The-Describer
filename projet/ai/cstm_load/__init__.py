import os
from tqdm import tqdm
import nltk
import pickle
from PIL import Image
from collections import Counter
from pycocotools.coco import COCO

class Vocab(object):
    """ Vocabulary wrapper """

    def __init__(self):
        """ Create a new Vocabulary. """

        # Word to index
        self.__w2i = {}

        # Index to word
        self.__i2w = {}

        # Current index
        self.__index = 0
 
    def __call__(self, token : str):
        """ Get the index of the token, of <unk> if not exists """

        if not token in self.__w2i:
            return self.__w2i['<unk>']

        return self.__w2i[token]

    def getToken(self, index : int):
        """ Get the token that corresponds to the index """

        return self.__i2w[index]
 
    def __len__(self):
        """ Get the length of the vocabulary """

        return len(self.__w2i)

    def add_token(self, token : str):
        """ Add a token to the vocabulary """

        if not token in self.__w2i:
            self.__w2i[token] = self.__index
            self.__i2w[self.__index] = token
            self.__index += 1

    def translate(self, sentence : list):
        """ Translate list of indexes to a sentance (list of tokens) """

        words = []
        for word in sentence:
            word = self.getToken(word)
            words.append(word)
            if word == '<end>':
                break

        return " ".join(words)

    def save(self, output : str):
        """ Save the vocabulary wrapper """
        
        print("\n\nSave the vocabulary wrapper to '{}'".format(output))
        with open(output, 'wb') as f:
            pickle.dump(self, f)

    def load(output : str):
        """ Load the vocabulary wrapper """
    
        print("\n\nLoad the vocabulary wrapper from '{}'".format(output))
        with open(output, 'rb') as f:
            return pickle.load(f)

# Step 1: Build the vocabulary wrapper and save it to disk.
def build_vocabulary(json, threshold=4):
    """ Construct the vocabulary wrapper and return it """

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
    vocab.add_token('<pad>')    # \n
    vocab.add_token('<start>')  # start
    vocab.add_token('<end>')    # end
    vocab.add_token('<unk>')    # unknowed word (not exists in the vocabulary i.e.)

    # Keep the tokens that appears more than threshold 
    for token, cnt in counter.items():
        if threshold <= cnt:
            vocab.add_token(token)
        
    return vocab


def build_and_store_vocabulary(input : str, output : str):
    """ Build the vocabulary wrapper and save it to disk """

    print("\n\n==> build_and_store_vocabulary()")

    print("Load the data from '{}'".format(input))
    vocab = build_vocabulary(input)
    print("Total vocabulary size: {}".format(len(vocab)))

    vocab.save(output)

# Step 2: Resize the images

def load_image(input_image_file_path : str, device, transform=None):
    """ Load and transform an image """

    img = Image.open(input_image_file_path)
    img = img.resize([224, 224], Image.LANCZOS).convert('RGB')
    
    if transform is not None:
        img = transform(img).unsqueeze(0)
    
    return img.to(device)

def reshape_image(image, shape):
    """ Reshape an image """
    
    # Resize an image to the given shape
    return image.resize(shape, Image.ANTIALIAS)
 
def reshape_images(io, shape):
    """ Reshape all images in a dataset """

    print("\n\n==> reshape_images()")
    
    # Load the images
    for obj in io:

        input = obj["input"]
        output = obj["output"]

        # We create the output directory if it does not exist
        if not os.path.exists(output):
            os.makedirs(output)
            
        print("Resize images in path '{}'".format(output))
    
        # For every image in the input directory
        images = os.listdir(input)
        for im in tqdm(images):

            with open(os.path.join(input, im), 'r+b') as f:
                with Image.open(f) as image:

                    # Reshape them
                    image = reshape_image(image, shape)

                    # Save them to the output directory
                    image.save(os.path.join(output, im), image.format)
