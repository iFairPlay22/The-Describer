import pickle
from PIL import Image

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

    def getToken(self, index):
        return self.__i2w[index]
 
    # Get the length of the vocabulary
    def __len__(self):
        return len(self.__w2i)

    def translate(self, sentence):

        words = []
        for word in sentence:
            word = self.getToken(word)
            words.append(word)
            if word == '<end>':
                break

        return " ".join(words)

    def load(output):
    
        print("\n\nLoad the vocabulary wrapper from '{}'".format(output))
        with open(output, 'rb') as f:
            return pickle.load(f)

def load_image(input_image_file_path, device, transform=None):
    img = Image.open(input_image_file_path)
    img = img.resize([224, 224], Image.LANCZOS).convert('RGB')
    
    if transform is not None:
        img = transform(img).unsqueeze(0)
    
    return img.to(device)
