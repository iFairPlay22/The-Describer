import torch
from torchvision import transforms
import torchvision.transforms as transforms
import nltk

def init():
    """ Load the required string librairies """
    
    print("\n\n==> Download (nltk)...")
    nltk.download('punkt')
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    nltk.download('stopwords')
    print()

# Actions to do
TODO = [ 
    # "install" 
    # "train"
    # "test"
    "predict"
]

# Network constants
TOTAL_EPOCHS = 5
BATCH_SIZE = 128
STEP = 0.001

# Paths
IMAGES_PATH = [
    { "input" : './data_dir/train2014/', "output" : './data_dir/resized_images/train' },
    { "input" : './data_dir/val2014/',   "output" : './data_dir/resized_images/eval' },
]
CAPTIONS_PATH = [
    './data_dir/annotations/captions_train2014.json',
    './data_dir/annotations/captions_val2014.json'
]
JSON_PATH = './data_dir/jsons/'
JSON_FILE_NAME = 'datasets.json'
OUTPUT_VOCABULARY_PATH = './data_dir/vocabulary.pkl'
OUTPUT_MODELS_PATH = './models_dir/'
OUTPUT_PLOTS_PATH = './output_dir/'
INPUT_IMAGE_TO_TEST_PATH = './sample.jpg'

# Images
IMAGE_SHAPE = [256, 256]
TRANSFORM = transforms.Compose([ 
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(), 
    transforms.Normalize(
        (0.485, 0.456, 0.406), 
        (0.229, 0.224, 0.225)
    )
])

# Cuda or cpu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')