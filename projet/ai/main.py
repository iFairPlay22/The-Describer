import spacy 
import nltk
import torch
from torchvision import transforms
import torchvision.transforms as transforms
from cstm_load import Vocab
import cstm_load as cstm_load
import cstm_train as cstm_train
import cstm_predict as cstm_predict
import cstm_model as cstm_model

def download():
    
    print("\n\n==> Download (nltk)...")
    nltk.download('punkt')
    nltk.download('omw-1.4')
    nltk.download('wordnet')

    print("\n\n==> Download (spacy)...")
    spacyEn = spacy.load('en_core_web_sm')

    return spacyEn

if __name__ == "__main__":

    # Network constants
    totalEpochs = 20
    batch_size = 128
    step = 0.001

    # String management
    spacyEn = download()

    # Paths
    images_path = [
        { "input" : './data_dir/train2014/', "output" : './data_dir/resized_images/test2014' },
        { "input" : './data_dir/val2014/',   "output" : './data_dir/resized_images/val2014' },
    ]
    captions_path = [
        './data_dir/annotations/captions_train2014.json',
        './data_dir/annotations/captions_val2014.json'
    ]
    output_vocabulary_path = './data_dir/vocabulary.pkl'
    output_models_path = './models_dir/'
    output_plot_path = './output_dir/'
    input_image_to_test_path = './sample.jpg'

    # Images
    image_shape = [256, 256]
    transform = transforms.Compose([ 
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(
            (0.485, 0.456, 0.406), 
            (0.229, 0.224, 0.225)
        )
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    todo = [ 
        # "install", 
        # "train",
        # "test"
        "predict"
    ]

    if "install" in todo:

        # Load the captions from "input_train_annotations_captions_train_path", generate the vocabulary and save it to "output_vocabulary_path"
        cstm_load.build_and_store_vocabulary(captions_path[0], output_vocabulary_path)

        # Load the images from "images_path['input']", resize them to "image_shape" dimentions and save them in "images_path['output']"
        cstm_load.reshape_images(images_path, image_shape)


    if "train" in todo or "test" in todo or "predict" in todo:

        # Load the vocabulary
        vocabulary = Vocab.load(output_vocabulary_path)

        # Load the models (encoder + decoder)
        fullModel = cstm_model.FullModel(device, image_shape, vocabulary)

        # Load the weights of the previous training
        fullModel.load(output_models_path)

        # Step 2 : Train the model
        if "train" in todo:

            cstm_train.train(totalEpochs, batch_size, step, vocabulary, fullModel, images_path, captions_path, output_models_path, output_plot_path, device, transform, spacyEn, withTestDataset=False)

        # Step 3 : Test the model 
        if "test" in todo:

            cstm_train.testAll(vocabulary, fullModel, batch_size, images_path, captions_path, device, transform, spacyEn, withTestDataset=True, withTrainDataset=False)

        # Step 3 : Make a prediction
        if "predict" in todo:

            # Prepare an image
            img_tensor = cstm_load.load_image(input_image_to_test_path, device, transform)

            # Predict the caption
            predicted_sentence = cstm_predict.predict(img_tensor, vocabulary, fullModel)
        
            # Print out the generated caption
            print(predicted_sentence["words"])
            