import cstm_vars as v
import cstm_load as cstm_load
import cstm_train as cstm_train
import cstm_predict as cstm_predict
import cstm_model as cstm_model
from cstm_load import Vocab

if __name__ == "__main__":

    # Load the required string librairies
    v.init()

    if not v.TODO:
        # Bad options selected
        print("Please select at least one of the following options : install / train / test / predict...")

    if "install" in v.TODO:

        # Load the captions from "input_train_annotations_captions_train_path", generate the vocabulary and save it to "output_vocabulary_path"
        vocabulary = cstm_load.build_and_store_vocabulary()

        # Load the images from "images_path['input']", resize them to "image_shape" dimentions and save them in "images_path['output']"
        cstm_load.reshape_images()

        # Load the captions and the image paths and save them in a json file
        cstm_load.create_and_store_json_datasets(vocabulary)

    if "train" in v.TODO or "test" in v.TODO or "predict" in v.TODO:

        # Load the vocabulary
        vocabulary = Vocab.load()

        # Load the models (encoder + decoder)
        fullModel = cstm_model.FullModel(vocabulary)

        # Load the weights of the previous training
        fullModel.load()

        # Step 2 : Train the model
        if "train" in v.TODO:

            cstm_train.train(vocabulary, fullModel, withTestDataset=True)

        # Step 3 : Test the model 
        if "test" in v.TODO:

            cstm_train.test(vocabulary, fullModel, withTestDataset=True, withTrainDataset=False)

        # Step 3 : Make a prediction
        if "predict" in v.TODO:

            # Prepare an image
            img_tensor = cstm_load.load_image()

            # Predict the caption
            predicted_sentence = cstm_predict.predict(img_tensor, vocabulary, fullModel)
        
            # Print out the generated caption
            print(predicted_sentence["words"])
            