from cstm_load import Vocab
import cstm_load as cstm_load
import cstm_predict as cstm_predict
import cstm_model as cstm_model
import deepl
import time
import torch
from torchvision import transforms
import torchvision.transforms as transforms


class IADecode:
    def __init__(self):
        self.image_shape = [256, 256]
        self.output_vocabulary_path = './data_dir/vocabulary.pkl'
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.vocabulary = Vocab.load(self.output_vocabulary_path)
        self.transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            )
        ])
        self.translator = deepl.Translator(
            "ceb1434a-ff88-92ab-b90e-de6e04fca8b3:fx")

        self.fullModel = cstm_model.FullModel(
            self.device, self.image_shape, self.vocabulary)
        self.fullModel.load()

    def checkSupportedLanguage(self, lang):
        lang = lang.upper()
        supported_languages = ["BG", "CS", "DA", "DE", "EL", "ES", "FI", "FR", "HU",
                               "ID", "IT", "JA", "LT", "LV", "NL", "PL", "PT-PT", "PT-BR", "RO", "RU", "SK", "SL", "SV", "TR", "ZH"]

        for l in supported_languages:

            if l in lang:
                return l
        return None

    def getPrediction(self, image_path, trad_lang="en"):

        # timer
        start = time.time()

        # Prepare an image
        img_tensor = cstm_load.load_image(
            image_path, self.device, self.transform)

        # Predict the caption
        predicted_sentence = cstm_predict.predict(
            img_tensor, self.vocabulary, self.fullModel)

        # Print out the generated caption

        predicted_sentence = predicted_sentence['words'][0]
        predicted_sentence = predicted_sentence.replace('<start> ', '')
        predicted_sentence = predicted_sentence.replace('<end>', '')

        duration2 = time.time() - start
        trad_lang = self.checkSupportedLanguage(trad_lang)
        if(trad_lang != None):
            translated_sentence = self.translator.translate_text(
                predicted_sentence, target_lang=trad_lang)
            predicted_sentence = translated_sentence.text
        duration = time.time() - start
        print("Time taken: " + str(duration) + "  "+str(duration2))
        return predicted_sentence
