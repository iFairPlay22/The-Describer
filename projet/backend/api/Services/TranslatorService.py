import deepl
from dotenv import dotenv_values


class Translator:
    def __init__(self):
        config = dotenv_values("./config/.env.local")
        self.translator = deepl.Translator(
            config["DEEPL_KEY"])

    def __checkSupportedLanguage(self, lang):
        lang = lang.upper()
        supported_languages = ["BG", "CS", "DA", "DE", "EL", "ES", "FI", "FR", "HU",
                               "ID", "IT", "JA", "LT", "LV", "NL", "PL", "PT-PT", "PT-BR", "RO", "RU", "SK", "SL", "SV", "TR", "ZH"]

        for l in supported_languages:

            if l in lang:
                return l
        return None

    def translate(self, text, lang):
        lang = self.__checkSupportedLanguage(lang)
        if(lang != None):
            translated_sentence = self.translator.translate_text(
                text, target_lang=lang)
            return translated_sentence.text
        return text
