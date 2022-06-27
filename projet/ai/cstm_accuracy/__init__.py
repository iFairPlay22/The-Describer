import cstm_vars as v
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import re

class AccuracyBasedOnSynonyms():

    def getRatios():
        return list([ { "min": 0.1 * i, "sum": 0 } for i in range(0, 11) ])

    def __init__(self):

        self.__ratios = AccuracyBasedOnSynonyms.getRatios()
        self.__cummulatedRatios = AccuracyBasedOnSynonyms.getRatios()
        self.__stopWords = stopwords.words('english') + ["a"]
        self.__allRatios = []

    def calculateAccuracy(self, s1 : str, s2 : str):

        # Remove punctuation
        s1 = re.sub(r'[^\w\s]', '', " ".join(s1)).split(" ")
        s2 = re.sub(r'[^\w\s]', '', " ".join(s2)).split(" ")

        # Remove stop words 
        s1KeyWords = set( w for w in s1  if w and not(w in self.__stopWords) )
        if len(s1KeyWords) == 0:
            self.__ratios[0]["sum"] += 1
            self.__allRatios.append(0)
            return ;

        # Check if we can find sense similarities
        commonKeyWordsWithSynonyms = [
            syn
            for s1KeyWord in s1KeyWords 
            for syn in self.__getSynonyms(s1KeyWord)
            if any(s2[idx : idx + len(syn)] == syn for idx in range(len(s2) - len(syn) + 1))
        ]

        # Update ratios
        currentRatio = min(1.0, max(0.0, len(commonKeyWordsWithSynonyms) / len(s1KeyWords)))
        self.__allRatios.append(currentRatio)

        # Classic ratios
        for i in range(len(self.__cummulatedRatios) - 1, 0 - 1, -1):
            if self.__ratios[i]["min"] <= currentRatio:
                self.__ratios[i]["sum"] += 1
                break

        # Cummulated ratios
        for ratio in self.__cummulatedRatios:
            if ratio["min"] <= currentRatio:
                ratio["sum"] += 1
                
    def __getSynonyms(self, word : str):
        """ Returns the synonyms of a word """

        if not word:
            return []

        all_synonyms_with_ = { lemm.name() for syn in wordnet.synsets(word) for lemm in syn.lemmas() }
        all_synonyms_with_.add(word)
        
        if word[-1] != "s":
            all_synonyms_with_.add(word + 's')
            all_synonyms_with_.add(word + 'es')
        else:
            all_synonyms_with_.add(word[:-1])

        return list(map(lambda word_with_: word_with_.split("_"), all_synonyms_with_))

    def getRatioAverage(self):
        return sum(self.__allRatios) * 100 / len(self.__allRatios)

    def getDetailedRatios(self):
        return self.__ratios, self.__cummulatedRatios