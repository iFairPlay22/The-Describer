from Models.IADecode import IADecode


class IADecodeManager:
    def __init__(self, count):
        self.decoders = decoders = [None]*count
        for i in range(count):
            decoders[i] = IADecode()
        self.decodersUsed = [False]*4
        self.count = count

    def __selectDecoder(self):
        for i in range(self.count):
            if not self.decodersUsed[i]:
                self.decodersUsed[i] = True
                return (self.decoders[i], i)
        return None, -1

    def __freeDecoder(self, i):
        self.decodersUsed[i] = False

    def getPrediction(self, path, lang):
        decoder = None
        while(decoder == None):
            decoder, decoder_id = self.__selectDecoder()

        prediction = decoder.getPrediction(path, lang)
        self.__freeDecoder(decoder_id)
        return prediction
