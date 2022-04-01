import time

class Timer(object):

    def __init__(self):
        self.__start = self.__getCurrentTime()

    def __getCurrentTime():
        return time.time()

    def stop(self):
        self.__end = time.time()

    def getElapsedTime(self):
        return self.__end - self.__start

    def print(self):
        return print("\n\n==> Time elapsed: {}".format(self.__getElapsedTime()))
