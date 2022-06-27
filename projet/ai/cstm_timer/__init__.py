import time

class Timer(object):
    """ Allows to count the time elapsed since the timer was started """

    def __init__(self):
        """ Initializes the timer """
        self.start()

    def __getCurrentTime():
        """ Return the current time """

        return time.time()

    def start(self):
        """ Starts the timer """

        self.__start = self.__getCurrentTime()

    def stop(self):
        """ Stops the timer """

        self.__end = self.__getCurrentTime()

    def getElapsedTime(self):
        """ Returns the elapsed time """
        return self.__end - self.__start

    def print(self):
        """ Prints the elapsed time """

        return print("\n\n==> Time elapsed: {}".format(self.__getElapsedTime()))

