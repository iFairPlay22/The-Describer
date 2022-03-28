
import numpy as np
import matplotlib.pyplot as plt
import time

class PercentagePlot:

    def __init__(self, xStart=0, xEnd=10, yStart=0, yEnd=100):

        plt.figure()
        plt.xlim(xStart, xEnd)
        plt.ylim(yStart, yEnd)

        self.__x = 0

    def addPoint(self, predictionScore):

        plt.scatter(self.__x, predictionScore)
        self.__x += 1

    def show(self):
        plt.show()
        plt.savefig('./images/' + str(time.time()) + '.png')


# p = PercentagePlot()
# p.addPoint(10)
# p.addPoint(20)
# p.show()