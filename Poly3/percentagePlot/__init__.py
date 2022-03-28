
import numpy as np
import matplotlib.pyplot as plt

class PercentagePlot:

    def __init__(self, xStart=0, xEnd=10, yStart=0, yEnd=100):

        plt.figure()
        plt.xlim(xStart, xEnd)
        plt.ylim(yStart, yEnd)

        self.__x = 0

    def addPoint(self, predictionScore):

        plt.scatter(self.__x, predictionScore)
        # plt.pause(60)

        self.__x += 1

    def show(self):
        plt.show()

# p = PercentagePlot()
# p.addPoint(10)
# p.addPoint(20)
# p.show()