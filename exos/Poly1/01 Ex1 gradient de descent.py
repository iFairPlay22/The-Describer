import math
import numpy as np
import matplotlib.pyplot as plt


# draw function g
def drawFunction(minx, maxx):
    axes = plt.gca()
    axes.set_xlim([minx, maxx])
    axes.set_ylim([-1, maxx])
    curveX = np.linspace(minx, maxx, 200)
    curveY = g(curveX)
    plt.plot(curveX, curveY)
    plt.title("Recherche du minimum")


def g(x):
    return x * x / 5 - 0.4 * x - 0.5

    # return np.vectorize(lambda y: math.sin(y))(x)


def gPrim(x):
    return 2 * x / 5 - 0.4

    # return np.vectorize(lambda y: math.cos(y))(x)

drawFunction(-5, 5)


x = 4
pas = 0.1

# recherche minimum function

for i in range(100):

    x -= pas * gPrim(x)

    # affichez la courbe de la fonction g
    plt.scatter(x, g(x),  s=50, c='red',  marker='x')

    # 0.1 second between each iteration, usefull for nice animation
    plt.pause(0.1)

plt.show()
