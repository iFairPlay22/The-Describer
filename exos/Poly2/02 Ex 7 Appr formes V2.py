import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
import math

######################################################

# (x,y, category)
points = []
N = 9    # number of points per class
K = 3     # number of classes

for i in range(N):
    r = i / N
    for k in range(K):
        t = (i * 4 / N) + (k * 4) + random.uniform(0, 0.2)
        points.append([(r*math.sin(t), r*math.cos(t)), k])

points = [
    [(0.5, 0.4), 0],
    [(0.8, 0.3), 0],
    [(0.3, 0.8), 0],
    [(-.4, 0.3), 1],
    [(-.3, 0.7), 1],
    [(-.7, 0.2), 1],
    [(-.4, -.5), 1],
    [(0.7, -.4), 2],
    [(0.5, -.6), 2]
]

######################################################
#
#  outils d'affichage -  NE PAS TOUCHER


def DessineFond(model):
    iS = ComputeCatPerPixel(model)
    levels = [-1, 0, 1, 2]
    c1 = ('r', 'g', 'b')
    plt.contourf(XXXX, YYYY, iS, levels, colors=c1)


def DessinePoints():
    c2 = ('darkred', 'darkgreen', 'lightblue')
    for point in points:
        coord = point[0]
        cat = point[1]
        plt.scatter(coord[0], coord[1],  s=50, c=c2[cat],  marker='o')


XXXX, YYYY = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))


##############################################################
#
#  PROJET

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self._inputN = 2
        self._outputM = 2000
        self._outputO = K

        self._layer1 = torch.nn.Linear(self._inputN, self._outputM)
        self._layer2 = torch.nn.Linear(self._outputM, self._outputO)

    def forward(self, allCoords):
        v1 = self._layer1(allCoords)
        v2 = F.relu(v1)
        v3 = self._layer2(v2)
        return v3


def Error(predictedScores, correctCategories):
    res = torch.FloatTensor([0])

    indices = [i for i in range(len(predictedScores))]
    goodCategScore = predictedScores[indices, correctCategories]

    for j in range(3):
        currentCategScore = predictedScores[:, j]
        diff = torch.sub(currentCategScore, goodCategScore)
        positiveDiff = torch.max(torch.tensor([0]), diff)
        res += torch.sum(positiveDiff)

    return res


def ComputeCatPerPixel(model):

    allCoords = torch.zeros(XXXX.shape[0], XXXX.shape[1], 2)
    allCoords[:, :, 0] = torch.FloatTensor(XXXX)
    allCoords[:, :, 1] = torch.FloatTensor(YYYY)

    allScores = model.forward(allCoords)
    allMaxScores = torch.argmax(allScores, dim=2)
    allMaxScoresNa = allMaxScores.detach().numpy()

    return allMaxScoresNa


model = Net()
optim = O.SGD(model.parameters(), lr=0.5)

pointsTensor = torch.FloatTensor([list(x[0]) for x in points])
categoriesTensor = torch.tensor([x[1] for x in points])

for iteration in range(1000):

    # Calcul
    optim.zero_grad()  # remet à zéro le calcul du gradient
    predictions = model.forward(pointsTensor)  # démarrage de la passe Forward

    # predictions = np.argmax(predictions, dim=2)

    # choisit une function de loss de PyTorch
    ErrTot = Error(predictions, categoriesTensor)
    ErrTot.backward()  # effectue la rétropropagation
    optim.step()  # algorithme de descente

    if (iteration % 20 == 0):
        print("Iteration : ", iteration, " ErrorTot : ", ErrTot.item())

        # Affichage
        DessineFond(model)  # dessine les zones
        DessinePoints()  # dessine les données sous forme de points
        plt.title(str(iteration))  # insère le n° de l’itération dans le titre
        plt.pause(2)  # pause en secondes pour obtenir un refresh
        plt.show(block=False)  # affichage sans blocage du thread

        if (ErrTot.item() == 0.0):
            print("On a convergé ! :)")
            break

plt.pause(2)
