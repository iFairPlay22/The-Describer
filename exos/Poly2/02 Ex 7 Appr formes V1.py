import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O

######################################################

# (x,y, category)
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

# Nous devons apprendre 3 catégories : 0 1 ou 2 suivant ce couple (x,y)

# Pour chaque échantillon, nous avons comme information [(x,y),cat]

# Construisez une couche Linear pour un échantillon prédit un score pour chaque catégorie

# Le plus fort score est associé à la catégorie retenue
# Pour calculer l'erreur, on connait la bonne catégorie k de l'échantillon de l'échantillon.
# On calcule Err = Sigma_(j=0 à nb_cat) max(0,Sj-Sk)  avec Sj score de la cat j

# Comment interpréter cette formule :
# La grandeur Sj-Sk nous donne l'écart entre le score de la bonne catégorie et le score de la cat j.
# Si j correspond à k, la contribution à l'erreur vaut 0, on ne tient pas compte de la valeur Sj=k dans l'erreur
# Sinon Si cet écart est positif, ce n'est pas bon signe, car cela sous entend que le plus grand
#          score ne correspond pas à la bonne catégorie et donc on obtient un malus.
#          Plus le mauvais score est grand? plus le malus est important.
#       Si cet écart est négatif, cela sous entend que le score de la bonne catégorie est supérieur
#          au score de la catégorie courante. Tout va bien. Mais il ne faut pas que cela influence
#          l'erreur car l'algorithme doit corriger les mauvaises prédictions. Pour cela, max(0,.)
#          permet de ne pas tenir compte de cet écart négatif dans l'erreur.


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self._inputN = 2
        self._outputM = 3
        self._layer1 = torch.nn.Linear(self._inputN, self._outputM)

    def forward(self, allCoords):
        return self._layer1(allCoords)


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

    # Affichage
    print("Iteration : ", iteration, " ErrorTot : ", ErrTot.item())
    DessineFond(model)  # dessine les zones
    DessinePoints()  # dessine les données sous forme de points
    plt.title(str(iteration))  # insère le n° de l’itération dans le titre
    plt.pause(2)  # pause en secondes pour obtenir un refresh
    plt.show(block=False)  # affichage sans blocage du thread

    if (ErrTot.item() == 0.0):
        print("On a convergé ! :)")
        break

plt.pause(2)
