import torch
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f(x, y, a):
    x_less_y = torch.sub(x, y)
    x_abs_y = torch.abs(x_less_y)
    a_mul_xy = torch.mul(x_abs_y, a)
    return torch.min(a_mul_xy, torch.FloatTensor([1]))
    # return min(a*abs(x-y),1)


def Err(estim, ref):
    sub = torch.sub(estim, ref)
    return torch.pow(sub, 2)
    # return (estim-ref)**2


def getX(X):
    return np.array([abs(x[0] - x[1]) for x in X])


def addF(X, Y, color):
    x = getX(np.array(X.detach().numpy()))
    y = np.array(Y.detach().numpy())

    plt.figure()
    plt.scatter(x, y, color=color)


def displayF():
    plt.show()


if __name__ == '__main__':

    # Comment installer Pytorch : https://pytorch.org/get-started/locally/

    ####################################################################
    #
    #  Objectif

    # On se propose comme dans l'ex vu en cours de faire apprendre
    # à une fonction le comportement de l'opérateur booléen != (différent)
    # => 1 si différent   => 0 si égal

    # L'apprentissage doit s'effectuer sur le set d'échantillons suivant :
    # (4,2)   (6,-3)    (1,1)    (3,3)
    # Cela sous-entend que si l'apprentissage réussit, l'évaluation en dehors
    # de ces valeurs peut quand même etre erronée.

    # La fonction choisie pour l'apprentissage sera : min(a*|xi-yi|,1)
    # avec -a- comme unique paramètre d'apprentissage

    # la fonction d'erreur sera simplement : (fnt(xi,yi)-verite_i)^2

    ####################################################################
    #
    #  Aspect technique

    # Pour forcer les tenseurs à utiliser des nombres flotants,
    # nous utilisons la syntaxe suivante :

    X = torch.FloatTensor([[4, 2], [6, -3],  [1, 1], [3, 3]])
    X.requires_grad = False

    Ref = torch.FloatTensor([1, 1, 0, 0])
    Ref.requires_grad = False

    # pour créer notre paramètre d'apprentissage et préciser que pytorch
    # devra gérer son calcul de gradient, nous écrivons :

    a = torch.FloatTensor([.1])
    a.requires_grad = True

    pas = torch.FloatTensor([0.01])

    # Passe FORWARD :
    # Essayez de vous passer d'une boucle for.
    # Utilisez les tenseurs pour traiter tous les échantillons en parallèle.
    # Calculez les valeurs en sortie de la fonction.
    # Calculez l'erreur totale sur l'ensemble de nos échantillons.
    # Les fonctions mathématiques sur les tenseurs s'utilisent ainsi :
    # torch.abs(..) / torch.min(..)  / torch.sum(..)  ...

    # timeout = 10   # [seconds]
    # timeout_end = time.time() + timeout

    # while time.time() < timeout_end:
    for i in range(100):

        x1 = X[:, 0]
        x2 = X[:, 1]
        yEstim = f(x1, x2, a)
        err = Err(yEstim, Ref)
        ErrTot = torch.sum(err)

        # Passe BACKWARD :
        # Lorsque le calcul de la passe Forward est terminé,
        # nous devons lancer la passe Backward pour calculer le gradient.
        # Le calcul du gradient est déclenché par la syntaxe :

        ErrTot.backward()

        # GRADIENT DESCENT :
        # Effectuez la méthode de descente du gradient pour modifier la valeur
        # du paramètre d'apprentissage a. Etrangement, il faut préciser à Pytorch
        # d'arrêter de calculer le gradient de a en utilisant la syntaxe ci-après.
        # De plus, il faut réinitialiser le gradient de a à zéro manuellement :

        with torch.no_grad():
            # print(f'{a.grad.item()=}')
            a -= pas * a.grad
            a.grad.zero_()

        # A chaque itération, affichez la valeur de a et de l'erreur totale
        print(f'{a.item()=}')
        print(f'{ErrTot.item()=}')

    print(f'Valeur finale de a = {a.item()}')

    x1 = X[:, 0]
    x2 = X[:, 1]
    y = f(x1, x2, a)
    err = Err(y, Ref)

    addF(X, Ref, "green")
    addF(X, y, "red")
    displayF()
