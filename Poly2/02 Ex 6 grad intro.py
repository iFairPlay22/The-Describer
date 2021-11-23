import torch
import time

def f(x,y,a):
    x_less_y = torch.sub(x,y)
    x_abs_y = torch.abs(x_less_y)
    a_mul_xy = torch.mul(x_abs_y,a)
    return torch.min(torch.FloatTensor([a_mul_xy,1]))
    # return min(a*abs(x-y),1)

def Err(estim,ref):
    sub = torch.sub(estim,ref)
    return torch.pow(sub,2)
    # return (estim-ref)**2
    
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

    # la fonction d'erreur sera simplement : |fnt(xi,yi)-verite_i|


    ####################################################################
    #
    #  Aspect technique

    # Pour forcer les tenseurs à utiliser des nombres flotants,
    # nous utilisons la syntaxe suivante :

    X = torch.FloatTensor([ [4,2], [6,-3],  [1,1] , [3,3] ])
    X.requires_grad = False

    Ref = torch.FloatTensor([1,1,0,0])
    Ref.requires_grad = False

    # pour créer notre paramètre d'apprentissage et préciser que pytorch
    # devra gérer son calcul de gradient, nous écrivons :

    a = torch.FloatTensor([ .1 ])
    a.requires_grad = True

    pas = torch.FloatTensor([0.1])

    # Passe FORWARD :
    # Essayez de vous passer d'une boucle for.
    # Utilisez les tenseurs pour traiter tous les échantillons en parallèle.
    # Calculez les valeurs en sortie de la fonction.
    # Calculez l'erreur totale sur l'ensemble de nos échantillons.
    # Les fonctions mathématiques sur les tenseurs s'utilisent ainsi :
    # torch.abs(..) / torch.min(..)  / torch.sum(..)  ...

    ErrTot = torch.FloatTensor([0])
    ErrTot.requires_grad = False


    #    timeout variable can be omitted, if you use specific value in the while condition
    timeout = 300   # [seconds]

    timeout_end = time.time() + timeout

    while time.time() < timeout_end:
        for i in range(len(X)):
            yEstim = f(X[i][0], X[i][1], a)
            yEstim.requires_grad = True

            erreur = Err(yEstim, Ref[i])
            ErrTot += erreur
            
        # Passe BACKWARD :
        # Lorsque le calcul de la passe Forward est terminé,
        # nous devons lancer la passe Backward pour calculer le gradient.
        # Le calcul du gradient est déclenché par la syntaxe :

        ErrTot.retain_grad()
        ErrTot.backward()

        # GRADIENT DESCENT :
        # Effectuez la méthode de descente du gradient pour modifier la valeur
        # du paramètre d'apprentissage a. Etrangement, il faut préciser à Pytorch
        # d'arrêter de calculer le gradient de a en utilisant la syntaxe ci-après.
        # De plus, il faut réinitialiser le gradient de a à zéro manuellement :

        with torch.no_grad() :
            print(f'{a.grad}')
            a -= pas *  a.grad
            a.grad.zero_()

        # A chaque itération, affichez la valeur de a et de l'erreur totale

        print(f'{a=}')
        print(f'{ErrTot=}')