import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Le réseau de neurones (2)


class Net(nn.Module):

    # Initialisation du réseau
    def __init__(self):
        super(Net, self).__init__()

        # Couche : Convolution
        # Prends des images RGB de dimention 32x32 => NBx3x32x32
        # Utilise un kernel de dimention 3x3
        self.C1 = nn.Conv2d(in_channels=3, out_channels=32,
                            kernel_size=(3, 3), stride=1, padding=1)

        # Couche : Convolution
        # Prends une liste de dimention NBx32x32x32 => NBx64*32*32
        # Utilise un kernel de dimention 3x3
        self.C2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)

        # Couche : Fully Connected
        # Utilise NBx64x32x32 neurones afin d'analyser chaque convolution
        # de l'image et les classifie en 10 catégories
        self.FC1 = nn.Linear(64*32*32, 10)

        # Permet de passer d'une probabilité d'appartenance à un calcul de dispersion
        self.criterion = nn.CrossEntropyLoss()

    # Fonction de calcul de la sortie du réseau
    def forward(self, x):

        # On applique la couche de convolution
        x = self.C1(x)  # Dimention (NB,32,32,32)

        # On casse la linéarité avec la fonction d'activation Relu
        x = F.relu(x)  # Dimention (NB,32,32,32)

        # On applique la couche de convolution
        x = self.C2(x)  # Dimention (NB,64,32,32)

        # On casse la linéarité avec la fonction d'activation Relu
        x = F.relu(x)  # Dimention (NB,64,32,32)

        # On applique l'applatissage (dense)
        x = x.reshape((x.shape[0], 64*32*32))  # Dimention (NB,64x32x32)

        # On applique la couche fully connected
        x = self.FC1(x)  # Dimention (10)

        return x

    # Calcul de l'erreur totale
    def Loss(self, Scores, target):
        # Dans Scores, on a { idCategorie => score}
        # La catégorie prédite est idCatégorie pour index(max(Scores.values()))

        # Permet de transformer des scores par catégorie en probabilité
        # d'appartenances à chaque catégorie
        y = F.softmax(Scores, dim=1)

        # On calcule la dispersion des données. Plus les données sont dispersées,
        # et moins on est sur de notre résultat, donc plus l'erreur totale est
        # grande.
        err = self.criterion(y, target)

        return err

    # Calcul du nombre de catégories correctement classifiées
    def TestOK(self, Scores, target):

        # On récupérère les indices des scores maximaux
        pred = Scores.argmax(dim=1, keepdim=True)
        pred = pred.reshape(target.shape)

        # On compte vrai quand la catégorie prédite est égale à celle à prédire
        eq = pred == target

        # On compte le nombre de lignes à vrai
        nbOK = eq.sum().item()

        return nbOK

##############################################################################

# Entrainement du réseau


def TRAIN(args, model, train_loader, optimizer, epoch):

    # Pour chaque batch (sous ensemble des données d'entrainement)
    for batch_it, (data, target) in enumerate(train_loader):

        # On réinitialise le gradient
        optimizer.zero_grad()

        # On calcule les scores du réseau (prédictions)
        Scores = model.forward(data)

        # On calcule l'erreur totale
        loss = model.Loss(Scores, target)

        # On ajuste les poids du réseau
        loss.backward()
        optimizer.step()

# Test des performances du réseau
def TEST(model, test_loader):
    ErrTot = 0
    nbOK = 0
    nbImages = 0

    # On désactive le calcul du gradient lors des tests, car on ne
    # le calcule que pendant les sessions d'entrainement
    with torch.no_grad():

        # Pour chaque batch (sous ensemble des données de test)
        for data, target in test_loader:

            # On calcule les scores du réseau (prédictions)
            Scores = model.forward(data)

            # On calcule le nombre de lignes correctement prédites
            nbOK += model.TestOK(Scores, target)

            # On calcule l'erreur totale
            ErrTot += model.Loss(Scores, target)

            # On update le nombre d'images par rappor à la taille du batch
            nbImages += data.shape[0]

    # On calcule le nombre moyen de lignes correctement prédites
    pc_success = 100. * nbOK / nbImages
    print(f'\nTest set:   Accuracy: {nbOK}/{nbImages} ({pc_success:.2f}%)\n')

##############################################################################


def main(batch_size):

    # Normalisation des couleurs (channels)
    # On uniformise les niveaux de proportion des couleurs dans l'image afin d'obtenir
    # une distribution statistique ayant une moyenne de 0 et une variance de 1
    TRS = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # On charge les images du dataset d'images "CIFAR10"
    path = './data/ex9'
    TrainSet = datasets.CIFAR10(
        path, train=True,  download=True, transform=TRS)
    TestSet = datasets.CIFAR10(
        path, train=False, download=True, transform=TRS)

    # On charge les données par paquets de taille batch_size
    train_loader = torch.utils.data.DataLoader(TrainSet, batch_size)
    test_loader = torch.utils.data.DataLoader(TestSet, len(TestSet))

    # On initialise le réseau de neurones
    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # On teste le réseau la première fois afin de voir les performances initiales
    TEST(model,  test_loader)

    # On va lancer 20 sessions d'entrainement / test afin de converger
    for epoch in range(20):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f'Train Epoch: {epoch:3}')

        # On entraine le réseau
        TRAIN(batch_size, model,  train_loader, optimizer, epoch)

        # On teste le réseau
        TEST(model,  test_loader)


# Lancement du programme principal
main(batch_size=64)
