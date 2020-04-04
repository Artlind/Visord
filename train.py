import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from model import VGG16
import pandas as pd

dataset_path = "dataset.csv"
batch_size = 32
epochs = 50

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print('Initialisation de cuda')
    torch.cuda.init()
else:
    print('Mode CPU')
    DEVICE = torch.device('cpu')
softmax = nn.Softmax(dim=1).to(DEVICE)



def train(model, epochs, train_loader, eval_loader, weights):
    """
    Train model sur epochs epochs avec les train et eval loader aux formats
    pytorch dataloader
    """
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.5,
                                                     patience=3,
                                                     verbose=True,
                                                     threshold=0.0001,
                                                     threshold_mode='rel',
                                                     cooldown=0,
                                                     min_lr=0,
                                                     eps=1e-08)
    loss_function = nn.CrossEntropyLoss(weight=weights).to(DEVICE)
    for epoch in range(epochs):
        model.train()
        for batch, labels in train_loader:
            model.zero_grad()
            output = model(batch)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        loss_train = evaluate_train(model, train_loader, weights)
        scheduler.step(loss_train)
        loss_test, acc = evaluate_test(model, eval_loader)
        print("Accuracy : ", acc)
    return model
        
        
        
def evaluate_train(model, train_loader, weights):
    """
    Evalues le model sur le train_loader pour calculer la loss totale
    """
    loss_function = nn.CrossEntropyLoss(weight=weights).to(DEVICE)
    loss = 0.
    batchs = 0.
    for batch, labels in train_loader:
        batch += 1
        output = model(batch)
        loss += loss_function(output, labels)
    return loss/batchs


def evaluate_test(model, eval_loader):
    """
    Evalue le model sur le dataset d evaluation
    Calcule la loss et la precision
    """
    loss_function = nn.CrossEntropyLoss().to(DEVICE)
    loss = 0.
    batchs = 0.
    acc = 0.
    for batch, labels in eval_loader:
        batch += 1
        output = model(batch)
        loss += loss_function(output, labels)
        acc += (output.argmax(1) == labels).float().mean().to(DEVICE)
    return loss/batchs, acc/batchs


def main():
    model = VGG16(2, False)
    datataset_train = torch.tensor(pd.read_csv(dataset_path)["train"])
    datataset_eval = torch.tensor(pd.read_csv(dataset_path)["eval"])
    train_loader = torch.utils.data.DataLoader(datataset_train,
                                               batch_size,
                                               shuffle=True,
                                               drop_last=True)
    eval_loader = torch.utils.data.DataLoader(datataset_eval,
                                               batch_size,
                                               shuffle=True,
                                               drop_last=True)
    return train(model, epochs, train_loader, eval_loader, weights= (1,1))
