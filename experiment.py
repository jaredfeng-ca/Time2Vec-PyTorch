from Data import ToyDataset
from periodic_activations import SineActivation, CosineActivation
import torch
from torch.utils.data import DataLoader
from Pipeline import AbstractPipelineClass
from torch import nn
from Model import Model

class ToyPipeline(AbstractPipelineClass):
    def __init__(self, model):
        self.model = model
    
    def train(self, n_epochs=30):
        loss_fn = nn.CrossEntropyLoss()

        dataset = ToyDataset()
        dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        for ep in range(n_epochs):
            for x, y in dataloader:
                optimizer.zero_grad()

                #x = x.unsqueeze(1)
                y_pred = self.model(x.float())
                loss = loss_fn(y_pred, y)

                loss.backward()
                optimizer.step()
                
                print("epoch: {}, loss:{}".format(ep, loss.item()))
    
    def preprocess(self, x):
        return x
    
    def decorate_output(self, x):
        return x

if __name__ == "__main__":
    pipe = ToyPipeline(Model("sin", 42))
    pipe.train()