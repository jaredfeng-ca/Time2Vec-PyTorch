from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ToyDataset(Dataset):
    def __init__(self, path):
        super(ToyDataset, self).__init__()
        
        df = pd.read_csv(path)
        self.x = df.drop("y", axis=1).values
        self.y = df["y"].values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return np.array(self.x[idx]), self.y[idx]

if __name__ == "__main__":
    path = './data/toy_dataset_v2.csv'
    dataset = ToyDataset(path)
    print(dataset)