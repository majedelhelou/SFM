from models.wgan_clipping import WGAN_CP
from utils.data_loader import FolderDataset
from torch.utils.data import DataLoader


def train():
    model = WGAN_CP(channels=1, generator_iters=400000)

    # modify the following line for the dataset folder
    train_set = FolderDataset(dataset='data/x2/')
    train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=128, shuffle=True)
    
    model.train(train_loader)


if __name__ == '__main__':
    train()
