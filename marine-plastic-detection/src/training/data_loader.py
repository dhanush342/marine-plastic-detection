from torch.utils.data import Dataset, DataLoader
import os
import cv2
import yaml

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.load_data()

    def load_data(self):
        with open(os.path.join(self.data_dir, 'data.yaml'), 'r') as file:
            data_info = yaml.safe_load(file)
            self.images = data_info['images']
            self.labels = data_info['labels']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.images[idx])
        image = cv2.imread(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(train_dir, valid_dir, batch_size=16):
    train_dataset = CustomDataset(train_dir)
    valid_dataset = CustomDataset(valid_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader