from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class Mnist(Dataset):
    def __init__(self, images, labels):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[255/2],
                                 std=[255])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.transform(Image.fromarray(self.images[index])), self.labels[index]
