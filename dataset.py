import torch.utils.data as u_data
import tqdm
from torchvision import transforms
import cv2


class Dataset(u_data.Dataset):
    def __init__(self, config, fileList):
        super().__init__()
        self.config = config
        self.fileList = fileList
        self.toTensor = transforms.ToTensor()

    def __getitem__(self, item):
        img = cv2.imread(self.fileList[item])  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.toTensor(img)
        return img

    def __len__(self):
        return self.fileList.__len__()


class RAMDataset(u_data.Dataset):
    def __init__(self, config, fileList):
        super().__init__()
        self.config = config
        self.fileList = fileList
        self._images = []
        self.toTensor = transforms.ToTensor()
        for file in tqdm.tqdm(self.fileList, desc='Loading dataset to RAM'):
            img = cv2.imread(file)  # BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.toTensor(img)
            self._images.append(img)

    def __len__(self):
        return self.fileList.__len__()

    def __getitem__(self, item):
        return self._images[item]
