import yaml
import os
import shutil


class Config:
    def __init__(self, file):
        super().__init__()
        with open(file) as f:
            self._dict = yaml.load(f, Loader=yaml.FullLoader)

    def __getattr__(self, name):
        return self._dict[name]

    def getDict(self):
        return self._dict

    def __getstate__(self):
        return self._dict

    def __setstate__(self, state):
        self._dict = state


def configDumper(configPath, dumpPath):
    """
    Example:
        configDumper('./config.yml', '/20210304/')
    :return: None
    """
    if not os.path.exists(dumpPath):
        os.makedirs(dumpPath)
    shutil.copy(configPath, dumpPath + 'dumpConfig.txt')


def visTensor(tensor):
    """
    using matplotlib to visualize a tensor\n
    only work on tensor with shape (b, c, h, w)\n
    will show the first batch\n
    if multi-channel (like feature map) will print the first channel\n
    else regard as a three channel image of RGB\n
    :param tensor: Input a tensor to visualize
    :return: None
    """
    import matplotlib.pyplot as plt
    if tensor.shape[1] == 3:
        plt.imshow(tensor[0].to('cpu').permute(1, 2, 0).detach().numpy())
    else:
        plt.imshow(tensor.to('cpu').permute(1, 2, 0).detach().numpy())
    plt.show()


def saveTensorAsImg(tensor):
    import matplotlib.pyplot as plt
    import datetime
    time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    tensor = tensor.clip(0., 1.)
    if tensor.shape[1] == 3:
        plt.imsave(str(time)+'.png', tensor[0].to('cpu').permute(1, 2, 0).detach().numpy())
    else:
        plt.imsave(str(time)+'.png', tensor.to('cpu').permute(1, 2, 0).detach().numpy())


def imgLoader(path):
    import cv2
    import torch
    import ops
    img = cv2.imread(path)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # CV operation here
    # CV operation here
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # HWC to CHW
    img = ops.convertImage(img)[0]
    return img
