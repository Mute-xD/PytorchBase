import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

"""
Be advised, tensors in ops are pushed to GPU by default
"""
DEVICE = 'cuda'  # 'cuda' or 'cpu'


def convertImage(*tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    [0-255] -> [0-1]

    :param tensor: input a tensor
    :return: tensor
    """
    return [t / 255 for t in tensor]


def disConvertImage(*tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    [0-1] -> [0-255]

    :param tensor: input a tensor
    :return: tensor
    """
    return [(t * 255).int() for t in tensor]


def clip(*tensor: torch.Tensor, min_=0., max_=1.) -> List[torch.Tensor]:
    """
    torch.clip

    :param min_:
    :param max_:
    :param tensor:
    :return: converted tensor, loss
    """
    ret = []
    loss = 0
    for t in tensor:
        clipped = t.clip(min_, max_)
        ret.append(clipped)
        loss += F.l1_loss(clipped, t)
    ret.append(loss)
    return ret


def maxOutputLimit(*tensor, config):
    """
    [0-1] -> [0-0.78]

    :param config:
    :param tensor:
    :return:
    """

    background = convertImage(torch.tensor(config.BACKGROUND_RGB, dtype=torch.float).to(DEVICE))[0]
    return [background * t for t in tensor]


def BGR2RGB(*tensor):
    ret = []
    for t in tensor:
        temp = torch.zeros_like(t)
        if t.ndim == 4:
            temp[:, 0, :, :] = t[:, 2, :, :]
            temp[:, 1, :, :] = t[:, 1, :, :]
            temp[:, 2, :, :] = t[:, 0, :, :]
        elif t.ndim == 3:
            temp[0, :, :] = t[2, :, :]
            temp[1, :, :] = t[1, :, :]
            temp[2, :, :] = t[0, :, :]
        else:
            raise Exception('Wrong dim')
        ret.append(temp)
    return ret


def RGB2BGR(*tensor):
    return BGR2RGB(tensor[0])


def RGB2BRG(*tensor):
    ret = []
    for t in tensor:
        temp = torch.zeros_like(t)
        if t.ndim == 4:
            temp[:, 0, :, :] = t[:, 2, :, :]
            temp[:, 1, :, :] = t[:, 0, :, :]
            temp[:, 2, :, :] = t[:, 1, :, :]
        elif t.ndim == 3:
            temp[0, :, :] = t[2, :, :]
            temp[1, :, :] = t[0, :, :]
            temp[2, :, :] = t[1, :, :]
        else:
            raise Exception('Wrong dim')
        ret.append(temp)
    return ret


class RGBandHSV(nn.Module):
    """
    https://blog.csdn.net/Brikie/article/details/115086835
    """
    def __init__(self, eps=1e-8):
        super(RGBandHSV, self).__init__()
        self.eps = eps

    def rgb2hsv(self, img):
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

        hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 2] == img.max(1)[0]]
        hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 1] == img.max(1)[0]]
        hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 0] == img.max(1)[0]]) % 6

        hue[img.min(1)[0] == img.max(1)[0]] = 0.0
        hue = hue / 6

        saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + self.eps)
        saturation[img.max(1)[0] == 0] = 0

        value = img.max(1)[0]

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value], dim=1)
        return hsv

    @staticmethod
    def hsv2rgb(hsv):
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
        # 对出界值的处理
        h = h % 1
        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6)
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - (f * s))
        t = v * (1 - ((1 - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb
