import cv2
import numpy as np
import random
import torch.nn.functional as F
import torchvision.transforms.functional as tf

from PIL import Image


class Colors:
    """ ANSI color codes """
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"


def cut_mix(_input, mask_1, _refer, mask_2) -> (Image, Image):
    """
    :param _input: PIL.Image
    :param mask_1: PIL.Image
    :param _refer: PIL.Image
    :param mask_2: PIL.Image

    :returns: cut-mixed image
    """
    random_gen = random.Random()

    _input_np = np.array(_input)
    mask_1_np = np.array(mask_1)
    _refer_np = np.array(_refer)
    mask_2_np = np.array(mask_2)

    h1, w1, _ = _input_np.shape
    h2, w2, _ = _refer_np.shape

    # cutout positions
    rand_x = random_gen.random() * 0.75
    rand_y = random_gen.random() * 0.75
    rand_w = random_gen.random() * 0.5
    rand_h = random_gen.random() * 0.5

    cx_1 = int(rand_x * w1)  # range of [0, 0.5]
    cy_1 = int(rand_y * h1)
    cw_1 = int((rand_w + 0.25) * w1)  # range of [0.25, 0.75]
    ch_1 = int((rand_h + 0.25) * h1)

    cx_2 = int(rand_x * w2)
    cy_2 = int(rand_y * h2)
    cw_2 = int((rand_w + 0.25) * w2)
    ch_2 = int((rand_h + 0.25) * h2)

    if cy_1 + ch_1 > h1: ch_1 = h1 - cy_1  # push overflowing area
    if cx_1 + cw_1 > w1: cw_1 = w1 - cx_1

    cutout_img = _refer_np[cy_2:cy_2 + ch_2, cx_2:cx_2 + cw_2]
    cutout_mask = mask_2_np[cy_2:cy_2 + ch_2, cx_2:cx_2 + cw_2]

    cutout_img = cv2.resize(cutout_img, (cw_1, ch_1))
    cutout_mask = cv2.resize(cutout_mask, (cw_1, ch_1), interpolation=cv2.INTER_NEAREST)

    _input_np[cy_1:cy_1 + ch_1, cx_1:cx_1 + cw_1] = cutout_img
    mask_1_np[cy_1:cy_1 + ch_1, cx_1:cx_1 + cw_1] = cutout_mask

    return Image.fromarray(_input_np.astype(np.uint8)), Image.fromarray(mask_1_np.astype(np.uint8))


def grey_to_heatmap(img):
    """
    img: numpy.ndarray, or [0, 255] range of integer

    return: numpy.ndarray
    """

    heatmap = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap


def center_padding(img, target_size, pad_digit=0):
    """"
    Args:
        img: torch.Tensor or PIL.Image -> (c, h, w)
        target_size: list -> (target_h, target_w)
        pad_digit: int

    Returns: torch.Tensor or PIL.Image -> (c, target_h, target_w)
    """
    is_tensor = True
    if isinstance(img, Image.Image):
        img = tf.to_tensor(img)
        is_tensor = False
    _, in_h, in_w = img.shape

    assert target_size[0] >= in_h and target_size[0] >= in_w, 'target_size should be larger than input_size to pad'

    pad_left = (target_size[1] - in_w) // 2
    pad_right = pad_left if (target_size[1] - in_w) % 2 == 0 else pad_left + 1
    pad_top = (target_size[0] - in_h) // 2
    pad_bot = pad_top if (target_size[0] - in_h) % 2 == 0 else pad_top + 1

    tensor_padded = F.pad(img, [pad_left, pad_right, pad_top, pad_bot], 'constant', pad_digit)

    return tensor_padded if is_tensor else tf.to_pil_image(tensor_padded)


def remove_center_padding(img):
    """"
    Args:
        img: torch.Tensor or PIL.Image -> (b, c, h, w)

    Returns: torch.Tensor or PIL.Image -> (c, target_h, target_w)
    """
    is_tensor = True
    if isinstance(img, Image.Image):
        img = tf.to_tensor(img)
        is_tensor = False
    _, in_h, in_w = img[0].shape

    if img.shape[-1] == 608:
        target_size = (584, 565)
    elif img.shape[-1] == 704:
        target_size = (605, 700)
    elif img.shape[-1] == 1024:
        target_size = (960, 999)
    elif img.shape[-1] == 640:
        target_size = (768, 640)
    else:
        raise ValueError('Input shape should be involved in [608, 704, 1024]')

    assert target_size[0] <= in_h and target_size[1] <= in_w, 'target_size should be smaller than input_size'

    pad_left = abs((target_size[1] - in_w) // 2)
    pad_top = abs((target_size[0] - in_h) // 2)

    tensor_unpadded = img[:, :, pad_top:pad_top + target_size[0], pad_left:pad_left + target_size[1]]
    _, _, out_h, out_w, = tensor_unpadded.shape

    assert target_size[0] == out_h and target_size[1] == out_w, 'target_size should be same with input_size'

    return tensor_unpadded if is_tensor else tf.to_pil_image(tensor_unpadded)


class TrainerCallBack:

    def train_callback(self):
        pass

    def iteration_callback(self):
        pass