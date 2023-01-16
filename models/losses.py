import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import torch.nn.functional as F

from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve


class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        y = y.squeeze(1)

        return self.loss(x, y)


class KLDivergence(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(KLDivergence, self).__init__()
        self.loss = nn.KLDivLoss(reduction=reduction)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, y):
        x = self.log_softmax(x)

        return self.loss(x, y)


class KLDivergenceLogit(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(KLDivergenceLogit, self).__init__()
        self.loss = nn.KLDivLoss(reduction=reduction)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        x = self.log_softmax(x)
        y = self.softmax(y)

        return self.loss(x, y)


# Distance Transform MSE Loss
class DTMSELoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self):
        super(DTMSELoss, self).__init__()
        self.threshold = 0.5
        self.mse = nn.MSELoss()

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            img_mask = img[batch] > self.threshold
            img_mask_dt = edt(img_mask)

            field[batch] = img_mask_dt

        return field

    def forward(self, pred: torch.Tensor, target: torch.Tensor, debug=False) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        pred_dt = torch.from_numpy(self.distance_field(pred.cpu().detach().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.cpu().detach().numpy())).float()

        loss = self.mse(pred_dt, target_dt)

        return loss


class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = torch.tensor(alpha)

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        pred_dt = torch.from_numpy(self.distance_field(pred.cpu().detach().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.cpu().detach().numpy())).float()

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error.cpu() * distance.cpu()
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss


class HausdorffERLoss(nn.Module):
    """Binary Hausdorff loss based on morphological erosion"""

    def __init__(self, alpha=2.0, erosions=10, **kwargs):
        super(HausdorffERLoss, self).__init__()
        self.alpha = alpha
        self.erosions = erosions
        self.prepare_kernels()

    def prepare_kernels(self):
        cross = np.array([cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))])
        bound = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

        self.kernel2D = cross * 0.2
        self.kernel3D = np.array([bound, cross, bound]) * (1 / 7)

    @torch.no_grad()
    def perform_erosion(
        self, pred: np.ndarray, target: np.ndarray, debug
    ) -> np.ndarray:
        bound = (pred - target) ** 2

        if bound.ndim == 5:
            kernel = self.kernel3D
        elif bound.ndim == 4:
            kernel = self.kernel2D
        else:
            raise ValueError(f"Dimension {bound.ndim} is nor supported.")

        eroted = np.zeros_like(bound)
        erosions = []

        for batch in range(len(bound)):

            # debug
            erosions.append(np.copy(bound[batch][0]))

            for k in range(self.erosions):

                # compute convolution with kernel
                dilation = convolve(bound[batch], kernel, mode="constant", cval=0.0)

                # apply soft thresholding at 0.5 and normalize
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0

                if erosion.ptp() != 0:
                    erosion = (erosion - erosion.min()) / erosion.ptp()

                # save erosion and add to loss
                bound[batch] = erosion
                eroted[batch] += erosion * (k + 1) ** self.alpha

                if debug:
                    erosions.append(np.copy(erosion[0]))

        # image visualization in debug mode
        if debug:
            return eroted, erosions
        else:
            return eroted

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        if debug:
            eroted, erosions = self.perform_erosion(
                pred.cpu().numpy(), target.cpu().numpy(), debug
            )
            return eroted.mean(), erosions

        else:
            eroted = torch.from_numpy(
                self.perform_erosion(pred.cpu().numpy(), target.cpu().detach().numpy(), debug)
            ).float()

            loss = eroted.mean()

            return loss


# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

        self.loss = nn.MSELoss()

    def forward(self, x, y):
        y = y.float()

        return self.loss(x, y)


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, x, y):
        y = y.float()
        BCE = F.binary_cross_entropy(x, y, reduction='mean')

        return BCE


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        targets = targets.float()
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class JaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(JaccardLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class FocalBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalBCELoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        targets = targets.float()
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** gamma

        return FocalTversky


class JSDivergence(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(JSDivergence, self).__init__()
        self.kld = KLDivergence(reduction=reduction)

    def forward(self, x, y):
        m = (x + y) / 2
        p = self.kld(x, m) / 2
        q = self.kld(y, m) / 2

        return p + q


class JSDivergenceLogit(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(JSDivergenceLogit, self).__init__()
        self.kld = KLDivergenceLogit(reduction=reduction)

    def forward(self, x, y):
        m = (x + y) / 2
        p = self.kld(x, m) / 2
        q = self.kld(y, m) / 2

        return p + q


class JSDivergenceBatch(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(JSDivergenceBatch, self).__init__()
        self.jsd = JSDivergence(reduction=reduction)

    def forward(self, x, y):
        x_batch, _, _, _ = x.shape

        for i in range(x_batch):
            pass

        return self.jsd(x, y)


class JSDivergenceLogitBatch(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(JSDivergenceLogitBatch, self).__init__()
        self.jsd = JSDivergenceLogit(reduction=reduction)

    def forward(self, x, y):
        x_batch, _, _, _ = x.shape

        loss_list = []

        for i in range(x_batch):
            loss_list.append(self.jsd(y, x[i]))

        return sum(loss_list) / len(loss_list)


class MSELoss_SSL(nn.Module):
    def __init__(self):
        super(MSELoss_SSL, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def softmax_mse_loss(self, inputs, targets):
        assert inputs.size() == targets.size(), f'{inputs.size()} {targets.size()}'
        inputs = F.softmax(inputs, dim=0)
        targets = F.softmax(targets, dim=0)

        return self.mse_loss(inputs, targets)

    def forward(self, x, y):
        assert y.shape[0] == 1, 'target batch size should be 1.'
        x_b, _, _, _ = x.shape
        y = y.float().squeeze(0)

        for idx in range(x_b):
            losses = [self.softmax_mse_loss(x[idx], y)]

        return sum(losses) / x_b
