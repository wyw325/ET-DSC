import  numpy as np
import torch.nn as nn
import torch
from skimage import measure
import  numpy
class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos=np.zeros(self.bins+1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg,i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos

    def get(self):

        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)


        return tp_rates, fp_rates, recall, precision

    def reset(self):

        self.tp_arr   = np.zeros([11])
        self.pos_arr  = np.zeros([11])
        self.fp_arr   = np.zeros([11])
        self.neg_arr  = np.zeros([11])
        self.class_pos= np.zeros([11])



class PD_FA():
    def __init__(self, nclass, bins):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins+1)
        self.PD = np.zeros(self.bins + 1)
        self.target= np.zeros(self.bins + 1)
    def update(self, preds, labels):

        for iBin in range(self.bins+1):
            score_thresh = iBin * (1/self.bins)
            predits  = np.array((preds > score_thresh).cpu()).astype('int64')
            predits  = np.reshape (predits,  (256,256))
            labelss = np.array((labels).cpu()).astype('int64') # P
            labelss = np.reshape (labelss , (256,256))

            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss , connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin]    += len(coord_label)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match   = []
            self.dismatch         = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)

                        del coord_image[m]
                        break

            self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
            self.FA[iBin]+=np.sum(self.dismatch)
            self.PD[iBin]+=len(self.distance_match)

    def get(self,img_num):

        Final_FA =  self.FA / ((256 * 256) * img_num)
        Final_PD =  self.PD /self.target

        return Final_FA,Final_PD


    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])


class F1():
    def __init__(self, nclass):
        super(F1, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        preds = preds.argmax(1)  # [B, H, W]
        labels = labels  # [B, H, W]

        for cls in range(self.nclass):
            pred_inds = (preds == cls)
            label_inds = (labels == cls)

            intersection = (pred_inds & label_inds).sum().item()
            pred_sum = pred_inds.sum().item()
            label_sum = label_inds.sum().item()

            self.total_inter[cls] += intersection
            self.total_pred[cls] += pred_sum
            self.total_label[cls] += label_sum

    def get(self):
        dsc = []
        for cls in range(self.nclass):
            denominator = self.total_pred[cls] + self.total_label[cls]
            if denominator == 0:
                dsc.append(1.0)  # if no label and no prediction, consider perfect
            else:
                dsc.append(2.0 * self.total_inter[cls] / (denominator + np.spacing(1)))
        mean_dsc = np.mean(dsc)
        return mean_dsc, dsc  # return mean DSC and per-class DSC

    def reset(self):
        self.total_inter = [0 for _ in range(self.nclass)]
        self.total_pred = [0 for _ in range(self.nclass)]
        self.total_label = [0 for _ in range(self.nclass)]


class mIoU():

    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union


    def get(self):

        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):

        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


class nIoU():
    def __init__(self):
        self.total_niou = 0.0
        self.total_back_niou = 0.0
        self.count = 0

    def update(self, preds, labels):
        """
        preds: Tensor of shape (B, 1, H, W), float, logits or probs
        labels: Tensor of shape (B, 1, H, W), int or float, binary ground truth
        """
        preds = preds.cpu()
        labels = labels.cpu()
        # 如果 preds 是 logits，就先过 sigmoid
        if preds.max() > 1.0:
            preds = torch.sigmoid(preds)

        # 二值化预测
        preds = (preds > 0.5).float()
        labels = (labels > 0.5).float()

        # 背景的二值化 (1 - 前景)
        back_preds = 1.0 - preds
        back_labels = 1.0 - labels

        B = preds.size(0)

        # 展平
        preds = preds.view(B, -1)
        labels = labels.view(B, -1)
        back_preds = back_preds.view(B, -1)
        back_labels = back_labels.view(B, -1)

        # 逐样本计算 TP, P, T
        TP = (preds * labels).sum(dim=1)           # TP[i]
        P = preds.sum(dim=1)                        # P[i]
        T = labels.sum(dim=1)                       # T[i]

        iou = TP / (T + P - TP + 1e-10)             # 防止除0

        # 计算背景的nIoU
        back_TP = (back_preds * back_labels).sum(dim=1)  # 背景的TP[i]
        back_P = back_preds.sum(dim=1)  # 背景的P[i]
        back_T = back_labels.sum(dim=1)  # 背景的T[i]
        back_iou = back_TP / (back_T + back_P - back_TP + 1e-10)  # 防止除0


        self.total_niou += iou.sum().item()
        self.total_back_niou += back_iou.sum().item()
        self.count += B

    def get(self):
        niou = self.total_niou / self.count if self.count > 0 else 0.0
        back_niou = self.total_back_niou / self.count if self.count > 0 else 0.0
        return niou, back_niou

    def reset(self):
        self.total_niou = 0.0
        self.count = 0
        self.total_back_niou = 0.0


class PRE():

    def __init__(self, nclass):
        super(PRE, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        # correct, labeled, back_correct, back_labeled = batch_pix_accuracy(preds, labels)
        correct, labeled = batch_pix_accuracy(preds, labels)
        # inter, union, back_inter, back_union = batch_intersection_union(preds, labels, self.nclass)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):

        # dsc = 1.0 * self.total_inter / (np.spacing(1) + self.total_label.cpu().numpy() - self.total_correct.cpu().numpy() + self.total_inter)
        dsc = 1.0 * self.total_inter / (np.spacing(1) + self.total_union + self.total_inter - self.total_label.cpu().numpy())
        # dsc = 1.0 * int(self.total_inter) / (np.spacing(1) + self.total_label - self.total_correct + total_inter)

        mdsc = dsc.mean()

        return mdsc
    def reset(self):

        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):

    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp

    return tp, pos, fp, neg, class_pos

def batch_pix_accuracy(output, target):

    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()



    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):

    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union

