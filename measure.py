import numpy as np


class SegmentationMetric_foreground(object):
    def __init__(self, numClass):
        self.numClass = numClass
        #self.confusionMatrix = np.zeros((self.numClass,) * 2)
        self.confusionMatrix_foreground = np.zeros((self.numClass - 1,) * 2)
        
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix_foreground).sum() /( self.confusionMatrix_foreground.sum()+ 1e-8)
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix_foreground) / (self.confusionMatrix_foreground.sum(axis=1) + 1e-8)
        return classAcc

    def kappa(self):
        """计算kappa值系数"""
        pe_rows = np.sum(self.confusionMatrix_foreground, axis=0)
        pe_cols = np.sum(self.confusionMatrix_foreground, axis=1)
        sum_total = sum(pe_cols)
        pe = np.dot(pe_rows, pe_cols) / (float(sum_total ** 2)+ 1e-8)
        po = np.trace(self.confusionMatrix_foreground) / (float(sum_total)+ 1e-8)
        return (po - pe) / (1 - pe)

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def classRecallAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it recall)
        # acc = (TP) / TP + FN
        recall = np.diag(self.confusionMatrix_foreground) / (self.confusionMatrix_foreground.sum(axis=0) + 1e-8)
        return recall

    def F1Score(self):
        F1Score = 2 * self.classPixelAccuracy() * self.classRecallAccuracy() /\
               (self.classPixelAccuracy() + self.classRecallAccuracy() + 1e-8)
        return np.average(F1Score)

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix_foreground)
        union = np.sum(self.confusionMatrix_foreground, axis=1) + np.sum(self.confusionMatrix_foreground, axis=0) - np.diag(
            self.confusionMatrix_foreground)
        IoU = intersection / (union)
        mIoU = np.nanmean(IoU)
        #cIoU = np.nanmean(IoU, axis = 1)
        return mIoU

    def genConfusionMatrix_foreground(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass-1)
        label = (self.numClass-1) * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=(self.numClass-1) ** 2)
        confusionMatrix_foreground = count.reshape(self.numClass-1, self.numClass-1)
        return confusionMatrix_foreground

    def addBatch(self, imgPredict, imgLabel):
        imgPredict = imgPredict.cpu()
        imgLabel = imgLabel.cpu()
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix_foreground += self.genConfusionMatrix_foreground(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix_foreground = np.zeros((self.numClass-1, self.numClass-1))
