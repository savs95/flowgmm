import torch
import torch.nn.functional as F
from torch import optim
from torchcontrib.optim import SWA

import collections
import numpy as np

class swaLR(optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, steps_per_epoch, num_epochs, swa_dec_pct, swa_start_pct, swa_freq_pct, swa_lr_factor, last_epoch=-1, verbose=False):
        if not isinstance(optimizer, SWA):
            raise TypeError("Expecting a SWA optimizer")
        
        lr_lambda = swa_learning_rate(num_epochs, swa_dec_pct, swa_start_pct, swa_lr_factor)
        self.swa_start = int(swa_start_pct * steps_per_epoch * num_epochs)
        self.swa_freq = int(swa_freq_pct * steps_per_epoch * num_epochs)
        print("swa_start is {}, swa_freq is {}".format(self.swa_start, self.swa_freq))
        self.steps_per_epoch = steps_per_epoch
        self.num_epochs = num_epochs
        super().__init__(optimizer=optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch, verbose=verbose)
         
    def step(self, epoch=None):
        super().step(epoch)            
        if self._step_count > self.swa_start and self._step_count % self.swa_freq == 0:
            self.optimizer.update_swa()
        
def swa_learning_rate(num_epochs, swa_dec_pct=.5, swa_start_pct=.75, swa_lr_factor=.5):
    def learning_rate_scheduler(epoch):
        t = epoch / num_epochs
        if t <= swa_dec_pct:
            factor = 1.0
        elif t <= swa_start_pct:
            factor = 1.0 - (1.0 - swa_lr_factor) * (t - swa_dec_pct) / (swa_start_pct - swa_dec_pct)
        else:
            factor = swa_lr_factor
        return factor
    return learning_rate_scheduler

def calibration_curve(outputs, labels, num_bins=20):
    """Compute calibration curve and ECE."""
    confidences = np.max(outputs, 1)
    num_inputs = confidences.shape[0]
    step = (num_inputs + num_bins - 1) // num_bins
    bins = np.sort(confidences)[::step]
    if num_inputs % step != 1:
        bins = np.concatenate((bins, [np.max(confidences)]))
    predictions = np.argmax(outputs, 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    accuracies = (predictions == labels)

    bin_confidences = []
    bin_accuracies = []
    bin_proportions = []

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) * (confidences < bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_confidences.append(avg_confidence_in_bin)
            bin_accuracies.append(accuracy_in_bin)
            bin_proportions.append(prop_in_bin)

    bin_confidences, bin_accuracies, bin_proportions = map(
          lambda lst: np.array(lst),
          (bin_confidences, bin_accuracies, bin_proportions))

    return {
      "confidence": bin_confidences,
      "accuracy": bin_accuracies,
      "proportions": bin_proportions,
      "ece": ece}

def get_ensembles_accuracy(results):
    targets = results[0][3]
    
    en_probs = np.mean(np.array(results)[:,1], axis=0)
    
    en_logits = np.mean(np.array(results)[:,0], axis=0)
    fl_probs = F.softmax(torch.from_numpy(en_logits), dim=1).numpy()

    en_preds = np.argmax(en_probs, axis=1)
    fl_pred = np.argmax(fl_probs, axis=1)

    en_acc = (en_preds == targets).mean()
    fl_acc = (fl_pred == targets).mean()
    
    return en_acc, fl_acc