import torch
import torch.nn.functional as F

import numpy as np
import json

class Ensembles:
    def __init__(self, trainers):
        self.num_models = len(trainers)
        self.trainers = trainers
        self.results = self.__init_results()
    def __init_results(self):
        results = {}
        results["Number of models"]=self.num_models
        results["Epochs"]=[]
        for index in range(self.num_models):
            results["Model {}".format(index+1)]=[]
        results["Deep Ensembles"]=[]
        results["Flow Ensembles"]=[]
        return results
    def update_results(self, epoch):
        outputs=[]
        self.results["Epochs"].append(epoch)
        for index, trainer in enumerate(self.trainers):
            logits, probs, preds, targets, acc = trainer.getTestOutputs()
            self.results["Model {}".format(index+1)].append(acc)
            outputs.append([logits, probs, preds, targets, acc])
        
        en_acc, fl_acc = self.__get_ensembles_accuracy(outputs)
        self.results["Deep Ensembles"].append(en_acc)
        self.results["Flow Ensembles"].append(fl_acc)
    def __get_ensembles_accuracy(self, outputs):
        targets = outputs[0][3]

        en_probs = np.mean(np.array(outputs)[:,1], axis=0)

        en_logits = np.mean(np.array(outputs)[:,0], axis=0)
        fl_probs = F.softmax(torch.from_numpy(en_logits), dim=1).numpy()

        en_preds = np.argmax(en_probs, axis=1)
        fl_pred = np.argmax(fl_probs, axis=1)

        en_acc = (en_preds == targets).mean()
        fl_acc = (fl_pred == targets).mean()

        return en_acc, fl_acc
    
    def write_results(filename):
        with open(filename, 'w') as outfile:
            json.dump(self.results, outfile)
            
    @staticmethod
    def load_results(filename):
        with open(filename) as json_file:
            results = json.load(json_file)
        return results
    
    @staticmethod
    def plot_results(results, ax, title, start=0):
        num_models = results["Number of models"]
        epochs_list = results["Epochs"]
        for index in range(num_models):
            accs = results["Model {}".format(index+1)]
            ax.plot(epochs_list[start:], accs[start:], lw=1, label="Model {}".format(index+1), color='C{}'.format(index)) 

        ens_probs_accs = results["Deep Ensembles"]
        ax.plot(epochs_list[start:], ens_probs_accs[start:], "-v", lw=2, markersize=8, label="Deep Ensembles", color='C6')

        ens_probs_logits = results["Flow Ensembles"]
        ax.plot(epochs_list[start:], ens_probs_logits[start:], "-*", lw=3, markersize=12, label="Flow Ensembles", color='C3')

        ax.set(xlabel='epoch', ylabel='accuracy', title=title)
        ax.grid()