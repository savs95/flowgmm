import torch, torchvision
import torch.nn.functional as F
from torch.optim import SGD,Adam,AdamW
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA
from torch import optim

from oil.datasetup.datasets import CIFAR10, split_dataset
from oil.utils.utils import Eval, LoaderTo, cosLr, dmap, FixedNumpySeed, FixedPytorchSeed

from flow_ssl.data.nlp_datasets import AG_News,YAHOO

from utils.utils import swaLR, calibration_curve, get_ensembles_accuracy
from models import RealNVPTabularWPrior, SemiFlow
from ensembles import Ensembles

import numpy as np
import os
import pandas as pd
from functools import partial

import warnings

def make_trainer(train_data, test_data, bs=5000, split={'train':200,'val':5000},
                network=RealNVPTabularWPrior, net_config={}, num_epochs=15,
                optim=AdamW, lr=1e-3, opt_config={'weight_decay':1e-5},
                swa=False, swa_config={'swa_dec_pct':.5, 'swa_start_pct':.75, 'swa_freq_pct':.05, 'swa_lr_factor':.1},
                trainer=SemiFlow,
                trainer_config={'log_dir':os.path.expanduser('~/tb-experiments/UCI/'),'log_args':{'minPeriod':.1, 'timeFrac':3/10}},
                dev='cuda', save=False):
    
    datasets = split_dataset(train_data,splits=split)
    datasets['_unlab'] = dmap(lambda mb: mb[0],train_data)
    datasets['test'] = test_data
        
    device = torch.device(dev)
    
    dataloaders = {k : LoaderTo(DataLoader(v,
                                         batch_size=min(bs,len(datasets[k])),
                                         shuffle=(k=='train'),
                                         num_workers=0,
                                         pin_memory=False),
                              device) 
                   for k, v in datasets.items()}
    dataloaders['Train'] = dataloaders['train']
    
    model = network(num_classes=train_data.num_classes, dim_in=train_data.dim, **net_config).to(device)
    
    opt_constr = partial(optim, lr=lr, **opt_config)
    
    if swa == True:
        lr_sched = partial(swaLR, num_epochs=num_epochs, **swa_config)
    else:
        lr_sched = cosLr(num_epochs)
    #     lr_sched = lambda e:1
    return trainer(model,dataloaders, swa, opt_constr,lr_sched, **trainer_config)