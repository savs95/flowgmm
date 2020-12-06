from ensembles import Ensembles
from flowgmm_trainer import make_trainer

from flow_ssl.data.nlp_datasets import AG_News,YAHOO

import argparse
import json
import os
import warnings
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Test accuracies between Deep Ensembles and Flow Ensembles')
    
    parser.add_argument("--dataset", help="Text dataset (YAHOO or AG_News)",
                        choices=["AG_News", "YAHOO"], default="AG_News")
    
    parser.add_argument("--labeled", help="Number of labeled data",
                        default=200, type=int)
    
    parser.add_argument("--net_config", help="Flow configuration",
                        default={'k':1024,'coupling_layers':7,'nperlayer':1})
    
    parser.add_argument("--num_models", help="Number of models in ensembles",
                        default=3, type=int)
    
    parser.add_argument("--num_epochs", help="Number of training epochs",
                        default=100, type=int)
    
    parser.add_argument("--test_epochs", help="Number of training epochs per test phase",
                        default=5, type=int)
    
    parser.add_argument("--lr", help="Learning rate",
                        default=3e-4, type=float)
    
    parser.add_argument("--unlab_weight", help="Unlabeled data weight",
                        default=.6, type=float)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    dataset = AG_News if args.dataset=="AG_News" else YAHOO
    labeled = args.labeled
    net_config = args.net_config
    num_models = args.num_models
    num_epochs = args.num_epochs
    test_epochs = args.test_epochs
    lr = args.lr
    unlab_weight = args.unlab_weight
    
    train_data, test_data = dataset(), dataset(train=False) 
    trainers=[
        make_trainer(
            train_data=train_data,
            test_data=test_data,
            split={'train':labeled,'val':5000},
            net_config=net_config,
            num_epochs=num_epochs,
            lr=lr,
            trainer_config={'unlab_weight':unlab_weight}
        ) for i in range(num_models)
    ]
    ensembles = Ensembles(trainers)
    epoch = 0
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        while epoch < num_epochs:
            
            for trainer in trainers:
                trainer.train(test_epochs)
                
            epoch += test_epochs
            ensembles.update_results(epoch)
            
    print(ensembles.results)
            
    os.makedirs('results', exist_ok = True)
    filename = "{}-ensembles-{:03}.json".format(args.dataset, random.randrange(1, 10**3))
    with open(os.path.join('results',filename), 'w') as outfile:
        json.dump(ensembles.results, outfile)