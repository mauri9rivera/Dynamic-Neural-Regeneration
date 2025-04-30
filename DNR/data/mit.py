import numpy as np
import torch
from DNR.data.datasets import load_dataset, load_dataset_linear_eval


class MIT67:
    def __init__(self, cfg):

        if cfg.cs_kd:
            sampler_type = 'pair'
        else:
            sampler_type = 'default'
            
        if cfg.use_val:
            trainloader, valloader, tst_loader = load_dataset('MIT67_v2', 
                                                  cfg.data, 
                                                  sampler_type, batch_size=cfg.batch_size)
            self.num_classes = trainloader.dataset.num_classes

            self.train_loader = trainloader
            self.tst_loader = tst_loader
            self.val_loader = valloader    
        elif cfg.use_train_val:
            trainloader, valloader, tst_loader = load_dataset('MIT67_v3', 
                                                  cfg.data, 
                                                  sampler_type, batch_size=cfg.batch_size)
            self.num_classes = trainloader.dataset.num_classes

            self.train_loader = trainloader
            self.tst_loader = tst_loader
            self.val_loader = valloader                
        else:
            if cfg.eval_linear:
                trainloader, valloader = load_dataset_linear_eval('MIT67',
                                                      cfg.data,
                                                      sampler_type, batch_size=cfg.linear_batch_size)
            else:
                trainloader, valloader = load_dataset('MIT67',
                                                      cfg.data,
                                                      sampler_type, batch_size=cfg.batch_size)

            self.num_classes = trainloader.dataset.num_classes

            self.train_loader = trainloader
            self.tst_loader = valloader
            self.val_loader = valloader