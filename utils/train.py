# *----------------------------------------------------------------------------*
# * Copyright (C) 2023 Politecnico di Torino, Italy                            *
# * SPDX-License-Identifier: Apache-2.0                                        *
# *                                                                            *
# * Licensed under the Apache License, Version 2.0 (the "License");            *
# * you may not use this file except in compliance with the License.           *
# * You may obtain a copy of the License at                                    *
# *                                                                            *
# * http://www.apache.org/licenses/LICENSE-2.0                                 *
# *                                                                            *
# * Unless required by applicable law or agreed to in writing, software        *
# * distributed under the License is distributed on an "AS IS" BASIS,          *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
# * See the License for the specific language governing permissions and        *
# * limitations under the License.                                             *
# *                                                                            *
# * Author:  Matteo Risso <matteo.risso@polito.it>                             *
# *          Beatrice Alessandra Motetti <beatrice.motetti@polito.it>          *
# *----------------------------------------------------------------------------*
import os
import pathlib
import numpy as np
import random
import torch

# seeding everything to maximize reproducibility
def set_seed(seed=23):
    #os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.use_deterministic_algorithms(True, warn_only=True)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False


# tries to load a model from a checkpoint directory (if existing)
def try_load_checkpoint(model, checkpoint_dir, device):
    if os.path.exists(checkpoint_dir / 'best.ckp'):
        saved_info = torch.load(
            checkpoint_dir / 'best.ckp', map_location='cpu')
        model.load_state_dict(saved_info['model_state_dict'])
        model = model.to(device)
        return True
    else:
        return False


class CheckPoint():
    """
    save/load a checkpoint based on a metric
    """

    def __init__(self, dir, net, optimizer,
                 mode='min', fmt='ck_{epoch:03d}.pt', save_best_only=False):
        if mode not in ['min', 'max']:
            raise ValueError("Early-stopping mode not supported")
        self.dir = pathlib.Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.format = fmt
        self.save_best_only = save_best_only
        self.net = net
        self.optimizer = optimizer
        self.val = None
        self.epoch = None
        self.best_path = None

    def __call__(self, epoch, val):
        val = float(val)
        if self.val is None:
            self.update_and_save(epoch, val)
        elif self.mode == 'min' and val < self.val:
            self.update_and_save(epoch, val)
        elif self.mode == 'max' and val > self.val:
            self.update_and_save(epoch, val)

    def update_and_save(self, epoch, val):
        self.epoch = epoch
        self.val = val
        self.update_best_path()
        self.save()

    def update_best_path(self):
        if not self.save_best_only:
            self.best_path = self.dir / self.format.format(**self.__dict__)
        else:
            self.best_path = self.dir / 'best.pt'

    def save(self, path=None):
        if path is None:
            path = self.best_path
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val': self.val,
        }, path)

    def load_best(self):
        if self.best_path is None:
            raise FileNotFoundError("Best path not set!")
        self.load(self.best_path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class EarlyStopping():
    """
    stop the training when the loss does not improve.
    """

    def __init__(self, patience=20, mode='min'):
        if mode not in ['min', 'max']:
            raise ValueError("Early-stopping mode not supported")
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_val = None

    def __call__(self, val):
        val = float(val)
        if self.best_val is None:
            self.best_val = val
        elif self.mode == 'min' and val < self.best_val:
            self.best_val = val
            self.counter = 0
        elif self.mode == 'max' and val > self.best_val:
            self.best_val = val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early Stopping!")
                return True
        return False
