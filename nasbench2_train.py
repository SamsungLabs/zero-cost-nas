# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import argparse
import pickle
import time
import torch
import torch.optim as optim
import torch.nn.functional as F


from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers import ProgressBar

from foresight.models import *
from foresight.pruners import *
from foresight.dataset import *

def get_num_classes(args):
    return 100 if args.dataset == 'cifar100' else 10 if args.dataset == 'cifar10' else 120

def setup_experiment(net, args):

    optimiser = optim.SGD(
        net.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=0, last_epoch=-1)

    train_loader, val_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset, args.num_data_workers, resize=args.img_size)

    return optimiser, lr_scheduler, train_loader, val_loader

def parse_arguments():
    parser = argparse.ArgumentParser(description='EcoNAS Training Pipeline for NAS-Bench-201')
    parser.add_argument('--api_loc', default='data/NAS-Bench-201-v1_0-e61699.pth',
                        type=str, help='path to API')
    parser.add_argument('--outdir', default='./',
                        type=str, help='output directory')
    parser.add_argument('--outfname', default='test',
                        type=str, help='output filename')
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--init_channels', default=4, type=int)
    parser.add_argument('--img_size', default=8, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--num_data_workers', type=int, default=2, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
    parser.add_argument('--dataload_info', type=int, default=1, help='number of batches to use for random dataload or number of samples per class for grasp dataload')
    parser.add_argument('--start', type=int, default=5, help='start index')
    parser.add_argument('--end', type=int, default=10, help='end index')
    parser.add_argument('--write_freq', type=int, default=5, help='frequency of write to file')
    parser.add_argument('--logmeasures', action="store_true", default=False, help='add extra logging for predictive measures')
    args = parser.parse_args()
    args.device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    return args


def train_nb2():
    args = parse_arguments()
    archs = pickle.load(open(args.api_loc,'rb'))
    
    pre='cf' if 'cifar' in args.dataset else 'im'
    if args.outfname == 'test':
        fn = f'nb2_train_{pre}{get_num_classes(args)}_r{args.img_size}_c{args.init_channels}_e{args.epochs}.p'
    else:
        fn = f'{args.outfname}.p'
    op = os.path.join(args.outdir,fn)

    print('outfile =',op)
    cached_res = []

    #loop over nasbench2 archs
    for i, arch_str in enumerate(archs):

        if i < args.start:
            continue
        if i >= args.end:
            break 

        res = {'idx':i, 'arch_str':arch_str, 'logmeasures':[]}

        net = nasbench2.get_model_from_arch_str(arch_str, get_num_classes(args), init_channels=args.init_channels)
        net.to(args.device)

        optimiser, lr_scheduler, train_loader, val_loader = setup_experiment(net, args)
        
        #start training
        criterion = F.cross_entropy
        trainer = create_supervised_trainer(net, optimiser, criterion, args.device)
        evaluator = create_supervised_evaluator(net, {
            'accuracy': Accuracy(),
            'loss': Loss(criterion)
        }, args.device)

        pbar = ProgressBar()
        pbar.attach(trainer)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_epoch(engine):
                
            #change LR
            lr_scheduler.step()

            #run evaluator
            evaluator.run(val_loader)

            #metrics
            metrics = evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['loss']

            pbar.log_message(f"Validation Results - Epoch: {engine.state.epoch} Avg accuracy: {round(avg_accuracy*100,2)}% Val loss: {round(avg_loss,2)} Train loss: {round(engine.state.output,2)}")

            measures = {}
            if args.logmeasures:
                measures = predictive.find_measures(net, 
                                    train_loader, 
                                    (args.dataload, args.dataload_info, get_num_classes(args)),
                                    args.device)
            measures['train_loss'] = engine.state.output
            measures['val_loss'] = avg_loss
            measures['val_acc'] = avg_accuracy
            measures['epoch'] = engine.state.epoch
            res['logmeasures'].append(measures)

        #at epoch zero
        #run evaluator
        evaluator.run(val_loader)

        #metrics
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        measures = {}
        if args.logmeasures:
            measures = predictive.find_measures(net, 
                                train_loader, 
                                (args.dataload, args.dataload_info, get_num_classes(args)),
                                args.device)
        measures['train_loss'] = 0
        measures['val_loss'] = avg_loss
        measures['val_acc'] = avg_accuracy
        measures['epoch'] = 0
        res['logmeasures'].append(measures)


        #run training
        stime = time.time()
        trainer.run(train_loader, args.epochs)
        etime = time.time()

        res['time'] = etime-stime

        #print(res)
        cached_res.append(res)

        #write to file
        if i % args.write_freq == 0 or i == args.end-1 or i == args.start + 10:
            print(f'writing {len(cached_res)} results to {op}')
            pf=open(op, 'ab')
            for cr in cached_res:
                pickle.dump(cr, pf)
            pf.close()
            cached_res = []

if __name__ == '__main__':
    train_nb2()