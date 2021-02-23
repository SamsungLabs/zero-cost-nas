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

import pickle
import torch
import argparse

from foresight.models import *
from foresight.pruners import *
from foresight.dataset import *
from foresight.weight_initializers import init_net

def get_num_classes(args):
    return 100 if args.dataset == 'cifar100' else 10 if args.dataset == 'cifar10' else 120

def parse_arguments():
    parser = argparse.ArgumentParser(description='Zero-cost Metrics for NAS-Bench-201')
    parser.add_argument('--api_loc', default='data/NAS-Bench-201-v1_0-e61699.pth',
                        type=str, help='path to API')
    parser.add_argument('--outdir', default='./',
                        type=str, help='output directory')
    parser.add_argument('--init_w_type', type=str, default='none', help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='none', help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--num_data_workers', type=int, default=2, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
    parser.add_argument('--dataload_info', type=int, default=1, help='number of batches to use for random dataload or number of samples per class for grasp dataload')
    parser.add_argument('--seed', type=int, default=42, help='pytorch manual seed')
    parser.add_argument('--write_freq', type=int, default=1, help='frequency of write to file')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=0, help='end index')
    parser.add_argument('--noacc', default=False, action='store_true', help='avoid loading NASBench2 api an instead load a pickle file with tuple (index, arch_str)')
    args = parser.parse_args()
    args.device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    return args

if __name__ == '__main__':
    args = parse_arguments()
    
    if args.noacc:
        api = pickle.load(open(args.api_loc,'rb'))
    else:
        from nas_201_api import NASBench201API as API
        api = API(args.api_loc)
    
    torch.manual_seed(args.seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    train_loader, val_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset, args.num_data_workers)

    cached_res = []
    pre='cf' if 'cifar' in args.dataset else 'im'
    pfn=f'nb2_{pre}{get_num_classes(args)}_seed{args.seed}_dl{args.dataload}_dlinfo{args.dataload_info}_initw{args.init_w_type}_initb{args.init_b_type}.p'
    op = os.path.join(args.outdir,pfn)

    
    args.end = len(api) if args.end == 0 else args.end

    #loop over nasbench2 archs
    for i, arch_str in enumerate(api):

        if i < args.start:
            continue
        if i >= args.end:
            break 

        res = {'i':i, 'arch':arch_str}

        net = nasbench2.get_model_from_arch_str(arch_str, get_num_classes(args))
        net.to(args.device)

        init_net(net, args.init_w_type, args.init_b_type)
        
        arch_str2 = nasbench2.get_arch_str_from_model(net)
        if arch_str != arch_str2:
            print(arch_str)
            print(arch_str2)
            raise ValueError

        measures = predictive.find_measures(net, 
                                            train_loader, 
                                            (args.dataload, args.dataload_info, get_num_classes(args)),
                                            args.device)

        res['logmeasures']= measures

        if not args.noacc:
            info = api.get_more_info(i, 'cifar10-valid' if args.dataset=='cifar10' else args.dataset, iepoch=None, hp='200', is_random=False)

            trainacc = info['train-accuracy']
            valacc   = info['valid-accuracy']
            testacc  = info['test-accuracy']
        
            res['trainacc']=trainacc
            res['valacc']=valacc
            res['testacc']=testacc
        
        #print(res)
        cached_res.append(res)

        #write to file
        if i % args.write_freq == 0 or i == len(api)-1 or i == 10:
            print(f'writing {len(cached_res)} results to {op}')
            pf=open(op, 'ab')
            for cr in cached_res:
                pickle.dump(cr, pf)
            pf.close()
            cached_res = []

