import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import os
import argparse
import copy
import random
from PIL import Image
import numpy as np
import csv
from sklearn.preprocessing import minmax_scale
from scipy.special import softmax

import sys
import csv
import pandas as pd
from torch.distributions import Categorical
import tensorflow_probability as tfp
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def seed_everything(seed=12):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser(description='TS Training')
    parser.add_argument('--model', type=str, default='resnet34', help='net type')
    parser.add_argument('--dataset', type=str, default= 'cifar10', help='dataset')
    parser.add_argument('--normalized', type=str, default= 'cifar10', help='dataset')
    parser.add_argument('--trans_model', action='store_true', help='transformer or not')
    parser.add_argument('--num_classes', type=int, default=10, help='number classes')
    parser.add_argument('--b', type=int, default=1024, help='batch size for dataloader')
    parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--num_epoch', default=200, type=int, help='epoch number')
    parser.add_argument('--smoothing', type=float, default=0.1, help='label smoothing factor')
    parser.add_argument('--lr_densenet', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_vgg16', default=0.001, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--test_batch_size', default=2048, type=int, help='batch size')  
    
    if 'ipykernel' in sys.modules:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    return args


def ece_eval(preds, targets, n_bins=15, bg_cls = -1):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences, predictions = np.max(preds,1), np.argmax(preds,1)#confidences: pred prob; predictions: pred classes
    confidences, predictions = confidences[targets>bg_cls], predictions[targets>bg_cls]#len: 10000
    accuracies = (predictions == targets[targets>bg_cls]) 
    
    Bm, acc, conf = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    ece = 0.0
    bin_idx = 0
   
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)#boolean vector of len 100
        bin_size = np.sum(in_bin)
        Bm[bin_idx] = bin_size
        if bin_size > 0:  
            accuracy_in_bin = np.sum(accuracies[in_bin])
            acc[bin_idx] = accuracy_in_bin / Bm[bin_idx]
            confidence_in_bin = np.sum(confidences[in_bin])
            conf[bin_idx] = confidence_in_bin / Bm[bin_idx]

        bin_idx += 1
    ece = (Bm * np.abs((acc - conf))).sum()/ Bm.sum()
    ece_level = (Bm * (conf - acc)).sum()/ Bm.sum()
    return ece, acc, conf, Bm, ece_level

def optimal_T(logits, labels, upper=None, lower=None):
    best_ece = np.inf
    best_t = 0
    for T in np.arange(lower, upper, 0.01):
        logits_ = torch.tensor(logits/T)
        logits_all =F.softmax(logits_, dim=1).detach().cpu().numpy()
        ece,_,_,_,_ = ece_eval(logits_all, labels)
        if ece < best_ece:
            best_ece = ece
            best_t = T
    return np.round(best_ece,3), best_t

def classwise_ece(logits, labels):
    ece_per_class = []
    ece_level_per_class = []
    for i in range (len(logits[1])):
        ece_c, acc_c, conf_c, Bm_c, level_c = ece_eval(logits[labels==i], labels[labels==i])
        ece_per_class.append(np.round(ece_c,3))

        ece_level_per_class.append(np.round(level_c,3))

    return np.array(ece_per_class).mean(), np.array(ece_level_per_class).mean(), ece_level_per_class

def miscalibration_level(ece_level_per_class):    
    ece_level_per_class = np.array(ece_level_per_class)
    pos = ece_level_per_class[ece_level_per_class>=0]
    neg = ece_level_per_class[ece_level_per_class<0]
    # print("num of over&under confident classes:",len(pos)/len(ece_level_per_class), len(neg)/len(ece_level_per_class))

    overall_level = (pos.sum()*len(pos) + neg.sum()*len(neg))/len(ece_level_per_class)
    return np.round(overall_level,4), np.round(pos.mean(),4), np.round(neg.mean(),4), len(pos)/len(ece_level_per_class), len(neg)/len(ece_level_per_class)

def accuracy_calcuation(logits,labels,T=1):
    
    predicted = (np.array(logits)/T).argmax(1)
    total = len(labels)
    correct = np.sum(predicted==np.array(labels))
    return correct/total

def get_brier(preds, targets):
    one_hot_targets = np.zeros(preds.shape)
    one_hot_targets[np.arange(len(targets)), targets] = 1.0
    return np.mean(np.sum((preds - one_hot_targets) ** 2, axis=1))



def result(logits, labels, criterion, msc=1):
    logits_np = np.array(logits/msc)
    labels_np =np.array(labels)
    softmaxes = softmax(logits_np, axis=1)
    logits_brier = np.array(softmaxes)

    logits = torch.tensor(logits/msc)
    labels_ = torch.tensor(labels)
    nll = criterion(logits, labels_)
    accuracy = accuracy_calcuation(logits,labels)
    brier_score = get_brier(logits_brier, labels)
    logits_ =F.softmax(logits, dim=1).detach().cpu().numpy()
    ece, acc, conf, Bm, ece_diff = ece_eval(logits_, labels)
    mean_ece_per_class, mean_ece_level_per_class, ece_level_per_class = classwise_ece(logits_, labels)
    cw_mc_score, mean_over_mc, mean_under_mc, num_pos, num_neg = miscalibration_level(ece_level_per_class)
    
    return ece, np.round(accuracy,3), np.round(nll.item(),3),np.round(brier_score,3),np.round(mean_ece_per_class,3), np.round(cw_mc_score,3), np.round(mean_over_mc,3), np.round(mean_under_mc,3),ece_level_per_class,  np.round(num_pos,3), np.round(num_neg,3)
    


def normalized_mcs_t(mcs_t, args):
    mcs_t = np.array(mcs_t)

    q1 = np.percentile(mcs_t, 25, interpolation = 'midpoint')
    q3 = np.percentile(mcs_t, 75, interpolation = 'midpoint')
    IQR = q3 - q1
    upper = q3 + 1.5 * IQR
    lower = q1 - 1.5 * IQR

    mcs_t[mcs_t<lower] = lower
    mcs_t[mcs_t>upper] = upper

    if args.normalized == 'min_max':
        norm = minmax_scale(mcs_t, feature_range=(0,1))
    elif args.normalized == 'max':
        norm = mcs_t/max(mcs_t)
    elif args.normalized == 'no_norm':
        norm = mcs_t
    return norm


def tuning_ece_level_factor(opt_t, norm_ece_level, logits, labels, device):
    
    best_ece, best_ece_level_factor,best_ece_level_per_class, best_acc = np.inf, 0, opt_t, 0

    acc_ot = accuracy_calcuation(logits,labels, T = opt_t)
    for i, gemma in enumerate(np.arange(-2, 2, 0.001)):
        
        ece_level_t = opt_t * (1+ norm_ece_level * gemma) 
        
        logits_temp = torch.tensor(logits/ece_level_t)
        logits_all_temp = F.softmax(logits_temp, dim=1).numpy()
        
        ece_c, acc_c, conf_c, Bm_c, diff_c = ece_eval(logits_all_temp, labels) #acc_c is the accuracy in each bin
        acc_temp = accuracy_calcuation(logits,labels, T = ece_level_t)
       
        if ece_c < best_ece :
            
            best_acc = acc_temp
            best_ece = ece_c
            best_ece_level_factor = gemma
            best_ece_level_per_class = ece_level_t
    print('acc_ece', best_acc, best_ece)
    
    return best_ece_level_per_class, np.round(best_ece_level_factor,4)

def write_csv(filename, data):
    with open(filename, 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)

def main_result(args, result_file, criterion):

    link = 'logits/rebuttal_logits/{}_logits_{}_bs.npy'.format(str(args.dataset), str(args.model) )
    link_valid = 'logits/rebuttal_logits/{}_logits_{}_valid.npy'.format(str(args.dataset), str(args.model))

    logits, labels = np.load(link,allow_pickle=True)
    logits_valid, labels_valid = np.load(link_valid,allow_pickle=True)


    #Baseline
    ece, acc, nll, brier, cw_ece, cw_mc_score, over_mc, under_mc, ece_level_per_class,  num_pos, num_neg = result(logits, labels, criterion)
    write_csv(result_file, ['bs', str(args.dataset), str(args.model), str(args.normalized), str(acc), str(nll),str(brier),
        str(np.round(ece,4)),  str(cw_ece),
        str(cw_mc_score), str(over_mc), str(num_pos), str(under_mc), str(num_neg), 0])

    #optimal T
    ece_ot, opt_t = optimal_T(logits_valid, labels_valid, upper=2, lower=0.5)
    
    ece_ot, acc_ot, nll_ot, brier_ot, cw_ece_ot, cw_mc_score_ot, over_mc_ot, under_mc_ot, ece_level_per_class_ot,  num_pos_ot, num_neg_ot = result(logits, labels, criterion, msc=opt_t)
    write_csv(result_file, ['opt_{}'.format(str(np.round(opt_t,3))), str(args.dataset), str(args.model), str(args.normalized), 
        str(acc_ot), str(nll_ot),str(brier_ot), str(np.round(ece_ot,4)),  
        str(cw_ece_ot),str(cw_mc_score_ot), str(over_mc_ot), str(num_pos_ot), str(under_mc_ot), str(num_neg_ot), 0])

    #mcs_T
    
    _, _,  _, _, _, _, _, _, mcs_ot_v, _, _, = result(logits_valid, labels_valid, criterion, msc=opt_t)
    
    normed_mcs = normalized_mcs_t(np.array(mcs_ot_v), args)
    
    mcs_t, gemma = tuning_ece_level_factor(opt_t, normed_mcs, logits_valid, labels_valid, device)
    np.save('logits/{}_{}_mcst.npy'.format(str(args.dataset), str(args.model)), (mcs_t),allow_pickle=True, fix_imports=True)

    ece_t, acc_t, nll_t, brier_t, cw_ece_mcst, cw_mc_score_t, over_mc_t, under_mc_t, ece_level_per_class_t,  num_pos_t, num_neg_t = result(logits, labels, criterion, msc=mcs_t)
    write_csv(result_file, ['mcs_t', str(args.dataset), str(args.model), str(args.normalized),  str(acc_t), str(nll_t),str(brier_t), 
        str(np.round(ece_t,4)), str(cw_ece_mcst),str(cw_mc_score_t), 
        str(over_mc_t), str(num_pos_t), str(under_mc_t), str(num_neg_t),str(gemma)])

def main():
    seed_everything()

    result_file = "img_results_in_rebuttal_overpara.csv"
    write_csv(result_file, ['type','dataset', 'model', 'normalized', 'accuracy', 'nll','brier', 'ece', 
        'cw_ece' ,'mcs', 'over_conf', 'num_pos','under_conf', 'num_neg','gemma'])

    criterion = torch.nn.CrossEntropyLoss()

    models_all = ['resnet34', 'densenet121', 'vgg16']

    # models_in = ['efficientnet', 'vit', 'swintran', 'deit', 
    #     'cait', 'beit', 'coat', 'crossvit', 'convmixer', 'convnext']
    
    datasets_all = ['cifar10', 'cifar100', 'tinyimagenet', 'in']

    models_in_ = ['inception3', 'resnet50', 'resnext101', 'resnext50', 'vgg19']
    models_ = ['densenet169', 'densenet201']

    
    # for data in datasets_all:
    #     for model in models_all:
    #         args.model = model
    #         args.dataset = data
    #         print(args.model, args.dataset)
    #         main_result(args, result_file, criterion)

    for model in models_:
        args.model = model
        args.dataset = 'in'
        print(args.model, args.dataset)
        main_result(args, result_file, criterion)

    
if __name__ == "__main__": 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = get_args()
    main()
