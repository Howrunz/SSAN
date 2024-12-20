import os
import cv2
import sys
import importlib
import argparse
import statistics
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from difflib import SequenceMatcher
import time

from dataset import CROHME_DATASET
from dataset import ALPHABET

from utils import load_config, load_checkpoint, compute_edit_distance

opj = os.path.join


parser = argparse.ArgumentParser(description='model testing')
parser.add_argument('--dataset', default='CROHME', type=str, help='数据集名称')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--expname', default='')
args = parser.parse_args()


def cal_score(word_probs, word_label, mask):
    line_right = 0
    if word_probs is not None:
        _, word_pred = word_probs.max(2)
    word_scores = [SequenceMatcher(None, s1[:int(np.sum(s3))], s2[:int(np.sum(s3))], autojunk=False).ratio() * (
                len(s1[:int(np.sum(s3))]) + len(s2[:int(np.sum(s3))])) / len(s1[:int(np.sum(s3))]) / 2
                   for s1, s2, s3 in zip(word_label.cpu().detach().numpy(), word_pred.cpu().detach().numpy(),
                                         mask.cpu().detach().numpy())]

    batch_size = len(word_scores)
    for i in range(batch_size):
        if word_scores[i] == 1:
            line_right += 1

    ExpRate = line_right / batch_size
    word_scores = np.mean(word_scores)
    return word_scores, ExpRate

if not args.dataset:
    print('请提供数据集名称')
    exit(-1)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

config_root = './'
if args.dataset == 'CROHME':
    config_file = opj('results', args.expname, f'{args.expname}.yaml')
    word_path = './data/crohme_alphabet.txt'
    data_path = {
        'train': '/data02/zhr/data/2_MER/CROHME/npy_file_gau/Train_crohme_dict_height128_gau.npy',
    }
else:
    config_file = opj('results', args.expname, f'{args.expname}.yaml')
    word_path = './data/hme100k_alphabet_new.txt'
    data_path = {
        'hme100k': '/data02/zhr/data/2_MER/HME100K/pickle_file_old/Train_full_ss_dict_h128_gau.npy'
    }

"""加载config文件"""
params = load_config(config_file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params['device'] = device
params['local_rank'] = device
words = ALPHABET(word_path)
params['alphabet_num'] = len(words)
fp = opj('results', args.expname, 'best_model.pth')

model_name = params['model']
module = importlib.import_module(f"models.{model_name}")
model = getattr(module, model_name)(params).to(device)

data_loader = {}
for key in data_path.keys():
    test_dataset = CROHME_DATASET(params, data_path[key], words, res=False, istrain=False)
    loader = DataLoader(test_dataset, batch_size=1, collate_fn=test_dataset.collate_fn, pin_memory=True)
    data_loader[key] = loader

ignore = ['sos', 'eos']

load_checkpoint(fp, model, multi_gpu=False)
logging.info(f'Now Testing {fp}')
model.eval()
total = 0

with torch.no_grad():
    for key in data_loader.keys():
        line_right = 0
        E_rate = 0
        e1, e2, e3 = 0, 0, 0
        model_time = 0
        real_sum = 0
        number_ori = {}
        alpha_matrix_allsample = {}

        for i, sampled_batch in enumerate(tqdm(data_loader[key])):
            name = sampled_batch['name']
            sampled_batch = {k: sampled_batch[k].to(device) for k in dict(list(sampled_batch.items())[:-1])}
            sampled_batch.update({'is_train': False})
            a = time.time()

            probs, loss_list, alphas = model(**sampled_batch)
            model_time += (time.time() - a)

            prediction = words.decode(probs.max(2)[1][0]).replace(' eos', '')
            ground_true = words.decode(sampled_batch['labels'][0]).replace(' eos', '')
            if prediction == ground_true:
                total += 1

            pred_list = prediction.split()

            alpha_matrix = []
            for alphanum in range(len(pred_list)):
                if pred_list[alphanum] in ignore:
                    continue
                else:
                    map = (alphas[0, alphanum] > 2e-2).type(torch.float32)
                    alpha_matrix.append(map.cpu().numpy())
            alpha_matrix_allsample[name[0]] = alpha_matrix
        np.save(f'./alignment_cp/{args.expname}.npy', alpha_matrix_allsample)
        print(total / len(data_loader[key]))
