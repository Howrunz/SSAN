import os
import time
import subprocess
import importlib
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from dataset import CROHME_DATASET, ALPHABET
from utils import load_config, Meter, cal_score, compute_edit_distance
opj = os.path.join

def eval_model(model, eval_loader):
    model.eval()
    loss_meter = Meter()
    word_right, exp_right, length, cal_num = 0, 0, 0, 0
    e1, e2, e3 = 0, 0, 0
    pred = []

    with torch.no_grad():
        if args.single: bar = tqdm(total=len(eval_loader), desc=f'Evaluating {args.model}', unit='imgs')
        for batch_idx, sampled_batch in enumerate(eval_loader):
            name = sampled_batch['name']
            batch, time = sampled_batch['labels'].shape[:2]
            sampled_batch = {k: sampled_batch[k].to(params['device']) for k in dict(list(sampled_batch.items())[:-1])}
            sampled_batch.update({'is_train': False})
            probs, loss_list, alphas = model(**sampled_batch)
            loss = sum(loss_list)
            loss_meter.add(loss.item())

            wordRate, ExpRate = cal_score(probs, sampled_batch['labels'], sampled_batch['labels_mask'])
            prediction = eval_loader.dataset.alphabet.decode(probs.max(2)[1][0]).replace(" eos", "")
            ground_true = eval_loader.dataset.alphabet.decode(sampled_batch['labels'][0]).replace(" eos", "")
            distance = compute_edit_distance(prediction, ground_true)

            if ExpRate == 1:
                assert prediction == ground_true

            if ExpRate == 0:
                pred.append((probs, sampled_batch['labels'], name[0]))
            word_right = word_right + wordRate * time
            exp_right = exp_right + ExpRate * batch
            length = length + time
            cal_num = cal_num + batch
            if distance <= 1:
                e1 += 1
            if distance <= 2:
                e2 += 1
            if distance <= 3:
                e3 += 1
            if bar:
                bar.set_postfix({
                    "exprate": exp_right / cal_num
                })
                bar.update(1)

        return loss_meter.mean, word_right / length, exp_right / cal_num, pred, [e1 / cal_num, e2 / cal_num, e3 / cal_num]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epo', default=0, type=int)
    parser.add_argument('--expname', default='', type=str)
    parser.add_argument('--gpu', default='', type=str)
    parser.add_argument('--single', action='store_true')
    parser.add_argument('--dataset', default='CROHME', type=str, choices=['CROHME', 'HME100K'])
    parser.add_argument('--model', default='SSAN_CAN_DWAP', type=str, choices=['SSAN_DWAP', 'SSAN_CAN_DWAP', 'SSAN_SAM_DWAP'])
    parser.add_argument('--cp', default='', type=str)
    args = parser.parse_args()

    config_path = "config.yaml" if args.single else opj("results", args.expname, f"{args.expname}.yaml")
    params = load_config(config_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = params['Test_GPU'] if args.gpu == '' else args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device
    params['local_rank'] = device

    params['dataset_name'] = args.dataset if args.single else params['dataset_name']
    params['model'] = args.model if args.single else params['model']
    checkpoint_path = args.cp if args.single else opj(params['checkpoint_dir'], args.expname, "temp_model.pth")

    alphabet = ALPHABET(params['alphabet_path'])
    eval_loaders = []
    if params['dataset_name'] == 'CROHME':
        keywords = ['2014', '2016', '2019']
        for year in keywords:
            eval_dataset = CROHME_DATASET(params, params['eval_image_path'].replace('2014', year), alphabet, istrain=False)
            eval_loader = DataLoader(eval_dataset, batch_size=1,
                                     num_workers=params['workers'],
                                     collate_fn=eval_dataset.collate_fn, pin_memory=True)
            eval_loaders.append(eval_loader)
    elif params['dataset_name'] == 'HME100K':
        keywords = ['HME100K']
        eval_dataset = CROHME_DATASET(params, params['eval_image_path'], alphabet, istrain=False)
        eval_loader = DataLoader(eval_dataset, batch_size=1,
                                 num_workers=params['workers'],
                                 collate_fn=eval_dataset.collate_fn, pin_memory=True)
        eval_loaders.append(eval_loader)

    params['alphabet_num'] = len(alphabet)

    model_name = params['model']
    module = importlib.import_module(f"models.{model_name}")
    model = getattr(module, model_name)(params).to(device)

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state['model'])

    for idx, eval_loader in enumerate(eval_loaders):
        eval_loss, eval_word_score, eval_exprate, pred, error = eval_model(model, eval_loader)
        if args.single:
            print(f'{args.model} Evaluation Results: \nloss: {eval_loss:.4f} word score: {eval_word_score:.4f} \n\tExpRate: {eval_exprate:.4f}\n\t1Error: {error[0]:.4f} \n\t2Error: {error[1]:.4f} \n\t3Error: {error[2]}')
        else:
            with open(opj(params['checkpoint_dir'], args.expname, f"exprate_{keywords[idx]}.best"), "r") as bestrate:
                best_exprate = float(bestrate.read())
            if eval_exprate > best_exprate:
                with open(opj(params['checkpoint_dir'], args.expname, f"exprate_{keywords[idx]}.best"), "w") as bestrate:
                    bestrate.write(str(eval_exprate))
            with open(opj(params['checkpoint_dir'], args.expname, "eval.log"), "a", encoding='utf-8') as eval_log_file:
                eval_log_file.write(f'Epoch: {args.epo+1} loss: {eval_loss:.4f} word score: {eval_word_score:.4f} ExpRate: {eval_exprate:.4f}\n')
                eval_log_file.write(f'Epoch: {args.epo+1} 1Error: {error[0]:.4f} 2Error: {error[1]:.4f} 3Error: {error[2]}\n')

            if eval_exprate > best_exprate:
                filename = f'{os.path.join(params["checkpoint_dir"], args.expname)}/{keywords[idx]}_ExpRate-{eval_exprate:.4f}_{args.epo}.pth'
                torch.save({'model': model.state_dict()}, filename)
                torch.save({'model': model.state_dict()}, f'{os.path.join(params["checkpoint_dir"], args.expname)}/best_model.pth')
