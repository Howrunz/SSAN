import os
import subprocess
import sys
import time
import shutil
import random
import logging
import argparse
import importlib
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from tensorboardX import SummaryWriter

from dataset import get_crohme_dataset
from utils import load_config, update_lr, Meter, cal_score, compute_edit_distance, load_checkpoint, save_checkpoint
opj = os.path.join


def train_model(params, model, optimizer, epoch, curr_now, train_loader, writer=None, logging=None):
    model.train()
    train_loader.sampler.set_epoch(epoch)
    loss_meter = Meter()
    word_right, exp_right, length, cal_num = 0, 0, 0, 0

    bar = tqdm(total=len(train_loader)//params['train_part'], desc=f'Training Epoch {epoch}', unit='steps') if params['rank']==0 else None
    for batch_idx, sampled_batch in enumerate(train_loader):
        name = sampled_batch['name']
        batch, time_steps = sampled_batch['labels'].shape[:2]
        sampled_batch = {k: sampled_batch[k].cuda() for k in dict(list(sampled_batch.items())[:-1])}
        if not 'lr_decay' in params or params['lr_decay'] == 'cosine':
            update_lr(optimizer, epoch, batch_idx, len(train_loader), params['epochs'], params['lr'])
        optimizer.zero_grad()
        probs, loss_list, alphas = model(**sampled_batch)
        loss = sum(loss_list)
        loss.backward()

        if params['gradient_clip']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient'])
        optimizer.step()
        loss_meter.add(loss.item())

        wordRate, ExpRate = cal_score(probs, sampled_batch['labels'], sampled_batch['labels_mask'])
        word_right = word_right + wordRate * time_steps
        exp_right = exp_right + ExpRate * batch
        length = length + time_steps
        cal_num = cal_num + batch

        if writer:
            current_step = params['epochs'] * len(train_loader) * curr_now + epoch * len(train_loader) + batch_idx + 1
            for num, loss_item in enumerate(loss_list):
                writer.add_scalar(f'train/loss_{params["loss"]["name"][num]}', loss_item.item(), current_step)
            writer.add_scalar('train/loss_total', loss.item(), current_step)
            writer.add_scalar('train/WordRate', wordRate, current_step)
            writer.add_scalar('train/ExpRate', ExpRate, current_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], current_step)

        if bar:
            bar.set_postfix({
                'loss': loss.item() / batch,
                "exprate": exp_right / cal_num
            })
            bar.update(1)

        if batch_idx >= len(train_loader)//params['train_part']:
            break

    if writer:
        writer.add_scalar('epoch/train_loss', loss_meter.mean, params['epochs'] * curr_now + epoch+1)
        writer.add_scalar('epoch/train_WordRate', word_right / length, params['epochs'] * curr_now + epoch+1)
        writer.add_scalar('epoch/train_ExpRate', exp_right / cal_num, params['epochs'] * curr_now + epoch + 1)

    if logging:
        logging.info(
            f'{epoch + 1} loss:{loss_meter.mean:.4f} WRate:{word_right / length:.4f} Train ERate:{exp_right / cal_num:.4f}')
    return loss_meter.mean, word_right / length, exp_right / cal_num


def eval_model(params, model, epoch, curr_now, eval_loader, writer=None):
    model.eval()
    loss_meter = Meter()
    word_right, exp_right, length, cal_num = 0, 0, 0, 0
    e1, e2, e3 = 0, 0, 0
    pred = []

    bar = tqdm(total=len(eval_loader), desc=f'Evaluating', unit='steps') if params['rank'] == 0 else None
    with torch.no_grad():
        for batch_idx, sampled_batch in enumerate(eval_loader):
            name = sampled_batch['name']
            batch, time_steps = sampled_batch['labels'].shape[:2]
            sampled_batch = {k: sampled_batch[k].cuda() for k in dict(list(sampled_batch.items())[:-1])}
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
            word_right = word_right + wordRate * time_steps
            exp_right = exp_right + ExpRate * batch
            length = length + time_steps
            cal_num = cal_num + batch
            if distance <= 1:
                e1 += 1
            if distance <= 2:
                e2 += 1
            if distance <= 3:
                e3 += 1

            if writer:
                current_step = params['epochs'] * len(eval_loader) * curr_now + epoch * len(eval_loader) + batch_idx + 1
                for num, loss_item in enumerate(loss_list):
                    writer.add_scalar(f'eval/loss_{params["loss"]["name"][num]}', loss_item.item(), current_step)
                writer.add_scalar('train/loss_total', loss.item(), current_step)
                writer.add_scalar('eval/WordRate', wordRate, current_step)
                writer.add_scalar('eval/ExpRate', ExpRate, current_step)

            if bar: bar.update(1)

        if writer:
            writer.add_scalar('epoch/eval_loss', loss_meter.mean, params['epochs'] * curr_now + epoch + 1)
            writer.add_scalar('epoch/eval_WordRate', word_right / length, params['epochs'] * curr_now + epoch + 1)
            writer.add_scalar('epoch/eval_ExpRate', exp_right / len(eval_loader.dataset), params['epochs'] * curr_now + epoch + 1)

        dist.barrier()
        return loss_meter.mean, word_right / length, exp_right / cal_num, pred, [e1 / cal_num, e2 / cal_num, e3 / cal_num]


def main(args):
    now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    print(now)
    params = load_config(args.cfg)
    os.environ['CUDA_VISIBLE_DEVICES'] = params["GPU"]
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])
    model_name = params['model']
    module = importlib.import_module(f"models.{model_name}")

    exp_name = f'{params["experiment"]}_{now}_{model_name}'
    device_num = len(params["GPU"].split(','))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device
    params['multi_gpu'] = True if device_num > 1 else False
    params['local_rank'] = int(os.environ["LOCAL_RANK"])
    dataset_name = params['dataset_name']
    if dataset_name == 'CROHME':
        keywords = ['2014', '2016', '2019']
    elif dataset_name == 'HME100K':
        keywords = ['HME100K']
    dist.init_process_group(backend='nccl', init_method='env://')
    params['rank'] = dist.get_rank()
    for i in range(len(params["loss"]["name"])): print(f'{params["loss"]["name"][i]} loss weight: {params["loss"]["weight"][i]}')

    train_loader = get_crohme_dataset(params)

    if params['rank'] == 0:
        writer = SummaryWriter(f'{params["checkpoint_dir"]}/{exp_name}/TB_logger')
    else:
        writer = None

    if params['rank'] == 0:
        save_path = opj(params['checkpoint_dir'], exp_name, "Code")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for fname in os.listdir('.'):
            if fname in ['geoopt', 'contextual_loss', 'data', 'hyrnn', 'models', 'poincare-embeddings', 'poincare_glove']:
                shutil.copytree(fname, opj(save_path, fname), ignore=shutil.ignore_patterns("*.npy", "*.pth", "*.pyc", "__pycache__"))
            if '.py' in fname or '.yaml' in fname or '.txt' in fname:
                shutil.copy(fname, opj(save_path, fname))

    with torch.cuda.device(params['local_rank']):
        model = getattr(module, model_name)(params).cuda()

        model.name = exp_name
        if params['finetune']:
            load_checkpoint(params['checkpoint'], model, multi_gpu=False)

        if params['rank'] == 0:
            os.system(f'cp {args.cfg} {opj(params["checkpoint_dir"], exp_name, exp_name)}.yaml')
            logging.basicConfig(filename=opj(params["checkpoint_dir"], exp_name, "log.txt"), level=logging.INFO,
                                format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
            logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
            for keyword in keywords:
                with open(opj(params["checkpoint_dir"], exp_name, f"exprate_{keyword}.best"), "w", encoding="utf-8") as er:
                    er.write("0.0")

        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[params['local_rank']])

        for flag in range(params['curr_time']):
            optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), lr=float(params['lr']),
                                                                  eps=float(params['eps']),
                                                                  weight_decay=float(params['weight_decay']))
            if flag: load_checkpoint(opj(params['checkpoint_dir'], exp_name, 'best_model.pth'), model, multi_gpu=params['multi_gpu'])

            print("Start training......")
            for epoch in range(params['epochs']):
                train_loss, train_word_score, train_exprate = train_model(params, model, optimizer, epoch, flag, train_loader, writer, logging=logging)
                dist.barrier()
                train_exprate = torch.tensor(train_exprate).cuda()
                train_word_score = torch.tensor(train_word_score).cuda()
                dist.reduce(train_exprate, 0, op=dist.ReduceOp.SUM)
                dist.reduce(train_word_score, 0, op=dist.ReduceOp.SUM)
                train_exprate /= device_num
                train_word_score /= device_num

                if epoch >= params['valid_start']:
                    if params['rank'] == 0:
                        torch.save({'model': model.module.state_dict()}, f'{os.path.join(params["checkpoint_dir"], model.module.name)}/temp_model.pth')
                        command = [
                            'python', 'test_model_back.py',
                            '--epo', str(epoch),
                            '--expname', exp_name,
                        ]
                        subprocess.Popen(command)

        if params['rank'] == 0:
            command = [
                'python', 'generate_attnmap.py',
                '--dataset', dataset_name,
                '--gpu', str(params['Test_GPU']),
                '--expname', exp_name
            ]
            subprocess.Popen(command)

        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='./config.yaml', type=str)
    args = parser.parse_args()

    main(args)
