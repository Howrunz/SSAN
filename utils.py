import os
import yaml
import math
import numpy as np
import cv2
import skimage.filters as filters
from difflib import SequenceMatcher
import torch


def gaussian_2d(shape):
    h_interval, w_interval = 0, 0
    if shape[0] < 16:
        height_new = (16 // shape[0] + 1) * shape[0]
        height_new = height_new if (height_new-shape[0])%2==0 else height_new+1
        h_interval = (height_new - shape[0]) // 2
        shape[0] = height_new
    if shape[1] < 16:
        width_new = (16 // shape[1] + 1) * shape[1]
        width_new = width_new if (width_new-shape[1])%2==0 else width_new+1
        w_interval = (width_new - shape[1]) // 2
        shape[1] = width_new

    sigma_x, sigma_y = shape[1]/4, shape[0]/4
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h, h_interval, w_interval


def get_gaussian(img):
    """二维高斯分布"""
    pad = np.zeros_like(img, dtype=np.float32)
    img = (img > filters.threshold_sauvola(img, 43, 0.043)).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    dtype = [('x', int), ('y', int), ('w', int), ('h', int), ('total', int)]
    pixel_dict = np.sort(np.array([tuple(x) for x in stats[1:]], dtype=dtype), order='total')[::-1]
    for item in pixel_dict:
        x, H_interv, W_interv = gaussian_2d([item[3], item[2]])

        H_down = item[1] - H_interv
        if H_down < 0:
            x = x[H_interv-item[1]:, ...]
            H_down = 0

        H_up = item[1] + item[3] + H_interv
        if H_up > img.shape[0]:
            x = x[:x.shape[0]-H_up+img.shape[0], ...]
            H_up = img.shape[0]

        W_down = item[0] - W_interv
        if W_down < 0:
            x = x[..., W_interv-item[0]:]
            W_down = 0

        W_up = item[0] + item[2] + W_interv
        if W_up > img.shape[1]:
            x = x[..., :x.shape[1]-W_up+img.shape[1]]
            W_up = img.shape[1]

        pad[H_down:H_up, W_down:W_up] = x

    return pad

def gen_counting_label(labels, channel, tag):
    b, t = labels.size()
    device = labels.device
    counting_labels = torch.zeros((b, channel))
    if tag:
        ignore = [0, 1, 107, 108, 109, 110]
    else:
        ignore = []
    for i in range(b):
        for j in range(t):
            k = labels[i][j]
            if k in ignore:
                continue
            else:
                counting_labels[i][k] += 1
    return counting_labels.to(device)

def load_config(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except:
        with open(yaml_path, 'r', encoding='UTF-8') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    return params

def update_lr(optimizer, current_epoch, current_step, steps, epochs, initial_lr):
    if current_epoch < 1:
        new_lr = initial_lr / steps * (current_step + 1)
    elif 1 <= current_epoch <= 200:
        new_lr = 0.5 * (1 + math.cos(
            (current_step + 1 + (current_epoch - 1) * steps) * math.pi / (200 * steps))) * initial_lr
    else:
        new_lr = 0.5 * (1 + math.cos(
            (current_step + 1 + (current_epoch - 1) * steps) * math.pi / (epochs * steps))) * initial_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def save_checkpoint(model, optimizer, train_expRate, ExpRate_score, epoch, optimizer_save=False, path='checkpoints'):
    filename = f'{os.path.join(path, model.module.name)}/{model.module.name}_trainExpRate-{train_expRate:.4f}_ExpRate-{ExpRate_score:.4f}_{epoch}.pth'
    if optimizer_save:
        state = {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
    else:
        state = {
            'model': model.module.state_dict()
        }
    torch.save(state, filename)
    torch.save(state, f'{os.path.join(path, model.module.name)}/best_model.pth')
    print(f'Save checkpoint: {filename}\n')
    return filename


def load_checkpoint(path, model, optimizer=None, multi_gpu=True):
    state = torch.load(path, map_location='cpu')
    if optimizer is not None and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    else:
        print(f'No optimizer in the pretrained model')
    if multi_gpu:
        model.module.load_state_dict(state['model'])
    else:
        model.load_state_dict(state['model'])


class Meter:
    def __init__(self, alpha=0.9):
        self.nums = []
        self.exp_mean = 0
        self.alpha = alpha

    @property
    def mean(self):
        return np.mean(self.nums)

    def add(self, num):
        if len(self.nums) == 0:
            self.exp_mean = num
        self.nums.append(num)
        self.exp_mean = self.alpha * self.exp_mean + (1 - self.alpha) * num


def cal_score(word_probs, word_label, mask):
    line_right = 0
    if word_probs is not None:
        _, word_pred = word_probs.max(2)
    word_scores = [SequenceMatcher(None, s1, s2, autojunk=False).ratio() * (len(s1) + len(s2)) / len(s1) / 2 if len(s1) > 0 and len(s2) > 0 else 0
                   for s1, s2 in zip(word_label.cpu().detach().numpy(), word_pred.cpu().detach().numpy())]

    batch_size = len(word_scores)
    for i in range(batch_size):
        if word_scores[i] == 1:
            line_right += 1

    ExpRate = line_right / batch_size
    word_scores = np.mean(word_scores)
    return word_scores, ExpRate

def cal_distance(word1, word2):
    m = len(word1)
    n = len(word2)
    if m*n == 0:
        return m+n
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            a = dp[i-1][j] + 1
            b = dp[i][j-1] + 1
            c = dp[i-1][j-1]
            if word1[i-1] != word2[j-1]:
                c += 1
            dp[i][j] = min(a, b, c)
    return dp[m][n]


def compute_edit_distance(prediction, label):
    prediction = prediction.strip().split(' ')
    label = label.strip().split(' ')
    distance = cal_distance(prediction, label)
    return distance
