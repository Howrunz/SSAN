import os.path
import random
import pickle as pkl
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from utils import get_gaussian


class CROHME_DATASET(Dataset):
    def __init__(self, params, image_path, alphabet, res=True, istrain=True):
        super(CROHME_DATASET, self).__init__()
        self.params = params
        self.data = np.load(image_path, allow_pickle=True).item()
        self.alphabet = alphabet
        self.res = res
        self.istrain = istrain
        self.patch_rate = self.params['dataset']['patch_rate']
        self.mask_percent = self.params['dataset']['mask_percent']
        self.patch_size = self.params['dataset']['patch_size']
        self.ratio = self.params['densenet']['ratio']
        self.similar = ["1", "|", "l", "0", "o", "O", "2", "z", "Z", "3", "B", "\\beta", "5", "s", "S", "6",
                        "b", "9", "q", "p", "C", "c", "I", "P", "x", "X", "v", "V", "a", "\\alpha", "\\ldots",
                        "\\cdots"]

        if self.params['dataset']['localization'] and not self.params['dataset']['localization_finetune']:
            print("Use Gaussizn Origin")
        elif self.params['dataset']['localization_finetune'] and self.istrain:
            print("Use Finetune Localization Map")
            self.auxiliary_data = np.load(self.params['dataset']['localization_finetune_path'], allow_pickle=True).item()

        if self.params['dataset']['similar']:
            print("Use Similar Matrix")
            with open(self.params['dataset']['matrix_path'], 'rb') as f:
                matrix_total = pkl.load(f)
            self.total_matrix = torch.Tensor(matrix_total['TOTAL'])
            self.single_matrix = matrix_total['SINGLE']

    def __len__(self):
        return len(self.data["LABEL"])

    def __getitem__(self, index):
        name = self.data["NAME"][index]
        labels = self.data["LABEL"][index].split()

        image, scale = self.resize_aug(self.img_resize((255 - self.data["HW"][index]), 128))
        image = self.remove_patches(image, self.auxiliary_data[name], labels, scale)[np.newaxis, ...].astype(np.float32) / 255. if self.params['dataset']['symbol_mask'] and self.istrain else image[np.newaxis, ...].astype(np.float32) / 255.
        images = torch.Tensor(image)

        labels = self.alphabet.encode(labels + ['eos'])
        labels = torch.LongTensor(labels)

        # SSAN
        if self.params['dataset']['localization'] and not self.params['dataset']['localization_finetune']:
            if self.params['dataset']['augmentation']: auxiliary = cv2.resize(self.data["GAU"][index], (image.shape[2], image.shape[1]))
            else: auxiliary = self.data["GAU"][index]
            auxiliary = torch.FloatTensor(auxiliary[::32, ::32])
        elif self.params['dataset']['localization_finetune'] and self.istrain:
            if self.params['dataset']['augmentation']:
                auxiliary = np.sum(np.array(self.auxiliary_data[name]), axis=0)
                auxiliary = cv2.resize(auxiliary, (int(auxiliary.shape[1] * scale),
                                                   int(auxiliary.shape[0] * scale)),
                                       interpolation=cv2.INTER_NEAREST)
            else: auxiliary = np.sum(np.array(self.auxiliary_data[name]), axis=0)
            auxiliary = torch.FloatTensor(auxiliary[::2, ::2] > 0.)
        elif self.params['dataset']['localization'] and not self.istrain:
            auxiliary = torch.FloatTensor(self.data["GAU"][index][::32, ::32])
        else:
            auxiliary = None

        return images, labels, name, auxiliary

    def resize_aug(self, img):
        scale = 1
        if self.params['dataset']['augmentation']:
            if self.istrain:
                height, width = img.shape
                scale = random.uniform(self.params['dataset']['augmentation_scale'][0], self.params['dataset']['augmentation_scale'][1])
                img = cv2.resize(img, (int(width*scale), int(height*scale)))
            return img, scale
        else: return img, scale

    def img_resize(self, img, height):
        if self.res:
            width = img.shape[1] * height // img.shape[0]
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        else:
            return img

    def remove_patches(self, image, mask_total, labels, scale):
        if self.params['dataset']['symbol_mask'] and self.istrain:
            if_remove = np.random.choice([True, False], p=[self.patch_rate, 1-self.patch_rate])
            if if_remove:
                mask_result = np.ones_like(image)
                for Lnum, mask in enumerate(mask_total):
                    try:
                        if labels[Lnum] not in self.similar: continue
                    except:
                        continue
                    height, width = image.shape
                    mask = cv2.resize(mask, (int(mask.shape[1] * self.ratio * scale), int(mask.shape[0] * self.ratio * scale)), interpolation=cv2.INTER_NEAREST)[:height, :width]

                    foreground_indices = np.argwhere(mask > 0)
                    num_foreground = len(foreground_indices)
                    num_pixels_to_remove = int(num_foreground * (self.mask_percent / 100))
                    selected_indices = foreground_indices[np.random.choice(num_foreground, num_pixels_to_remove, replace=False)]

                    half_patch = self.patch_size // 2
                    y1 = np.clip(selected_indices[:, 0] - half_patch, 0, height)
                    y2 = np.clip(selected_indices[:, 0] + half_patch, 0, height)
                    x1 = np.clip(selected_indices[:, 1] - half_patch, 0, width)
                    x2 = np.clip(selected_indices[:, 1] + half_patch, 0, width)

                    for i in range(len(selected_indices)):
                        mask_result[y1[i]:y2[i], x1[i]:x2[i]] = 0

                return image * mask_result
            else:
                return image
        else:
            return image

    def collate_fn(self, batch_data):
        max_width, max_height, max_length = 0, 0, 0
        new_batch, name = [], []
        frame_size = 32
        channel = batch_data[0][0].shape[0]

        for item in batch_data:
            max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
            max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
            max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
            new_batch.append(item)
        max_width = max_width - (max_width % frame_size) + frame_size if max_width % frame_size != 0 else max_width
        max_height = max_height - (max_height % frame_size) + frame_size if max_height % frame_size != 0 else max_height

        images, image_masks = torch.zeros((len(new_batch), channel, max_height, max_width)), torch.zeros((len(new_batch), 1, max_height, max_width))
        labels, labels_masks = torch.zeros((len(new_batch), max_length)).long(), torch.zeros((len(new_batch), max_length))
        # SSAN
        if self.params['dataset']['localization']:
            auxiliary = torch.zeros((len(new_batch), channel, max_height//32, max_width//32))
        else: auxiliary = None

        for i in range(len(new_batch)):
            _, h, w = new_batch[i][0].shape
            images[i][:, :h, :w] = new_batch[i][0]
            image_masks[i][:, :h, :w] = 1
            l = new_batch[i][1].shape[0]
            labels[i][:l] = new_batch[i][1]
            labels_masks[i][:l] = 1
            if self.params['dataset']['localization'] and self.params['dataset']['localization_finetune'] and self.istrain:
                auxiliary[i][0, :h//(self.ratio*2), :w//(self.ratio*2)] = new_batch[i][3][:h//(self.ratio*2), :w//(self.ratio*2)]
            name.append(new_batch[i][2])

        # SAM
        if self.params['dataset']['similar'] and self.istrain:
            similar_matrix = torch.eye(max_length).repeat(len(new_batch), 1, 1)
            for mnum in range(len(new_batch)):
                smatrix = self.single_matrix[name[mnum]]
                similar_matrix[mnum, :smatrix.shape[0], :smatrix.shape[1]] = torch.FloatTensor(smatrix)
        else: similar_matrix = None

        result = {'images': images,
                  'images_mask': image_masks,
                  'labels': labels,
                  'labels_mask': labels_masks}
        if self.params['dataset']['localization']: result.update({'auxiliary': auxiliary})
        if self.params['dataset']['similar'] and self.istrain: result.update({'matrix': similar_matrix})
        result.update({'name': name})
        return result

class ALPHABET:
    def __init__(self, alpha_path):
        with open(alpha_path) as f:
            words = f.readlines()
        self.words_dict = {words[i].strip(): i for i in range(len(words))}
        self.words_index_dict = {i: words[i].strip() for i in range(len(words))}

    def __len__(self):
        return len(self.words_dict)

    def encode(self, labels):
        label_index = [self.words_dict[item] for item in labels]
        return label_index

    def decode(self, label_index):
        label = ' '.join([self.words_index_dict[int(item)] for item in label_index])
        return label

def get_crohme_dataset(params):
    alphabet = ALPHABET(params['alphabet_path'])
    params['alphabet_num'] = len(alphabet)

    train_dataset = CROHME_DATASET(params, params['train_image_path'], alphabet)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler,
                              num_workers=params['workers'], collate_fn=train_dataset.collate_fn, pin_memory=True, drop_last=True)
    print(f'train dataset: {len(train_dataset)} train steps: {len(train_loader)} ')

    return train_loader

def make_NPY_file(img_path, printed_dir, label_path, tar_dir, npy_name):
    npy_dict = {
        "HW": [],
        "PR": [],
        "GAU": [],
        "NAME": [],
        "LABEL": []
    }
    with open(img_path, 'rb') as i:
        img_data = pkl.load(i)

    with open(label_path, 'r') as l:
        latex_list = l.readlines()

    for item in latex_list:
        name, *label = item.strip().split()
        hw_img = cv2.resize(img_data[name], (img_data[name].shape[1] * 128 // img_data[name].shape[0], 128), interpolation=cv2.INTER_CUBIC)
        pr_img = cv2.resize(cv2.imread(os.path.join(printed_dir, f'{name}.jpg'), cv2.IMREAD_GRAYSCALE), (hw_img.shape[1], hw_img.shape[0]))
        gaussian_map = get_gaussian(pr_img)

        npy_dict["HW"].append(hw_img)
        npy_dict["PR"].append(pr_img)
        npy_dict["GAU"].append(gaussian_map)
        npy_dict["NAME"].append(name)
        npy_dict["LABEL"].append(' '.join(label))

    np.save(os.path.join(tar_dir, f"{npy_name}_crohme.npy"), npy_dict)


if __name__ == '__main__':
    img_path = './data/19_test_images.pkl'
    printed_dir = './PrintedIMG'
    label_path = './data/19_test_labels.txt'
    tar_dir = './data'
    npy_name = 'Test2019'

    make_NPY_file(img_path, printed_dir, label_path, tar_dir, npy_name)