from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from collections import defaultdict
from io import BytesIO  # Added import for BytesIO

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

#from pycocotools.coco import COCO
from skimage import io
import matplotlib.pyplot as plt 
from matplotlib import cm

import nltk, sklearn
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def prepare_data(data):
    imgs, captions, captions_lens, class_ids, keys, wrong_caps, \
     wrong_caps_len, wrong_cls_id, noise, word_labels = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if False:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    noise = noise[sorted_cap_indices]
    word_labels = word_labels[sorted_cap_indices]

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    keys = [keys[i] for i in sorted_cap_indices.numpy()]

    if False:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    w_sorted_cap_lens, w_sorted_cap_indices = \
        torch.sort(wrong_caps_len, 0, True)

    wrong_caps = wrong_caps[w_sorted_cap_indices].squeeze()
    wrong_cls_id = wrong_cls_id[w_sorted_cap_indices].numpy()

    if False:
        wrong_caps = Variable(wrong_caps).cuda()
        w_sorted_cap_lens = Variable(w_sorted_cap_lens).cuda()
    else:
        wrong_caps = Variable(wrong_caps)
        w_sorted_cap_lens = Variable(w_sorted_cap_lens)


    ##
    return [real_imgs, captions, sorted_cap_lens,
            class_ids, keys, wrong_caps, w_sorted_cap_lens, wrong_cls_id, noise, word_labels]



def get_imgs(img_path, bbox, imsize, do_augment=False, image_cache=None):
    """
    Load image with caching of raw bytes to improve performance on repeated accesses.
    Raw bytes are cached before any transformations like cropping to maintain compression.
    """
    if image_cache is None: image_cache = {}
    if img_path in image_cache:
        raw_bytes = image_cache[img_path]
    else:
        with open(img_path, 'rb') as f:
            raw_bytes = f.read()
        image_cache[img_path] = raw_bytes

    img = Image.open(BytesIO(raw_bytes)).convert('RGB')
    width, height = img.size

    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    w, h = img.size
    if do_augment:
        if random.random() < 0.5:
            img = F.hflip(img)
        crop_side = random.randint(int(min(w, h) * 0.7), int(min(w, h) * 1.0))
        left = random.randint(0, w - crop_side)
        top = random.randint(0, h - crop_side)
        img = F.crop(img, top, left, crop_side, crop_side)
        img = F.resize(img, (imsize, imsize), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    else:
        # if w != h:
        #     min_side = min(w, h)
        #     left = (w - min_side) // 2
        #     top = (h - min_side) // 2
        #     img = F.crop(img, top, left, min_side, min_side)
        crop_side = int(min(w, h) * 0.9)
        left = random.randint(0, w - crop_side)
        top = random.randint(0, h - crop_side)
        img = F.crop(img, top, left, crop_side, crop_side)
        img = F.resize(img, (imsize, imsize), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)

    return img

class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train'):
        self.transform = None
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = None
        self.embeddings_num = 10
        self.imsize = 256
        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)
        self.split = split
        self.filenames, self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(data_dir, split)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)
        self.image_cache = {}
        print(f"CUB200 {split} dataset loaded with {len(self)} examples")

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    from nltk.tokenize import RegexpTokenizer
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            # this train_captions_new hold index of each word in sentence
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new, ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                print("filepath", filepath)
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print(f'Loaded from: {filepath}, Vocab size: {n_words}')
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names

        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding='latin1')
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((18, 1), dtype='int64')
        x_len = num_words
        if num_words <= 18:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  
            np.random.shuffle(ix)
            ix = ix[:18]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = 18
        return x, x_len

    def __getitem__(self, global_index):
        index = global_index // self.embeddings_num
        key = self.filenames[index]
        cls_id = self.class_id[index]
        # print(f"glindex: {global_index}, index: {index}, key: {key}, cls_id: {cls_id}")

        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir

        img_name = f'{data_dir}/images/{key}.jpg'
        imgs = get_imgs(img_name, bbox=None, imsize=self.imsize, do_augment=self.split == 'train', image_cache=self.image_cache)
        imgs = np.array(imgs) / 255.0
        imgs = imgs.transpose(2, 0, 1)

        # sent_ix = random.randint(0, self.embeddings_num)
        # new_sent_ix = index * self.embeddings_num + sent_ix
        new_sent_ix = global_index
        caps, cap_len = self.get_caption(new_sent_ix)

        return {
            "img": imgs,
            "input_ids": torch.from_numpy(caps).squeeze(-1),
            "attention_mask": torch.ones((caps.shape[0],), dtype=torch.bool)
        }

    def __len__(self):
        return len(self.filenames) * self.embeddings_num