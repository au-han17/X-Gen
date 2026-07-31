import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset
from .tokenizers import modePAD
from torch.nn.utils.rnn import pad_sequence
import itertools


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer,split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),#将短边缩放至256，长宽比保持不变
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)
#image_id, image, reports, reports_eos,report_masks,report_length
    @staticmethod
    def collate_fn(data):
        
        images_id, images, reports, reports_eos= zip(*data)
        images = torch.stack(images, 0)


        reports = itertools.chain(*reports)
        reports = pad_sequence(list(reports), batch_first=True, padding_value=modePAD)
        reports_eos = itertools.chain(*reports_eos)
        reports_eos = pad_sequence(list(reports_eos), batch_first=True,padding_value=modePAD)

        return images_id, images,reports,reports_eos

