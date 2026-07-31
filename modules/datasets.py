import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import random
from .tokenizers import modeEOS

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer,split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        
        self.examples = self.ann[self.split]
  
        for i in range(len(self.examples)):
            repo = []
            repo_eos = []
            for j in range(len(self.examples[i]['report'])):
                tmp_repo = self.examples[i]['report'][j]
                tmp_ids = tokenizer(tmp_repo)[:self.max_seq_length]
                tmp_ids_eos = tmp_ids + [modeEOS]
                tmp_ids = torch.tensor(tmp_ids,dtype=torch.long)
                tmp_ids_eos = torch.tensor(tmp_ids_eos,dtype=torch.long)
                repo.append(tmp_ids)
                repo_eos.append(tmp_ids_eos)
            self.examples[i]['report_ids'] = repo
            self.examples[i]['report_ids_eos'] = repo_eos
      


    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        reports = example['report_ids']
        reports_eos = example['report_ids_eos']
  

        sample = (image_id, image, reports, reports_eos)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        reports = example['report_ids']
        reports_eos = example['report_ids_eos']
        sample = (image_id, image, reports, reports_eos)
        return sample
