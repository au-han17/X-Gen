import torch.nn as nn
import numpy as np
from modules.config import _C as cfg
from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer,modemodules):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.cap_model = EncoderDecoder(args, tokenizer)
        self.modemodules = modemodules
        #self.modeEmbeddings = ModeEmbeddings
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        #product np.prod([2,3,4])==24
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
    #a default string representation of an object in Python:the class name and the object's memory address.when you use super().__str__() in a subclass of torch.nn.Module, you will get a string representation that describes the model's structure rather than just the class name and memory address
    def forward_iu_xray(self, fc_feats, att_feats,targets=None, mode='train',*,cap):
        
        if mode == 'train':
            output = self.cap_model(fc_feats, att_feats, targets, cap = cap,mode='forward')
        elif mode == 'sample':
            output, _ = self.cap_model(fc_feats, att_feats, cap =cap, mode='sample')
        else:
            raise ValueError
        return output
    

    def forward_mimic_cxr(self, fc_feats, att_feats, targets=None, mode='train',*,cap):
        
        if mode == 'train':
            output = self.cap_model(fc_feats, att_feats, targets, cap=cap,mode='forward')
        elif mode == 'sample':
            output, _ = self.cap_model(fc_feats, att_feats, cap=cap,mode='sample')
        else:
            raise ValueError
        return output

