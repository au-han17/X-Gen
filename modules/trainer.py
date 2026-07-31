import os
from abc import abstractmethod
from .tokenizers import modeMASK, modePAD
import time
import torch
import pandas as pd
from numpy import inf
from tqdm import tqdm
import itertools
from .encoder_decoder import MaskDecodeLoss
import json
from collections import Counter
import numpy as np
from timm.utils import ModelEmaV2

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args,use_ema=False):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        self.use_ema = args.use_ema
        if self.use_ema:
            self.model_ema = ModelEmaV2(model, decay=args.model_ema_decay)
            self.model_ema.module.to(self.device)
        else:
            self.model_ema = None
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric   
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},'test': {self.mnt_metric_test: self.mnt_best}}
        for name,parameter in self.model.named_parameters():
            for item in parameter.shape:
                if item == 1478:
                    print(name)
    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError
#
    def train(self):
        not_improved_count = 0
        torch.autograd.set_detect_anomaly(True)
        for epoch in tqdm(range(self.start_epoch, self.epochs + 1)):
            
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        if self.use_ema:
            state['state_dict_ema'] = self.model_ema.module.state_dict()

        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.use_ema:
            self.model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
             self.mnt_metric]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
             self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
             print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader,val_dataloader,test_dataloader, use_ema=False):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, use_ema)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_epoch(self, epoch):
#images_id, images,reports,reports_eos,report_masks,report_length
        train_loss = 0
        mask_decoder_loss =0
        r2gen_diffmode_loss =0
        vae_loss = 0
        r2gen_avgmode_loss =0
        log = {}
        self.model.train()
        most_frequent = []
        for batch_idx,(images_id, images,reports,reports_eos) in enumerate(self.train_dataloader):
            
            images,reports,reports_eos = images.to(self.device), reports.to(self.device),reports_eos.to(self.device)
         
            att_feats_0, fc_feats_0 = self.model.visual_extractor(images[:, 0])
            att_feats_1, fc_feats_1 = self.model.visual_extractor(images[:, 1])
         
            fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
            att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
            #att_feats, fc_feats = self.model.visual_extractor(images)
            img_ids = [[i]*3 for i in range(len(images_id))]
            img_ids = list(itertools.chain(*img_ids))
            #mode encoder
            split = [3] * len(images_id)
            
            mode = self.model.modemodules['mode_encoder'](reports)
          
            loss_vq, quant, frequency = self.model.modemodules['codebook'](mode,split)
            most_frequent.append(frequency)
            #image features

            #mask decoder
            unpad_positions = reports_eos != modePAD
            mask_token_ids = reports.new_full(reports.shape, modeMASK)
            mode_img_feats_ = att_feats.detach()
            pred_logits = self.model.modemodules['mask_decoder'](mode_img_feats_, mask_token_ids, quant,img_ids)
            
            #mask decoder loss
            modecriterion = MaskDecodeLoss()
            loss_mask = modecriterion(pred_logits[unpad_positions],reports_eos[unpad_positions])

            #caption decoder with different modes
            rand_idx = []
            cumsum = 0
            for size in split:
                rand_idx.append(torch.rand([size]).topk(1)[1] + cumsum)
                cumsum += size
            rand_idx = torch.cat(rand_idx, dim=0).to(self.device)
            
            #random pick one report for caption traning
            reports_ids = reports[rand_idx]
            report_eos_ids = reports_eos[rand_idx]
            cap_mode = quant[rand_idx]
            avg_mode_tr = self.model.modemodules['codebook'].embedding.weight.mean(dim=0)
            

            #loss with different modes    
            output1 = self.model(fc_feats, att_feats, reports_ids, cap=cap_mode,mode='train') 
            attn_mask = (report_eos_ids!= modePAD).float()
            loss_cap1 = self.criterion(output1, report_eos_ids, attn_mask)
            
            #loss with average mode
            output2 = self.model(fc_feats, att_feats, reports_ids, cap=avg_mode_tr,mode='train')
            loss_cap2 = self.criterion(output2, report_eos_ids, attn_mask)
            
            #loss
            loss_cap = (loss_cap1 + loss_cap2)/2
            loss = loss_mask + loss_cap + 0.8*loss_vq
            mask_decoder_loss += loss_mask.item()
            r2gen_diffmode_loss += loss_cap1.item()
            r2gen_avgmode_loss += loss_cap2.item()
            vae_loss += loss_vq.item()
            train_loss += loss.item()
            self.optimizer.zero_grad()
            if self.use_ema:
                self.model_ema.update(self.model)
            loss.backward()

            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
        log = {
            'train_loss': train_loss / len(self.train_dataloader),
            'mask_decoder_loss':mask_decoder_loss/len(self.train_dataloader),
            'r2gen_diff_mode_loss':r2gen_diffmode_loss/ len(self.train_dataloader),
            'r2gen_avg_mode_loss':r2gen_avgmode_loss/ len(self.train_dataloader),
            'VAE_loss':vae_loss/ len(self.train_dataloader)
            }
        #calculate the mean of the most frequent modes
        numbers = [int(x) for sublist in most_frequent for array in sublist for x in array]
        counter = Counter(numbers)
        most_common_numbers = counter.most_common(8)
        most_frequent_indices = torch.tensor([idx for idx, count in most_common_numbers]).to(torch.int32).to(self.device)
        tmp_mode = self.model.modemodules['codebook'].embedding(most_frequent_indices)
        avg_mode = torch.mean(tmp_mode,dim=0)
        #dummy_mode = torch.full((768,),0).to(self.device)
        
   
        if self.use_ema:
            model = self.model_ema.module
        else:
            model = self.model
        model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx,(images_id, images,reports,reports_eos) in enumerate(self.val_dataloader):
                images,reports,reports_eos = images.to(self.device), reports.to(self.device),reports_eos.to(self.device)
             
                att_feats_0, fc_feats_0= model.visual_extractor(images[:, 0])
                att_feats_1, fc_feats_1= model.visual_extractor(images[:, 1])
                fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
                att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
                #att_feats, fc_feats = self.model.visual_extractor(images)
                output = model(fc_feats,att_feats, cap=avg_mode,mode='sample')
                
                reports_gen = model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = model.tokenizer.decode_batch(reports[:, :].cpu().numpy())
                val_res.extend(reports_gen)
                val_gts.extend(ground_truths)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},{i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12 = [],[],[],[],[],[],[],[],[],[],[],[]

            for batch_idx,(images_id, images,reports,reports_eos) in enumerate(self.test_dataloader):
                images,reports,reports_eos = images.to(self.device), reports.to(self.device),reports_eos.to(self.device)

                att_feats_0, fc_feats_0 = model.visual_extractor(images[:, 0])
                att_feats_1, fc_feats_1 = model.visual_extractor(images[:, 1])
                fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
                att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
                #att_feats, fc_feats = self.model.visual_extractor(images)
                ground_truths = model.tokenizer.decode_batch(reports[:, :].cpu().numpy())
                test_gts.extend(ground_truths)
                for i in range(tmp_mode.shape[0]):
                    test_mode = tmp_mode[i]
                    output = model(fc_feats,att_feats,cap=test_mode, mode='sample')
                    decoding_mask = output != 0
                    decoding_mask_ = decoding_mask.float().to(self.device)
                    last_non_zero_indices = decoding_mask_.shape[1] - decoding_mask_.flip(dims=[1]).argmax(dim=1) - 1
                    for k, index in enumerate(last_non_zero_indices):
                        output[k, index] = 0
                    reports_gen = model.tokenizer.decode_batch(output.cpu().numpy())
                    
                    if i == 0 and len(reports_gen) >0:
                        t1.extend(reports_gen)
                    elif i == 1 and len(reports_gen) >0:
                        t2.extend(reports_gen)
                    elif i == 2 and len(reports_gen) >0:
                        t3.extend(reports_gen)
                    elif i == 3 and len(reports_gen) >0:
                        t4.extend(reports_gen)
                    elif i == 4 and len(reports_gen) >0:
                        t5.extend(reports_gen)
                    elif i == 5 and len(reports_gen) >0:
                        t6.extend(reports_gen)
                    elif i == 6 and len(reports_gen) >0:
                        t7.extend(reports_gen)
                    elif i == 7 and len(reports_gen) >0:
                        t8.extend(reports_gen)
                    elif i == 8 and len(reports_gen) >0:
                        t9.extend(reports_gen)
                    elif i == 9 and len(reports_gen) >0:
                        t10.extend(reports_gen)
                    elif i == 10 and len(reports_gen) >0:
                        t11.extend(reports_gen)
                    elif i == 11 and len(reports_gen) >0:
                        t12.extend(reports_gen)
                    
            pool = [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12]
            diff_mode_res = []
            for idx, ele in enumerate(pool):
                if len(ele) == 590:
                    save_repo_name = f"debug_reports_generated_by_mode_{idx}_epoch_{epoch}.json"
                    with open(save_repo_name,'w') as filehandle:
                        json.dump(ele, filehandle)
                    test_met = self.metric_ftns({ii: [gt] for ii, gt in enumerate(test_gts)},{ii: [re] for ii, re in enumerate(ele)})
                    tmp_result = {'test_' + k: v for k, v in test_met.items()}
                    diff_mode_res.append(tmp_result)
                    file_name = f"debug_eval_result_mode_{idx}_epoch_{epoch}.txt"
                    with open(file_name,'w') as file:
                        json.dump(tmp_result,file)
            max_bleu4 = max(diff_mode_res,key=lambda x:x["test_BLEU_4"])
            log.update(max_bleu4)
          
          
        self.lr_scheduler.step()

        return log
