import sys
sys.path.append("../")
import os
import torch
import yaml
import logging
logging.getLogger("pytorch_transformers").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("vilbert.utils").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

from easydict import EasyDict as edict
from pytorch_transformers.tokenization_bert import BertTokenizer
from vilbert.datasets import ConceptCapLoaderTrain, ConceptCapLoaderVal
from vilbert.vilbert import VILBertForVLTasks, BertConfig, BertForMultiModalPreTraining
from vilbert.task_utils import LoadDatasetEval
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL

from PIL import Image
import cv2
import argparse
import glob
from types import SimpleNamespace
import pdb

import pickle
import json
from tqdm import tqdm
from scipy.stats import kendalltau
from torch.nn.functional import softmax
from utils import *
from dataset import CaptioningDataset
from torch.utils.data import DataLoader

from math import log
from itertools import chain
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial


class VilbertScore():
    
    def __init__(self, batch_size=32):
        
        self.from_pretrained= "../data/vilbert/multi_task_model.bin"
        self.bert_model="bert-base-uncased"
        self.config_file="../config/bert_base_6layer_6conect.json"        
        self.batch_size=32
        
        
        self.max_seq_length=101
        self.train_batch_size=1
        self.do_lower_case=True
        self.predict_feature=False
        self.seed=42
        self.num_workers=0
        self.baseline=False
        self.img_weight=1
        self.distributed=False
        self.objective=1
        self.visual_target=0
        self.dynamic_attention=False
        self.task_specific_tokens=True
        self.tasks='7'
        self.save_name=''
        self.in_memory=False        
        self.local_rank=-1
        self.split='mteval'
        self.clean_train_sets=True
        self.dataset='flickr8k'
        self.task=7
        self.layer=-1
        self.expname='pretrain_cls_sep'
        self.compute_correlation = False
        self.datadir = "../data/sample"
        
        
        if(self.from_pretrained == 'save/pretrained_model.bin'):
            self.task_specific_tokens = False
    
        with open('../vilbert_tasks.yml', 'r') as f:
            task_cfg = edict(yaml.safe_load(f))
            
        config = BertConfig.from_json_file(self.config_file)
        default_gpu=True

        if self.predict_feature:
            config.v_target_size = 2048
            config.predict_feature = True
        else:
            config.v_target_size = 1601
            config.predict_feature = False

        if self.task_specific_tokens:
            config.task_specific_tokens = True    

        if self.dynamic_attention:
            config.dynamic_attention = True

        config.visualization = True
        num_labels = 3129

        if self.baseline:
            self.model = BaseBertForVLTasks.from_pretrained(
                self.from_pretrained, config=config, num_labels=num_labels, default_gpu=default_gpu
                )
        else:
            self.model = VILBertForVLTasks.from_pretrained(
                self.from_pretrained, config=config, num_labels=num_labels, default_gpu=default_gpu
                )

        self.model.eval()
        cuda = torch.cuda.is_available()
        if cuda: self.model = self.model.cuda(0)
        
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.config = BertConfig.from_json_file(self.config_file)
        self.savedir = self.datadir
        
        
    def loaddata(self, path_image_feature, path_gen_caption, path_gt_caption):
        
        from dataset_custom import CaptioningDataset
        
        dataset = CaptioningDataset(path_image_feature = path_image_feature,
                                path_gen_caption   = path_gen_caption,
                                path_gt_caption   = path_gt_caption,
                                use_idf=False)
        
        self.path_image_feature = path_image_feature
        self.path_gen_caption = path_gen_caption
        self.path_gt_caption = path_gt_caption
        
        dataloader = DataLoader(dataset, self.batch_size, shuffle=False)
        
        self.dataloader = dataloader
        
        
        

    def _process(self, a, tokenizer=None):
        if not tokenizer is None:
            a = ["[CLS]"]+tokenizer.tokenize(a)+["[SEP]"]
            a = tokenizer.convert_tokens_to_ids(a)
        return set(a)

    def _get_idf_dict(self, arr, tokenizer, nthreads=1):

        """
        Returns mapping from word piece index to its inverse document frequency.
        Args:
            - :param: `arr` (list of str) : sentences to process.
            - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
            - :param: `nthreads` (int) : number of CPU threads to use
        """
        idf_count = Counter()
        num_docs = len(arr)

        process_partial = partial(self._process, tokenizer=tokenizer)

        with Pool(nthreads) as p:
            idf_count.update(chain.from_iterable(p.map(sefl._process_partial, arr)))

        idf_dict = defaultdict(lambda : log((num_docs+1)/(1)))
        idf_dict.update({idx:log((num_docs+1)/(c+1)) for (idx, c) in idf_count.items()})
        return idf_dict
    

    def _compute_bert_score(self, text_a, input_mask_a, segment_ids_a, 
                           features, spatials, image_mask, co_attention_mask, input_idf_a, task, x, y, 
                           use_idf=False, layer=-1):
        with torch.no_grad():
            p5 = []
            r5 = []
            f5 = []            

            st_c, sv_c, pt_c, pv_c, att_c = self.model.bert(
                text_a[:,x,:], features, spatials, segment_ids_a[:,x,:],
                input_mask_a[:,x,:], image_mask, co_attention_mask, task,
            output_all_encoded_layers=True)

            for i in range(1):
                st_g, sv_g, pt_g, pv_g, att_g = self.model.bert(
                    text_a[:,y+i,:], features, spatials, segment_ids_a[:,y+i,:],
                    input_mask_a[:,y+i,:], image_mask, co_attention_mask, task,
                output_all_encoded_layers=True)
                if(use_idf):
                    p, r, f = bert_score(st_g[layer], st_c[layer], input_idf_a[:,y+i, :], input_idf_a[:,x, :])
                else:
                    p, r, f = bert_score(st_g[layer], st_c[layer], input_mask_a[:,y+i, :], input_mask_a[:,x, :])
                p5.append(p)
                r5.append(r)
                f5.append(f)        

        p5a = np.average(p5, axis=0)
        r5a = np.average(r5, axis=0)
        f5a = np.average(f5, axis=0) 

        return p5a, r5a, f5a

    
    def compute(self):
        
        print("target data path")
        print("Images: ", self.path_image_feature)
        print("Generated Captions: ", self.path_gen_caption)
        print("Ground truth Captions: ", self.path_gt_caption)
        

        
        layer = self.layer

        prs_a = []
        rcs_a = []
        f1s_a = []

        use_idf = False

        #for idx in tqdm(range(len(scores))):
        for text_a, input_mask_a, segment_ids_a, features, spatials, image_mask, co_attention_mask, input_idf_a, idxs_ in tqdm(iter(self.dataloader)):
            text_a = text_a.cuda() 
            input_idf_a = input_idf_a.cuda()
            input_mask_a = input_mask_a.cuda()
            segment_ids_a = segment_ids_a.cuda()
            features = features.cuda()
            spatials = spatials.cuda()
            image_mask = image_mask.cuda()
            co_attention_mask = co_attention_mask.cuda()
            task = [self.task]
            task = torch.from_numpy(np.array(task)).cuda().unsqueeze(0).repeat(spatials.size(0), 1)
            #break
            
            with torch.no_grad():
                p5a, r5a, f5a = self._compute_bert_score(
                                                   text_a,
                                                   input_mask_a,
                                                   segment_ids_a, 
                                                   features,
                                                   spatials,
                                                   image_mask,
                                                   co_attention_mask,
                                                   input_idf_a,
                                                   task,
                                                   0,
                                                   1,
                                                   layer=self.layer)
                if(len(prs_a) == 0):
                    prs_a = p5a
                    rcs_a = r5a
                    f1s_a = f5a            
                else:
                    prs_a = np.concatenate((prs_a, p5a))
                    rcs_a = np.concatenate((rcs_a, r5a))
                    f1s_a = np.concatenate((f1s_a, f5a))

        return [prs_a, rcs_a, f1s_a]  
    
    


# if(args.compute_correlation):
#     scores = dataset.scores
#     print("Kendall Correlation Coefficient")
#     print("P: %.3f"%kendalltau(scores, prs_a)[0])
#     print("R: %.3f"%kendalltau(scores, rcs_a)[0])
#     print("F: %.3f"%kendalltau(scores, f1s_a)[0])

# # Save the results
# if os.path.isdir("results") == False:
#     os.mkdir("results")

# savefile = "results/" + args.dataset + ".csv"
# df_result = pd.DataFrame(data=[prs_a, rcs_a, f1s_a]).T
# df_result.columns = ["precision", "recall", "f1"]

# print('Saved the results to %s'%savefile)
# df_result.to_csv(savefile, index=False)

# with open(savefile, 'wb') as f:
#     pickle.dump(final_results, f)