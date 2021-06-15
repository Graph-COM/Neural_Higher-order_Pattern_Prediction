import numpy as np
import torch
import os
import random
import bisect
import histogram
from tqdm import tqdm

class EarlyStopMonitor(object):
    def __init__(self, max_round=5, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        self.epoch_count += 1
        
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        return self.num_round >= self.max_round


def roc_auc_score_me(y_true, y_score, multi_class='ovo'):
    a = [roc_auc_score(y_true, y_score, multi_class='ovo')]
    if len(y_true.shape) > 1:
        pass
    else:
        nb_classes = max(y_true) + 1
        one_hot_targets = np.eye(nb_classes)[y_true]
    for i in range(len(y_score[0])):
        a.append(roc_auc_score(one_hot_targets[:,i], y_score[:,i], average='weighted'))
    
    return a



class TripletSampler(object):
    def __init__(self, cls_tri, opn_tri, wedge, edge, ts_start, ts_train, ts_val, ts_end, set_all_nodes, DATA, interpretation_type=0, time_prediction_type=0, ablation_type=0):
        """
        This is the data loader. 
        In each epoch, it will be re-initialized, since the scale of different samples are too different. 
        In each epoch, we fix the size to the size of cls_tri, since it is usually the smallest.
        For cls_tri, since we have the constraint that the edge idx is increasing, we need to manually do a permutation.
        For cls_tri and opn_tri, we have src1, src2, dst, ts1, ts2, ts3, edge_idx1, edge_idx2, edge_idx3
        For wedge, we have src1, src2, dst, ts1, ts2, edge_idx1, edge_idx2
        For edge, we have src1, src2, dst, ts_1, edge_idx1
        Here different from the paper, we use cls_tri, opn_tri, wedge, edge(edge) instead of closure, triangle, wedge, edge
        """
        self.DATA = DATA
        self.interpretation_type = interpretation_type
        self.time_prediction_type = time_prediction_type
        self.ablation_type = ablation_type
        if self.interpretation_type > 0:
            self.num_class = 2
        elif self.time_prediction_type > 0:
            self.num_class = 1
        elif self.ablation_type > 0:
            self.num_class = 2
        else:
            self.num_class = 4
        self.set_all_nodes = set_all_nodes
        # unpack data
        self.src_1_cls_tri, self.src_2_cls_tri, self.dst_cls_tri, self.ts_cls_tri_1, self.ts_cls_tri_2, self.ts_cls_tri_3, self.edge_idx_cls_tri_1, self.edge_idx_cls_tri_2, self.edge_idx_cls_tri_3 = cls_tri
        self.src_1_opn_tri, self.src_2_opn_tri, self.dst_opn_tri, self.ts_opn_tri_1, self.ts_opn_tri_2, self.ts_opn_tri_3, self.edge_idx_opn_tri_1, self.edge_idx_opn_tri_2, self.edge_idx_opn_tri_3 = opn_tri
        self.src_1_wedge, self.src_2_wedge, self.dst_wedge, self.ts_wedge_1, self.ts_wedge_2, self.edge_idx_wedge_1, self.edge_idx_wedge_2 = wedge
        self.src_1_edge, self.src_2_edge, self.dst_edge, self.ts_edge_1, self.edge_idx_edge_1 = edge

        self.train_cls_tri_idx = (self.ts_cls_tri_1 > ts_start) * (self.ts_cls_tri_1 <= ts_train)
        self.val_cls_tri_idx = (self.ts_cls_tri_1 > ts_train) * (self.ts_cls_tri_1 <= ts_val)
        self.test_cls_tri_idx = (self.ts_cls_tri_1 > ts_val) * (self.ts_cls_tri_1 <= ts_end)

        self.train_opn_tri_idx = (self.ts_opn_tri_1 > ts_start) * (self.ts_opn_tri_1 <= ts_train)
        self.val_opn_tri_idx = (self.ts_opn_tri_1 > ts_train) * (self.ts_opn_tri_1 <= ts_val)
        self.test_opn_tri_idx = (self.ts_opn_tri_1 > ts_val) * (self.ts_opn_tri_1 <= ts_end)

        self.train_wedge_idx = (self.ts_wedge_1 > ts_start) * (self.ts_wedge_1 <= ts_train)
        self.val_wedge_idx = (self.ts_wedge_1 > ts_train) * (self.ts_wedge_1 <= ts_val)
        self.test_wedge_idx = (self.ts_wedge_1 > ts_val) * (self.ts_wedge_1 <= ts_end)

        self.train_edge_idx = (self.ts_edge_1 > ts_start) * (self.ts_edge_1 <= ts_train)
        self.val_edge_idx = (self.ts_edge_1 > ts_train) * (self.ts_edge_1 <= ts_val)
        self.test_edge_idx = (self.ts_edge_1 > ts_val) * (self.ts_edge_1 <= ts_end)

        # self.total_cls_tri:(sample size, 6): src1, src2, dst, ts1, edge_idx1, ts3(final timestamp); others similar
        self.total_cls_tri = np.concatenate(([self.src_1_cls_tri], [self.src_2_cls_tri], [self.dst_cls_tri], [self.ts_cls_tri_1], [self.edge_idx_cls_tri_1], [self.ts_cls_tri_3]), 0).T
        self.total_opn_tri = np.concatenate(([self.src_1_opn_tri], [self.src_2_opn_tri], [self.dst_opn_tri], [self.ts_opn_tri_1], [self.edge_idx_opn_tri_1], [self.ts_opn_tri_3]), 0).T
        self.total_wedge = np.concatenate(([self.src_1_wedge], [self.src_2_wedge], [self.dst_wedge], [self.ts_wedge_1], [self.edge_idx_wedge_1], [self.ts_wedge_2]), 0).T
        self.total_edge = np.concatenate(([self.src_1_edge], [self.src_2_edge], [self.dst_edge], [self.ts_edge_1], [self.edge_idx_edge_1], [self.ts_edge_1]), 0).T

        # print(self.src_1_cls_tri.shape, self.total_cls_tri.shape)
        self.train_cls_tri = self.total_cls_tri[self.train_cls_tri_idx]
        self.train_opn_tri = self.total_opn_tri[self.train_opn_tri_idx]
        self.train_wedge = self.total_wedge[self.train_wedge_idx]
        self.train_edge = self.total_edge[self.train_edge_idx]

        self.val_cls_tri = self.total_cls_tri[self.val_cls_tri_idx]
        self.val_opn_tri = self.total_opn_tri[self.val_opn_tri_idx]
        self.val_wedge = self.total_wedge[self.val_wedge_idx]
        self.val_edge = self.total_edge[self.val_edge_idx]

        self.test_cls_tri = self.total_cls_tri[self.test_cls_tri_idx]
        self.test_opn_tri = self.total_opn_tri[self.test_opn_tri_idx]
        self.test_wedge = self.total_wedge[self.test_wedge_idx]
        self.test_edge = self.total_edge[self.test_edge_idx]
        
        print('ts start  ',ts_start, 'ts train  ',ts_train, 'ts val  ', ts_val, 'ts end  ', ts_end)
        # print("finish permutation")
        self.size_train = min(len(self.train_cls_tri), len(self.train_opn_tri), len(self.train_wedge), len(self.train_edge))
        self.size_val = min(len(self.val_cls_tri), len(self.val_opn_tri), len(self.val_wedge), len(self.val_edge))
        self.size_test = min(len(self.test_cls_tri), len(self.test_opn_tri), len(self.test_wedge), len(self.test_edge))
        upper_limit_train = 30000
        if self.size_train > upper_limit_train:
            self.size_train = upper_limit_train
            print("upper limit for training", upper_limit_train)
        upper_limit_test_val = 6000
        if self.size_val > upper_limit_test_val:
            self.size_val = upper_limit_test_val
            print("upper limit for val", upper_limit_test_val)
        if self.size_test > upper_limit_test_val:
            self.size_test = upper_limit_test_val
            print("upper limit for testing", upper_limit_test_val)
        
        if (self.interpretation_type == 1) or (self.interpretation_type == 2) or (self.interpretation_type == 3) or (self.interpretation_type == 4) or (self.ablation_type == 1):
            self.train_label = np.concatenate((np.zeros(self.size_train), np.ones(self.size_train)))
            self.val_label = np.concatenate((np.zeros(self.size_val), np.ones(self.size_val)))
            self.test_label = np.concatenate((np.zeros(self.size_test), np.ones(self.size_test)))
            self.train_idx_list = np.arange(self.get_size())
            self.val_idx_list = np.arange(self.get_val_size())
            self.test_idx_list = np.arange(self.get_test_size())
        else:
            self.train_label = np.concatenate((np.zeros(self.size_train), np.ones(self.size_train), np.ones(self.size_train) * 2, np.ones(self.size_train) * 3))
            self.val_label = np.concatenate((np.zeros(self.size_val), np.ones(self.size_val), np.ones(self.size_val) * 2, np.ones(self.size_val) * 3))
            self.test_label = np.concatenate((np.zeros(self.size_test), np.ones(self.size_test), np.ones(self.size_test) * 2, np.ones(self.size_test) * 3))
            self.train_idx_list = np.arange(self.get_size())
            self.val_idx_list = np.arange(self.get_val_size())
            self.test_idx_list = np.arange(self.get_test_size())

        
        self.initialize()
        self.initialize_val()
        self.initialize_test()
        
        self.val_samples_num = len(self.val_data)
        self.test_samples_num = len(self.test_data)
        print("finish dataset")

    def choose_idx(self, a,b,c,d,e,f, idx):
        return a[idx], b[idx], c[idx], d[idx], e[idx], f[idx]

    def initialize(self):
        if self.interpretation_type > 0:
            if self.interpretation_type == 1: # closure(cls_tri) v.s. triangle(opn_tri)
                cls_tri_idx_epoch = np.random.choice(len(self.train_cls_tri), self.size_train, replace=False)
                opn_tri_idx_epoch = np.random.choice(len(self.train_opn_tri), self.size_train, replace=False)
                self.train_data = np.concatenate((self.train_cls_tri[cls_tri_idx_epoch], self.train_opn_tri[opn_tri_idx_epoch]))
                
            elif self.interpretation_type == 2: # closure(cls_tri) + triangle(opn_tri) v.s. wedge(wedge)
                cls_tri_idx_epoch = np.random.choice(len(self.train_cls_tri), int(self.size_train / 2), replace=False)
                opn_tri_idx_epoch = np.random.choice(len(self.train_opn_tri), self.size_train - int(self.size_train / 2), replace=False)
                wedge_idx_epoch = np.random.choice(len(self.train_wedge), self.size_train, replace=False)
                self.train_data = np.concatenate((self.train_cls_tri[cls_tri_idx_epoch], self.train_opn_tri[opn_tri_idx_epoch], self.train_wedge[wedge_idx_epoch]))

            elif self.interpretation_type == 3: # wedge(wedge) v.s. edge(edge)
                wedge_idx_epoch = np.random.choice(len(self.train_wedge), self.size_train, replace=False)
                edge_idx_epoch = np.random.choice(len(self.train_edge), self.size_train, replace=False)
                self.train_data = np.concatenate((self.train_wedge[wedge_idx_epoch], self.train_edge[edge_idx_epoch]))
            elif self.interpretation_type == 4: # closure(cls_tri) v.s. wedge(wedge)
                cls_tri_idx_epoch = np.random.choice(len(self.train_cls_tri), self.size_train, replace=False)
                wedge_idx_epoch = np.random.choice(len(self.train_wedge), self.size_train, replace=False)
                self.train_data = np.concatenate((self.train_cls_tri[cls_tri_idx_epoch], self.train_wedge[wedge_idx_epoch]))
        elif self.ablation_type == 1: # triangle(opn_tri) and wedge(wedge)
            opn_tri_idx_epoch = np.random.choice(len(self.train_opn_tri), self.size_train, replace=False)
            wedge_idx_epoch = np.random.choice(len(self.train_wedge), self.size_train, replace=False)
            self.train_data = np.concatenate((self.train_opn_tri[opn_tri_idx_epoch], self.train_wedge[wedge_idx_epoch]))
                 
        elif self.time_prediction_type > 0:
            if self.time_prediction_type == 1:
                cls_tri_idx_epoch = np.random.choice(len(self.train_cls_tri), self.size_train, replace=False)
                self.train_data = self.train_cls_tri[cls_tri_idx_epoch]                
            elif self.time_prediction_type == 2:
                opn_tri_idx_epoch = np.random.choice(len(self.train_opn_tri), self.size_train, replace=False)
                self.train_data = self.train_opn_tri[opn_tri_idx_epoch]
            elif self.time_prediction_type == 3:
                wedge_idx_epoch = np.random.choice(len(self.train_wedge), self.size_train, replace=False)
                self.train_data = self.train_wedge[wedge_idx_epoch]

            self.train_time_gt = np.float32(self.train_data[cls_tri_idx_epoch][5] - self.train_data[cls_tri_idx_epoch][3])
        else:
            cls_tri_idx_epoch = np.random.choice(len(self.train_cls_tri), self.size_train, replace=False)
            opn_tri_idx_epoch = np.random.choice(len(self.train_opn_tri), self.size_train, replace=False)
            wedge_idx_epoch = np.random.choice(len(self.train_wedge), self.size_train, replace=False)
            edge_idx_epoch = np.random.choice(len(self.train_edge), self.size_train, replace=False)
            self.train_data = np.concatenate((self.train_cls_tri[cls_tri_idx_epoch], self.train_opn_tri[opn_tri_idx_epoch], self.train_wedge[wedge_idx_epoch], self.train_edge[edge_idx_epoch]))

        self.idx = 0
        np.random.shuffle(self.train_idx_list)

    def initialize_val(self):
        if self.interpretation_type > 0:
            if self.interpretation_type == 1: # closure(cls_tri) v.s. triangle(opn_tri)
                cls_tri_idx_epoch = np.random.choice(len(self.val_cls_tri), self.size_val, replace=False)
                opn_tri_idx_epoch = np.random.choice(len(self.val_opn_tri), self.size_val, replace=False)
                self.val_data = np.concatenate((self.val_cls_tri[cls_tri_idx_epoch], self.val_opn_tri[opn_tri_idx_epoch]))
                
            elif self.interpretation_type == 2: # closure(cls_tri) + triangle(opn_tri) v.s. wedge(wedge)
                cls_tri_idx_epoch = np.random.choice(len(self.val_cls_tri), int(self.size_val / 2), replace=False)
                opn_tri_idx_epoch = np.random.choice(len(self.val_opn_tri), self.size_val - int(self.size_val / 2), replace=False)
                wedge_idx_epoch = np.random.choice(len(self.val_wedge), self.size_val, replace=False)
                self.val_data = np.concatenate((self.val_cls_tri[cls_tri_idx_epoch], self.val_opn_tri[opn_tri_idx_epoch], self.val_wedge[wedge_idx_epoch]))

            elif self.interpretation_type == 3: # wedge(wedge) v.s. edge(edge)
                wedge_idx_epoch = np.random.choice(len(self.val_wedge), self.size_val, replace=False)
                edge_idx_epoch = np.random.choice(len(self.val_edge), self.size_val, replace=False)
                self.val_data = np.concatenate((self.val_wedge[wedge_idx_epoch], self.val_edge[edge_idx_epoch]))
            elif self.interpretation_type == 4: # closure(cls_tri) v.s. wedge(wedge)
                cls_tri_idx_epoch = np.random.choice(len(self.val_cls_tri), self.size_val, replace=False)
                wedge_idx_epoch = np.random.choice(len(self.val_wedge), self.size_val, replace=False)
                self.val_data = np.concatenate((self.val_cls_tri[cls_tri_idx_epoch], self.val_wedge[wedge_idx_epoch]))
        elif self.ablation_type == 1: # triangle(opn_tri) and wedge(wedge)
            opn_tri_idx_epoch = np.random.choice(len(self.val_opn_tri), self.size_val, replace=False)
            wedge_idx_epoch = np.random.choice(len(self.val_wedge), self.size_val, replace=False)
            self.val_data = np.concatenate((self.val_opn_tri[opn_tri_idx_epoch], self.val_wedge[wedge_idx_epoch]))
                 
        elif self.time_prediction_type > 0:
            if self.time_prediction_type == 1:
                cls_tri_idx_epoch = np.random.choice(len(self.val_cls_tri), self.size_val, replace=False)
                self.val_data = self.val_cls_tri[cls_tri_idx_epoch]                
            elif self.time_prediction_type == 2:
                opn_tri_idx_epoch = np.random.choice(len(self.val_opn_tri), self.size_val, replace=False)
                self.val_data = self.val_opn_tri[opn_tri_idx_epoch]
            elif self.time_prediction_type == 3:
                wedge_idx_epoch = np.random.choice(len(self.val_wedge), self.size_val, replace=False)
                self.val_data = self.val_wedge[wedge_idx_epoch]

            self.val_time_gt = np.float32(self.val_data[cls_tri_idx_epoch][5] - self.val_data[cls_tri_idx_epoch][3])
        else:
            # print(len(self.val_cls_tri), self.size_val)
            cls_tri_idx_epoch = np.random.choice(len(self.val_cls_tri), self.size_val, replace=False)
            opn_tri_idx_epoch = np.random.choice(len(self.val_opn_tri), self.size_val, replace=False)
            wedge_idx_epoch = np.random.choice(len(self.val_wedge), self.size_val, replace=False)
            edge_idx_epoch = np.random.choice(len(self.val_edge), self.size_val, replace=False)
            self.val_data = np.concatenate((self.val_cls_tri[cls_tri_idx_epoch], self.val_opn_tri[opn_tri_idx_epoch], self.val_wedge[wedge_idx_epoch], self.val_edge[edge_idx_epoch]))

        self.idx = 0
        np.random.shuffle(self.val_idx_list)
    
    def initialize_test(self):
        if self.interpretation_type > 0:
            if self.interpretation_type == 1: # closure(cls_tri) v.s. triangle(opn_tri)
                cls_tri_idx_epoch = np.random.choice(len(self.test_cls_tri), self.size_test, replace=False)
                opn_tri_idx_epoch = np.random.choice(len(self.test_opn_tri), self.size_test, replace=False)
                self.test_data = np.concatenate((self.test_cls_tri[cls_tri_idx_epoch], self.test_opn_tri[opn_tri_idx_epoch]))
                
            elif self.interpretation_type == 2: # closure(cls_tri) + triangle(opn_tri) v.s. wedge(wedge)
                cls_tri_idx_epoch = np.random.choice(len(self.test_cls_tri), int(self.size_test / 2), replace=False)
                opn_tri_idx_epoch = np.random.choice(len(self.test_opn_tri), self.size_test - int(self.size_test / 2), replace=False)
                wedge_idx_epoch = np.random.choice(len(self.test_wedge), self.size_test, replace=False)
                self.test_data = np.concatenate((self.test_cls_tri[cls_tri_idx_epoch], self.test_opn_tri[opn_tri_idx_epoch], self.test_wedge[wedge_idx_epoch]))

            elif self.interpretation_type == 3: # wedge(wedge) v.s. edge(edge)
                wedge_idx_epoch = np.random.choice(len(self.test_wedge), self.size_test, replace=False)
                edge_idx_epoch = np.random.choice(len(self.test_edge), self.size_test, replace=False)
                self.test_data = np.concatenate((self.test_wedge[wedge_idx_epoch], self.test_edge[edge_idx_epoch]))
            elif self.interpretation_type == 4: # closure(cls_tri) v.s. wedge(wedge)
                cls_tri_idx_epoch = np.random.choice(len(self.test_cls_tri), self.size_test, replace=False)
                wedge_idx_epoch = np.random.choice(len(self.test_wedge), self.size_test, replace=False)
                self.test_data = np.concatenate((self.test_cls_tri[cls_tri_idx_epoch], self.test_wedge[wedge_idx_epoch]))
        elif self.ablation_type == 1: # triangle(opn_tri) and wedge(wedge)
            opn_tri_idx_epoch = np.random.choice(len(self.test_opn_tri), self.size_test, replace=False)
            wedge_idx_epoch = np.random.choice(len(self.test_wedge), self.size_test, replace=False)
            self.test_data = np.concatenate((self.test_opn_tri[opn_tri_idx_epoch], self.test_wedge[wedge_idx_epoch]))
                 
        elif self.time_prediction_type > 0:
            if self.time_prediction_type == 1:
                cls_tri_idx_epoch = np.random.choice(len(self.test_cls_tri), self.size_test, replace=False)
                self.test_data = self.test_cls_tri[cls_tri_idx_epoch]                
            elif self.time_prediction_type == 2:
                opn_tri_idx_epoch = np.random.choice(len(self.test_opn_tri), self.size_test, replace=False)
                self.test_data = self.test_opn_tri[opn_tri_idx_epoch]
            elif self.time_prediction_type == 3:
                wedge_idx_epoch = np.random.choice(len(self.test_wedge), self.size_test, replace=False)
                self.test_data = self.test_wedge[wedge_idx_epoch]

            self.test_time_gt = np.float32(self.test_data[cls_tri_idx_epoch][5] - self.test_data[cls_tri_idx_epoch][3])
        else:
            cls_tri_idx_epoch = np.random.choice(len(self.test_cls_tri), self.size_test, replace=False)
            opn_tri_idx_epoch = np.random.choice(len(self.test_opn_tri), self.size_test, replace=False)
            wedge_idx_epoch = np.random.choice(len(self.test_wedge), self.size_test, replace=False)
            edge_idx_epoch = np.random.choice(len(self.test_edge), self.size_test, replace=False)
            self.test_data = np.concatenate((self.test_cls_tri[cls_tri_idx_epoch], self.test_opn_tri[opn_tri_idx_epoch], self.test_wedge[wedge_idx_epoch], self.test_edge[edge_idx_epoch]))

        self.idx = 0
        np.random.shuffle(self.test_idx_list)

    def get_size(self):
        return self.num_class * self.size_train
    
    def get_val_size(self):
        return self.num_class * self.size_val

    def get_test_size(self):
        return self.num_class * self.size_test

    def set_batch_size(self, batch_size):
        self.bs = batch_size
        self.idx = 0
    
    def reset(self):
        self.idx = 0

    def train_samples(self):
        s_idx = self.idx * self.bs
        e_idx = min(self.get_size(), s_idx + self.bs)
        if s_idx == e_idx:
            s_idx = 0
            e_idx = self.bs
            self.idx = 0
            # print("train error")
        batch_idx = self.train_idx_list[s_idx:e_idx]
        src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut = self.train_data[batch_idx,0], self.train_data[batch_idx,1], self.train_data[batch_idx,2], self.train_data[batch_idx,3], self.train_data[batch_idx,4]
        if self.time_prediction_type > 0:
            label_cut = self.train_time_gt[batch_idx]
        else:
            label_cut = self.train_label[batch_idx].astype(int)
        self.idx += 1
        # print(src_1_l_cut, type(src_1_l_cut))
        return src_1_l_cut.astype(int), src_2_l_cut.astype(int), dst_l_cut.astype(int), ts_l_cut, e_l_cut.astype(int), label_cut
    
    def train_samples_baselines(self):
        s_idx = self.idx * self.bs
        e_idx = min(self.get_size(), s_idx + self.bs)
        if s_idx == e_idx:
            s_idx = 0
            e_idx = self.bs
            self.idx = 0
            # print("train error")
        batch_idx = self.train_idx_list[s_idx:e_idx]
        if self.time_prediction_type > 0:
            label_cut = self.train_time_gt[batch_idx]
        else:
            label_cut = self.train_label[batch_idx].astype(int)
        self.idx += 1
        return batch_idx, label_cut

    def val_samples(self, bs = None):
        if bs == None:
            bs = self.bs
        s_idx = self.idx * bs
        e_idx = min(self.get_val_size(), s_idx + bs)
        if s_idx == e_idx:
            s_idx = 0
            e_idx = bs
            self.idx = 0
            # print("val error")
        
        batch_idx = self.val_idx_list[s_idx:e_idx]
        src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut = self.val_data[batch_idx,0], self.val_data[batch_idx,1], self.val_data[batch_idx,2], self.val_data[batch_idx,3], self.val_data[batch_idx,4]
        if self.time_prediction_type > 0:
            label_cut = self.val_time_gt[batch_idx]
        else:
            label_cut = self.val_label[batch_idx].astype(int)
        self.idx += 1
        return src_1_l_cut.astype(int), src_2_l_cut.astype(int), dst_l_cut.astype(int), ts_l_cut, e_l_cut.astype(int), label_cut
    
    def val_samples_baselines(self, bs = None):
        if bs == None:
            bs = self.bs
        s_idx = self.idx * bs
        e_idx = min(self.get_val_size(), s_idx + bs)
        if s_idx == e_idx:
            s_idx = 0
            e_idx = bs
            self.idx = 0
            # print("val error")
        
        batch_idx = self.val_idx_list[s_idx:e_idx]
        e_l_cut = self.val_idx[batch_idx]
        if self.time_prediction_type > 0:
            label_cut = self.val_time_gt[batch_idx]
        else:
            label_cut = self.val_label[batch_idx].astype(int)
        self.idx += 1
        return batch_idx, label_cut

    def test_samples(self, bs = None):
        if bs == None:
            bs = self.bs
        s_idx = self.idx * bs
        e_idx = min(self.get_test_size(), s_idx + bs)
        if s_idx == e_idx:
            s_idx = 0
            e_idx = bs
            self.idx = 0
            print("test error")
        batch_idx = self.test_idx_list[s_idx:e_idx]
        src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut = self.test_data[batch_idx,0], self.test_data[batch_idx,1], self.test_data[batch_idx,2], self.test_data[batch_idx,3], self.test_data[batch_idx,4]
        if self.time_prediction_type > 0:
            label_cut = self.test_time_gt[batch_idx]
        else:
            label_cut = self.test_label[batch_idx].astype(int)
        self.idx += 1
        return src_1_l_cut.astype(int), src_2_l_cut.astype(int), dst_l_cut.astype(int), ts_l_cut, e_l_cut.astype(int), label_cut
    
    def test_samples_baselines(self, bs = None):
        if bs == None:
            bs = self.bs
        s_idx = self.idx * bs
        e_idx = min(self.get_test_size(), s_idx + bs)
        if s_idx == e_idx:
            s_idx = 0
            e_idx = bs
            self.idx = 0
            print("test error")
        batch_idx = self.test_idx_list[s_idx:e_idx]
        e_l_cut = self.test_idx[batch_idx]
        if self.time_prediction_type > 0:
            label_cut = self.test_time_gt[batch_idx]
        else:
            label_cut = self.test_label[batch_idx].astype(int)
        self.idx += 1
        return batch_idx, label_cut

    def inter_label(self, label_cut):
        """
        For interpretation, we have 3 tasks.
        class 0 vs class 1
        class 0 + class 1 vs class 2
        class 2 and class 3

        Abla:
        class 1 vs class 2

        return idx, label_cut
        """
        if self.interpretation_type == 1:
            idx = (label_cut == 0) + (label_cut == 1)
        elif self.interpretation_type == 2:
            idx_0 = label_cut == 0
            idx_1 = label_cut == 1
            idx_2 = label_cut == 2
            label_cut[idx_1] = 0
            label_cut[idx_2] = 1
            idx = idx_0 + idx_1 + idx_2
        elif self.interpretation_type == 3:
            idx_2 = label_cut == 2
            idx_3 = label_cut == 3
            label_cut[idx_2] = 0
            label_cut[idx_3] = 1
            idx = idx_2 + idx_3
        elif self.interpretation_type == 4:
            idx_1 = label_cut == 0
            idx_3 = label_cut == 2
            label_cut[idx_1] = 0
            label_cut[idx_3] = 1
            idx = idx_1 + idx_3    
        elif self.ablation_type == 1:
            idx_2 = label_cut == 1
            idx_3 = label_cut == 2
            label_cut[idx_2] = 0
            label_cut[idx_3] = 1
            idx = idx_2 + idx_3  
        else: # not interpretation
            idx = np.array(np.ones_like(label_cut), dtype=bool)
        return idx, label_cut


from sklearn.metrics import roc_auc_score
def roc_auc_score_multi(x, y):
    return roc_auc_score(x,y,multi_class='ovo')

def roc_auc_score_single(x,y):
    return roc_auc_score(x[:,1],y[:,1])

class edgeTripletSampler(object):
    def __init__(self, samples):
        src_1_list, src_2_list, dst_list, ts_list, e_idx_list = samples
        self.src_1_list = np.array(src_1_list)
        self.src_2_list = np.array(src_2_list)
        self.dst_list = np.array(dst_list)
        self.ts_list = np.array(ts_list)
        self.e_idx_list = np.array(e_idx_list)

    def sample(self, size):
        index = np.random.randint(0, len(self.src_1_list), size)
        return self.src_1_list[index], self.src_2_list[index], self.dst_list[index], self.ts_list[index], self.e_idx_list[index]

class RandTripletSampler(object):
    def __init__(self, samples):
        src_1_list, src_2_list, dst_list, ts_list, e_idx_list = samples
        self.src_1_list = np.concatenate(src_1_list)
        self.src_2_list = np.concatenate(src_2_list)
        self.dst_list = np.concatenate(dst_list)
        self.ts_list = np.concatenate(ts_list)
        self.e_idx_list = np.concatenate(e_idx_list)

    def sample(self, size):
        src_1_index = np.random.randint(0, len(self.src_1_list), size)
        src_2_index = np.random.randint(0, len(self.src_2_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_1_list[src_1_index], self.src_2_list[src_2_index], self.dst_list[dst_index], self.ts_list[src_1_index], self.e_idx_list[src_1_index]

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def process_sampling_numbers(num_neighbors, num_layers):
    num_neighbors = [int(n) for n in num_neighbors]
    if len(num_neighbors) == 1:
        num_neighbors = num_neighbors * num_layers
    else:
        num_layers = len(num_neighbors)
    return num_neighbors, num_layers
