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

# preprocess dataset
def preprocess_dataset(ts_list, src_list, dst_list, node_max, edge_idx_list, label_list, time_window_factor=0.05, time_start_factor=0.4):
    t_max = ts_list.max()
    t_min = ts_list.min()
    time_window = time_window_factor * (t_max - t_min)
    time_start = t_min + time_start_factor * (t_max - t_min)
    time_end = t_max - time_window_factor * (t_max - t_min)
    edges = {}  # edges dict: all the edges
    edges_idx = {}  # edges index dict: all the edges index, corresponding with edges
    adj_list = {}
    node_idx = {}
    node_simplex = {}
    ts_last = -1
    simplex_idx = 0
    simplex_ts = []
    list_simplex = set()
    node_first_time= {}
    node2simplex = None
    simplex_ts.append(ts_list[0])
    print(max(label_list))
    for i in tqdm(range(len(src_list))):
        ts = ts_list[i]
        src = src_list[i]
        tgt = dst_list[i]
        node_idx[src] = 1
        node_idx[tgt] = 1
        if (i>0) and (label_list[i] != label_list[i-1]):
            simplex_ts.append(ts)
        if (i>0) and (label_list[i] != label_list[i-1]):
            for _i in list(list_simplex):
                for _j in list(list_simplex):
                    for _l in list(list_simplex):
                        if (_i, _j, _l) in node_simplex:
                            continue
                        if len(set([_i, _j, _l])) != 3:
                            continue
                        
                        if ((_i, _j) in edges) and (edges[(_i, _j)] <= time_end) and (node_first_time[_l] < edges[(_i, _j)]): 
                        # assume w first appear at the same time as edge(u,v), then no previous information, no way to predict
                            timing = ts_list[i-1] - edges[(_i, _j)] 
                            if (timing > 0) and (timing < time_window):
                                node_simplex[(_i, _j, _l)] = simplex_idx
                                
            simplex_idx += 1
            list_simplex = set()

        list_simplex.add(src)
        list_simplex.add(tgt)
        
        if src in node_first_time:
            node_first_time[src] = min(node_first_time[src], ts)
        else:
            node_first_time[src] = ts

        if tgt in node_first_time:
            node_first_time[tgt] = min(node_first_time[tgt], ts)
        else:
            node_first_time[tgt] = ts

        if (src, tgt) in edges:
            if edges[(src, tgt)] > ts:
                edges[(src, tgt)] = ts
                edges_idx[(src, tgt)] = edge_idx_list[i]
        else:
            edges[(src, tgt)] = ts
            edges_idx[(src, tgt)] = edge_idx_list[i]
            if src in adj_list:
                adj_list[src].append(tgt)
            else:
                adj_list[src] = [tgt]
        # simplex, consider edge as undirected
        src = dst_list[i]
        tgt = src_list[i]
        if (src, tgt) in edges:
            if edges[(src, tgt)] > ts:
                edges[(src, tgt)] = ts
                edges_idx[(src, tgt)] = edge_idx_list[i]
        else:
            edges[(src, tgt)] = ts
            edges_idx[(src, tgt)] = edge_idx_list[i]
            if src in adj_list:
                adj_list[src].append(tgt)
            else:
                adj_list[src] = [tgt]
    
    print("node from ", min(node_idx), ' to ', max(node_idx))
    print('total nodes out  ', len(adj_list.keys()))
    print('total nodes  ', len(node_idx.keys()))
    print('simplex time', len(simplex_ts))
    print("close triangle", len(node_simplex.keys()))
    return find_triangle_closure(ts_list, node_max, edges, adj_list, edges_idx, node_simplex, simplex_ts, node_first_time, node2simplex, time_window_factor, time_start_factor)

def find_triangle_closure(ts_list, node_max, edges, adj_list, edges_idx, node_simplex, simplex_ts, node_first_time, node2simplex, time_window_factor, time_start_factor=0.4):
    positive_three_cycle = []
    positive_two_cycle = []
    positive_three_ffw = []
    positive_two_ffw = []
    negative = []
    node_max = int(node_max)
    t_max = ts_list.max()
    t_min = ts_list.min()
    time_window = time_window_factor * (t_max - t_min)
    time_start = t_min + time_start_factor * (t_max - t_min)
    time_end = t_max - time_window_factor * (t_max - t_min)
    
    # close triangle
    src_1_cls_tri = []
    src_2_cls_tri = []
    dst_cls_tri = []
    ts_cls_tri_1 = []
    ts_cls_tri_2 = []
    ts_cls_tri_3 = []
    edge_idx_cls_tri_1 = []
    edge_idx_cls_tri_2 = []
    edge_idx_cls_tri_3 = []
    count_cls_tri = 0

    # open triangle
    src_1_opn_tri = [] # feed forward
    src_2_opn_tri = []
    dst_opn_tri = []
    ts_opn_tri_1 = []
    ts_opn_tri_2 = []
    ts_opn_tri_3 = []
    edge_idx_opn_tri_1 = []
    edge_idx_opn_tri_2 = []
    edge_idx_opn_tri_3 = []
    count_opn_tri = 0

    # wedge
    src_1_wedge = []
    src_2_wedge = []
    dst_wedge = []
    ts_wedge_1 = []
    ts_wedge_2 = []
    count_wedge = 0
    edge_idx_wedge_1 = []
    edge_idx_wedge_2 = []

    # negative(only one edge between the first two nodes in three nodes)
    src_1_neg = []
    src_2_neg = []
    dst_neg = []
    ts_neg_1 = []
    edge_idx_neg_1 = []
    count_negative = 0 # <a,b>

    set_all_node = set(adj_list.keys())
    print(len(list(set_all_node)))

    dict_processed_bool = {}

    for k_idx, edge_i in enumerate(edges.keys()): # first edge
        i = edge_i[0]
        j = edge_i[1]
        if not (j in adj_list): # exist second edge
            continue
        # second edge (j,l) 
        x1 = edges[edge_i]
        if (x1 < time_start) or (x1 > time_end):
            continue
        x1_idx = edges_idx[edge_i]
        

        """
        deal with no interaction with the third nodes
        set_all_nodes - {(i, x)} - {(j,x)}
        calculate the original situation (only one link between the first two nodes)
        """
        if not ((i,j) in dict_processed_bool):
            dict_processed_bool[(i,j)] = 1
            dict_processed_bool[(j,i)] = 1
            set_negative = list(set_all_node - set(adj_list[j]) - set(adj_list[i]))
            for l in set_negative:
                # (i,j,l)
                if node_first_time[l] <= x1:
                    src_1_neg.append(i)
                    src_2_neg.append(j)
                    dst_neg.append(l)
                    ts_neg_1.append(x1)
                    edge_idx_neg_1.append(x1_idx)
                    count_negative += 1

        for l in adj_list[j]:
            if (l==j) or (l==i) or (node_first_time[l] >= x1):
                continue

            x2 = edges[(j,l)]
            x2_idx = edges_idx[(j,l)]
            
            if (x2 - x1 > time_window):
                src_1_neg.append(i)
                src_2_neg.append(j)
                dst_neg.append(l)
                ts_neg_1.append(x1)
                edge_idx_neg_1.append(x1_idx)
                count_negative += 1
                continue

            if (x1 > x2) or (x1 == x2 and x1_idx > x2_idx): # TODO: x1 >= x2
                continue

            l3 = 0             
            if (l,i) in edges:
                x3 = edges[(l,i)]
                x3_idx = edges_idx[(l,i)]
                if x3 - x1 > time_window:
                    src_1_neg.append(i)
                    src_2_neg.append(j)
                    dst_neg.append(l)
                    ts_neg_1.append(x1)
                    edge_idx_neg_1.append(x1_idx)
                    count_negative += 1
                    continue
                if ((x3 > x2) or (x3 == x2 and x3_idx > x2_idx)) and (x3 - x1 < time_window) and (x3 - x1 > 0): #TODO: x3 > x2
                    l3 = 1

            l1 = (i, j, l) in node_simplex
            if l1:                
                _ts = simplex_ts[node_simplex[(i, j, l)]]
                # (i,j,l)
                src_1_cls_tri.append(i)
                src_2_cls_tri.append(j)
                dst_cls_tri.append(l)
                ts_cls_tri_1.append(x1)
                ts_cls_tri_2.append(_ts) # changed
                ts_cls_tri_3.append(_ts) # changed
                edge_idx_cls_tri_1.append(x1_idx)
                edge_idx_cls_tri_2.append(x2_idx)
                edge_idx_cls_tri_3.append(x3_idx)

                # total positive cycle
                count_cls_tri += 1

            elif l3 == 1: # Triangle
                src_1_opn_tri.append(i)
                src_2_opn_tri.append(j)
                dst_opn_tri.append(l)
                ts_opn_tri_1.append(x1)
                ts_opn_tri_2.append(x2)
                ts_opn_tri_3.append(x3)
                edge_idx_opn_tri_1.append(x1_idx)
                edge_idx_opn_tri_2.append(x2_idx)
                edge_idx_opn_tri_3.append(x3_idx)

                # total positive cycle
                count_opn_tri += 1

            
            elif l3 == 0: # Wedge
                if (x2 - x1 > 0) and (x2 - x1 < time_window):
                    src_1_wedge.append(i)
                    src_2_wedge.append(j)
                    dst_wedge.append(l)

                    ts_wedge_1.append(x1)
                    ts_wedge_2.append(x2)
                    edge_idx_wedge_1.append(x1_idx)
                    edge_idx_wedge_2.append(x2_idx)
                    count_wedge += 1
           
    cls_tri = [np.array(src_1_cls_tri), np.array(src_2_cls_tri), np.array(dst_cls_tri), np.array(ts_cls_tri_1), np.array(ts_cls_tri_2), np.array(ts_cls_tri_3), np.array(edge_idx_cls_tri_1), np.array(edge_idx_cls_tri_2), np.array(edge_idx_cls_tri_3)]
    opn_tri = [np.array(src_1_opn_tri), np.array(src_2_opn_tri), np.array(dst_opn_tri), np.array(ts_opn_tri_1), np.array(ts_opn_tri_2), np.array(ts_opn_tri_3), np.array(edge_idx_opn_tri_1), np.array(edge_idx_opn_tri_2), np.array(edge_idx_opn_tri_3)]
    wedge = [np.array(src_1_wedge), np.array(src_2_wedge), np.array(dst_wedge), np.array(ts_wedge_1), np.array(ts_wedge_2), np.array(edge_idx_wedge_1), np.array(edge_idx_wedge_2)]
    nega = [np.array(src_1_neg), np.array(src_2_neg), np.array(dst_neg), np.array(ts_neg_1), np.array(edge_idx_neg_1)]

    print("Total sample number:  Cls Tri:  ", count_cls_tri, "Opn Tri:  ", count_opn_tri, "Wedge:  ", count_wedge, "Neg:  ", count_negative)
    return cls_tri, opn_tri, wedge, nega, set_all_node

class TripletSampler(object):
    def __init__(self, cls_tri, opn_tri, wedge, nega, ts_start, ts_train, ts_val, ts_end, set_all_nodes, DATA, interpretation_type=0, time_prediction_type=0, ablation_type=0):
        """
        This is the data loader. 
        In each epoch, it will be re-initialized, since the scale of different samples are too different. 
        In each epoch, we fix the size to the size of cls_tri, since it is usually the smallest.
        For cls_tri, since we have the constraint that the edge idx is increasing, we need to manually do a permutation.
        For cls_tri and opn_tri, we have src1, src2, dst, ts1, ts2, ts3, edge_idx1, edge_idx2, edge_idx3
        For 
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
        self.src_1_neg, self.src_2_neg, self.dst_neg, self.ts_neg_1, self.edge_idx_neg_1 = nega

        self.train_cls_tri_idx = (self.ts_cls_tri_1 > ts_start) * (self.ts_cls_tri_1 <= ts_train)
        self.val_cls_tri_idx = (self.ts_cls_tri_1 > ts_train) * (self.ts_cls_tri_1 <= ts_val)
        self.test_cls_tri_idx = (self.ts_cls_tri_1 > ts_val) * (self.ts_cls_tri_1 <= ts_end)

        self.train_opn_tri_idx = (self.ts_opn_tri_1 > ts_start) * (self.ts_opn_tri_1 <= ts_train)
        self.val_opn_tri_idx = (self.ts_opn_tri_1 > ts_train) * (self.ts_opn_tri_1 <= ts_val)
        self.test_opn_tri_idx = (self.ts_opn_tri_1 > ts_val) * (self.ts_opn_tri_1 <= ts_end)

        self.train_wedge_idx = (self.ts_wedge_1 > ts_start) * (self.ts_wedge_1 <= ts_train)
        self.val_wedge_idx = (self.ts_wedge_1 > ts_train) * (self.ts_wedge_1 <= ts_val)
        self.test_wedge_idx = (self.ts_wedge_1 > ts_val) * (self.ts_wedge_1 <= ts_end)

        self.train_neg_idx = (self.ts_neg_1 > ts_start) * (self.ts_neg_1 <= ts_train)
        self.val_neg_idx = (self.ts_neg_1 > ts_train) * (self.ts_neg_1 <= ts_val)
        self.test_neg_idx = (self.ts_neg_1 > ts_val) * (self.ts_neg_1 <= ts_end)

        self.train_src_1_cls_tri, self.train_src_2_cls_tri, self.train_dst_cls_tri, self.train_ts_cls_tri, self.train_edge_idx_cls_tri, self.train_endtime_cls_tri = self.choose_idx(self.src_1_cls_tri, self.src_2_cls_tri, self.dst_cls_tri, self.ts_cls_tri_1, self.edge_idx_cls_tri_1, self.ts_cls_tri_3, self.train_cls_tri_idx)
        self.val_src_1_cls_tri, self.val_src_2_cls_tri, self.val_dst_cls_tri, self.val_ts_cls_tri, self.val_edge_idx_cls_tri, self.val_endtime_cls_tri = self.choose_idx(self.src_1_cls_tri, self.src_2_cls_tri, self.dst_cls_tri, self.ts_cls_tri_1, self.edge_idx_cls_tri_1, self.ts_cls_tri_3, self.val_cls_tri_idx)
        self.test_src_1_cls_tri, self.test_src_2_cls_tri, self.test_dst_cls_tri, self.test_ts_cls_tri, self.test_edge_idx_cls_tri, self.test_endtime_cls_tri = self.choose_idx(self.src_1_cls_tri, self.src_2_cls_tri, self.dst_cls_tri, self.ts_cls_tri_1, self.edge_idx_cls_tri_1, self.ts_cls_tri_3, self.test_cls_tri_idx)

        self.train_src_1_opn_tri, self.train_src_2_opn_tri, self.train_dst_opn_tri, self.train_ts_opn_tri, self.train_edge_idx_opn_tri, self.train_endtime_opn_tri = self.choose_idx(self.src_1_opn_tri, self.src_2_opn_tri, self.dst_opn_tri, self.ts_opn_tri_1, self.edge_idx_opn_tri_1, self.ts_opn_tri_3, self.train_opn_tri_idx)
        self.val_src_1_opn_tri, self.val_src_2_opn_tri, self.val_dst_opn_tri, self.val_ts_opn_tri, self.val_edge_idx_opn_tri, self.val_endtime_opn_tri = self.choose_idx(self.src_1_opn_tri, self.src_2_opn_tri, self.dst_opn_tri, self.ts_opn_tri_1, self.edge_idx_opn_tri_1, self.ts_opn_tri_3, self.val_opn_tri_idx)
        self.test_src_1_opn_tri, self.test_src_2_opn_tri, self.test_dst_opn_tri, self.test_ts_opn_tri, self.test_edge_idx_opn_tri, self.test_endtime_opn_tri = self.choose_idx(self.src_1_opn_tri, self.src_2_opn_tri, self.dst_opn_tri, self.ts_opn_tri_1, self.edge_idx_opn_tri_1, self.ts_opn_tri_3, self.test_opn_tri_idx)

        self.train_src_1_wedge, self.train_src_2_wedge, self.train_dst_wedge, self.train_ts_wedge, self.train_edge_idx_wedge, self.train_endtime_wedge = self.choose_idx(self.src_1_wedge, self.src_2_wedge, self.dst_wedge, self.ts_wedge_1, self.edge_idx_wedge_1, self.ts_wedge_2, self.train_wedge_idx)
        self.val_src_1_wedge, self.val_src_2_wedge, self.val_dst_wedge, self.val_ts_wedge, self.val_edge_idx_wedge, self.val_endtime_wedge = self.choose_idx(self.src_1_wedge, self.src_2_wedge, self.dst_wedge, self.ts_wedge_1, self.edge_idx_wedge_1, self.ts_wedge_2, self.val_wedge_idx)
        self.test_src_1_wedge, self.test_src_2_wedge, self.test_dst_wedge, self.test_ts_wedge, self.test_edge_idx_wedge, self.test_endtime_wedge = self.choose_idx(self.src_1_wedge, self.src_2_wedge, self.dst_wedge, self.ts_wedge_1, self.edge_idx_wedge_1, self.ts_wedge_2, self.test_wedge_idx)

        self.train_src_1_neg, self.train_src_2_neg, self.train_dst_neg, self.train_ts_neg, self.train_edge_idx_neg, _ = self.choose_idx(self.src_1_neg, self.src_2_neg, self.dst_neg, self.ts_neg_1, self.edge_idx_neg_1, self.ts_neg_1, self.train_neg_idx)
        self.val_src_1_neg, self.val_src_2_neg, self.val_dst_neg, self.val_ts_neg, self.val_edge_idx_neg, _ = self.choose_idx(self.src_1_neg, self.src_2_neg, self.dst_neg, self.ts_neg_1, self.edge_idx_neg_1, self.ts_neg_1, self.val_neg_idx)
        self.test_src_1_neg, self.test_src_2_neg, self.test_dst_neg, self.test_ts_neg, self.test_edge_idx_neg, _ = self.choose_idx(self.src_1_neg, self.src_2_neg, self.dst_neg, self.ts_neg_1, self.edge_idx_neg_1, self.ts_neg_1, self.test_neg_idx)

        print('ts start  ',ts_start, 'ts train  ',ts_train, 'ts val  ', ts_val, 'ts end  ', ts_end)
        print("finish permutation")
        self.size = min(len(self.train_ts_cls_tri), len(self.train_ts_opn_tri), len(self.train_ts_wedge), len(self.train_ts_neg))
        self.size_val = min(len(self.val_ts_cls_tri), len(self.val_ts_opn_tri), len(self.val_ts_wedge), len(self.val_ts_neg))
        self.size_test = min(len(self.test_ts_cls_tri), len(self.test_ts_opn_tri), len(self.test_ts_wedge), len(self.test_ts_neg))
        upper_limit_train = 30000
        if self.size > upper_limit_train:
            self.size = upper_limit_train
            print("upper limit for training", upper_limit_train)
        upper_limit_test_val = 6000
        if self.size_val > upper_limit_test_val:
            self.size_val = upper_limit_test_val
            print("upper limit for val", upper_limit_test_val)
        if self.size_test > upper_limit_test_val:
            self.size_test = upper_limit_test_val
            print("upper limit for testing", upper_limit_test_val)
        
        if (self.interpretation_type == 1) or (self.interpretation_type == 2) or (self.interpretation_type == 3) or (self.interpretation_type == 4) or (self.ablation_type == 1):
            self.train_label_t = np.concatenate((np.zeros(self.size), np.ones(self.size)))
            self.val_label = np.concatenate((np.zeros(self.size_val), np.ones(self.size_val)))
            self.test_label = np.concatenate((np.zeros(self.size_test), np.ones(self.size_test)))
            self.train_idx_list = np.arange(self.get_size())
            self.val_idx_list = np.arange(self.get_val_size())
            self.test_idx_list = np.arange(self.get_test_size())
        else:
            self.train_label_t = np.concatenate((np.zeros(self.size), np.ones(self.size) * 1, np.ones(self.size) * 2, np.ones(self.size) * 3))
            self.val_label = np.concatenate((np.zeros(self.size_val), np.ones(self.size_val), np.ones(self.size_val) * 2, np.ones(self.size_val) * 3))
            self.test_label = np.concatenate((np.zeros(self.size_test), np.ones(self.size_test), np.ones(self.size_test) * 2, np.ones(self.size_test) * 3))
            self.train_idx_list = np.arange(self.get_size())
            self.val_idx_list = np.arange(self.get_val_size())
            self.test_idx_list = np.arange(self.get_test_size())

        
        self.initialize()
        self.initialize_val()
        self.initialize_test()
        
        self.val_samples_num = len(self.val_src_1)
        self.test_samples_num = len(self.test_src_1)
        print("finish dataset")

    def choose_idx(self, a,b,c,d,e,f, idx):
        return a[idx], b[idx], c[idx], d[idx], e[idx], f[idx]

    def initialize(self):
        if self.interpretation_type > 0:
            if self.interpretation_type == 1:
                cls_tri_idx_epoch = np.random.choice(len(self.train_src_1_cls_tri), self.size, replace=False)
                opn_tri_idx_epoch = np.random.choice(len(self.train_src_1_opn_tri), self.size, replace=False)
                self.train_src_1 = np.concatenate((self.train_src_1_cls_tri[cls_tri_idx_epoch], self.train_src_1_opn_tri[opn_tri_idx_epoch]))
                self.train_src_2 = np.concatenate((self.train_src_2_cls_tri[cls_tri_idx_epoch], self.train_src_2_opn_tri[opn_tri_idx_epoch]))
                self.train_dst = np.concatenate((self.train_dst_cls_tri[cls_tri_idx_epoch], self.train_dst_opn_tri[opn_tri_idx_epoch]))
                self.train_ts = np.concatenate((self.train_ts_cls_tri[cls_tri_idx_epoch], self.train_ts_opn_tri[opn_tri_idx_epoch]))
                self.train_idx = np.concatenate((self.train_edge_idx_cls_tri[cls_tri_idx_epoch], self.train_edge_idx_opn_tri[opn_tri_idx_epoch]))
            elif self.interpretation_type == 2:
                cls_tri_idx_epoch = np.random.choice(len(self.train_src_1_cls_tri), int(self.size / 2), replace=False)
                opn_tri_idx_epoch = np.random.choice(len(self.train_src_1_opn_tri), self.size - int(self.size / 2), replace=False)
                wedge_idx_epoch = np.random.choice(len(self.train_src_1_wedge), self.size, replace=False)
                self.train_src_1 = np.concatenate((self.train_src_1_cls_tri[cls_tri_idx_epoch], self.train_src_1_opn_tri[opn_tri_idx_epoch], self.train_src_1_wedge[wedge_idx_epoch]))
                self.train_src_2 = np.concatenate((self.train_src_2_cls_tri[cls_tri_idx_epoch], self.train_src_2_opn_tri[opn_tri_idx_epoch], self.train_src_2_wedge[wedge_idx_epoch]))
                self.train_dst = np.concatenate((self.train_dst_cls_tri[cls_tri_idx_epoch], self.train_dst_opn_tri[opn_tri_idx_epoch], self.train_dst_wedge[wedge_idx_epoch]))
                self.train_ts = np.concatenate((self.train_ts_cls_tri[cls_tri_idx_epoch], self.train_ts_opn_tri[opn_tri_idx_epoch], self.train_ts_wedge[wedge_idx_epoch]))
                self.train_idx = np.concatenate((self.train_edge_idx_cls_tri[cls_tri_idx_epoch], self.train_edge_idx_opn_tri[opn_tri_idx_epoch], self.train_edge_idx_wedge[wedge_idx_epoch]))
            elif self.interpretation_type == 3:
                wedge_idx_epoch = np.random.choice(len(self.train_src_1_wedge), self.size, replace=False)
                nega_idx_epoch = np.random.choice(len(self.train_src_1_neg), self.size, replace=False)
                self.train_src_1 = np.concatenate((self.train_src_1_wedge[wedge_idx_epoch], self.train_src_1_neg[nega_idx_epoch]))
                self.train_src_2 = np.concatenate((self.train_src_2_wedge[wedge_idx_epoch], self.train_src_2_neg[nega_idx_epoch]))
                self.train_dst = np.concatenate((self.train_dst_wedge[wedge_idx_epoch], self.train_dst_neg[nega_idx_epoch]))
                self.train_ts = np.concatenate((self.train_ts_wedge[wedge_idx_epoch], self.train_ts_neg[nega_idx_epoch]))
                self.train_idx = np.concatenate((self.train_edge_idx_wedge[wedge_idx_epoch], self.train_edge_idx_neg[nega_idx_epoch]))
            elif self.interpretation_type == 4:
                cls_tri_idx_epoch = np.random.choice(len(self.train_src_1_cls_tri), self.size, replace=False)
                wedge_idx_epoch = np.random.choice(len(self.train_src_1_wedge), self.size, replace=False)
                self.train_src_1 = np.concatenate((self.train_src_1_cls_tri[cls_tri_idx_epoch], self.train_src_1_wedge[wedge_idx_epoch]))
                self.train_src_2 = np.concatenate((self.train_src_2_cls_tri[cls_tri_idx_epoch], self.train_src_2_wedge[wedge_idx_epoch]))
                self.train_dst = np.concatenate((self.train_dst_cls_tri[cls_tri_idx_epoch], self.train_dst_wedge[wedge_idx_epoch]))
                self.train_ts = np.concatenate((self.train_ts_cls_tri[cls_tri_idx_epoch], self.train_ts_wedge[wedge_idx_epoch]))
                self.train_idx = np.concatenate((self.train_edge_idx_cls_tri[cls_tri_idx_epoch], self.train_edge_idx_wedge[wedge_idx_epoch]))
        elif self.ablation_type == 1:
            opn_tri_idx_epoch = np.random.choice(len(self.train_src_1_opn_tri), self.size, replace=False)
            wedge_idx_epoch = np.random.choice(len(self.train_src_1_wedge), self.size, replace=False)
            self.train_src_1 = np.concatenate((self.train_src_1_opn_tri[opn_tri_idx_epoch], self.train_src_1_wedge[wedge_idx_epoch]))
            self.train_src_2 = np.concatenate((self.train_src_2_opn_tri[opn_tri_idx_epoch], self.train_src_2_wedge[wedge_idx_epoch]))
            self.train_dst = np.concatenate((self.train_dst_opn_tri[opn_tri_idx_epoch], self.train_dst_wedge[wedge_idx_epoch]))
            self.train_ts = np.concatenate((self.train_ts_opn_tri[opn_tri_idx_epoch], self.train_ts_wedge[wedge_idx_epoch]))
            self.train_idx = np.concatenate((self.train_edge_idx_opn_tri[opn_tri_idx_epoch], self.train_edge_idx_wedge[wedge_idx_epoch]))      
        elif self.time_prediction_type > 0:
            if self.time_prediction_type == 1:
                cls_tri_idx_epoch = np.random.choice(len(self.train_src_1_cls_tri), self.size, replace=False)
                self.train_src_1 = self.train_src_1_cls_tri[cls_tri_idx_epoch]
                self.train_src_2 = self.train_src_2_cls_tri[cls_tri_idx_epoch]
                self.train_dst = self.train_dst_cls_tri[cls_tri_idx_epoch]
                self.train_ts = self.train_ts_cls_tri[cls_tri_idx_epoch]
                self.train_idx = self.train_edge_idx_cls_tri[cls_tri_idx_epoch]
                self.train_time_gt = np.float32(self.train_endtime_cls_tri[cls_tri_idx_epoch] - self.train_ts_cls_tri[cls_tri_idx_epoch])
            elif self.time_prediction_type == 2:
                opn_tri_idx_epoch = np.random.choice(len(self.train_src_1_opn_tri), self.size, replace=False)
                self.train_src_1 = self.train_src_1_opn_tri[opn_tri_idx_epoch]
                self.train_src_2 = self.train_src_2_opn_tri[opn_tri_idx_epoch]
                self.train_dst = self.train_dst_opn_tri[opn_tri_idx_epoch]
                self.train_ts = self.train_ts_opn_tri[opn_tri_idx_epoch]
                self.train_idx = self.train_edge_idx_opn_tri[opn_tri_idx_epoch]
                self.train_time_gt = np.float32(self.train_endtime_opn_tri[opn_tri_idx_epoch] - self.train_ts_opn_tri[opn_tri_idx_epoch])
            elif self.time_prediction_type == 3:
                wedge_idx_epoch = np.random.choice(len(self.train_src_1_wedge), self.size, replace=False)
                self.train_src_1 = self.train_src_1_wedge[wedge_idx_epoch]
                self.train_src_2 = self.train_src_2_wedge[wedge_idx_epoch]
                self.train_dst = self.train_dst_wedge[wedge_idx_epoch]
                self.train_ts = self.train_ts_wedge[wedge_idx_epoch]
                self.train_idx = self.train_edge_idx_wedge[wedge_idx_epoch]
                self.train_time_gt = np.float32(self.train_endtime_wedge[wedge_idx_epoch] - self.train_ts_wedge[wedge_idx_epoch])
        else:
            cls_tri_idx_epoch = np.random.choice(len(self.train_src_1_cls_tri), self.size, replace=False)
            opn_tri_idx_epoch = np.random.choice(len(self.train_src_1_opn_tri), self.size, replace=False)
            wedge_idx_epoch = np.random.choice(len(self.train_src_1_wedge), self.size, replace=False)
            nega_idx_epoch = np.random.choice(len(self.train_src_1_neg), self.size, replace=False)
            self.train_src_1 = np.concatenate((self.train_src_1_cls_tri[cls_tri_idx_epoch], self.train_src_1_opn_tri[opn_tri_idx_epoch], self.train_src_1_wedge[wedge_idx_epoch], self.train_src_1_neg[nega_idx_epoch]))
            self.train_src_2 = np.concatenate((self.train_src_2_cls_tri[cls_tri_idx_epoch], self.train_src_2_opn_tri[opn_tri_idx_epoch], self.train_src_2_wedge[wedge_idx_epoch], self.train_src_2_neg[nega_idx_epoch]))
            self.train_dst = np.concatenate((self.train_dst_cls_tri[cls_tri_idx_epoch], self.train_dst_opn_tri[opn_tri_idx_epoch], self.train_dst_wedge[wedge_idx_epoch], self.train_dst_neg[nega_idx_epoch]))
            self.train_ts = np.concatenate((self.train_ts_cls_tri[cls_tri_idx_epoch], self.train_ts_opn_tri[opn_tri_idx_epoch], self.train_ts_wedge[wedge_idx_epoch], self.train_ts_neg[nega_idx_epoch]))
            self.train_idx = np.concatenate((self.train_edge_idx_cls_tri[cls_tri_idx_epoch], self.train_edge_idx_opn_tri[opn_tri_idx_epoch], self.train_edge_idx_wedge[wedge_idx_epoch], self.train_edge_idx_neg[nega_idx_epoch]))

        self.idx = 0
        np.random.shuffle(self.train_idx_list)

    def initialize_val(self):
        if self.interpretation_type > 0:
            if self.interpretation_type == 1:
                cls_tri_idx_epoch = np.random.choice(len(self.val_src_1_cls_tri), self.size_val, replace=False)
                opn_tri_idx_epoch = np.random.choice(len(self.val_src_1_opn_tri), self.size_val, replace=False)
                
                self.val_src_1 = np.concatenate((self.val_src_1_cls_tri[cls_tri_idx_epoch], self.val_src_1_opn_tri[opn_tri_idx_epoch]))
                self.val_src_2 = np.concatenate((self.val_src_2_cls_tri[cls_tri_idx_epoch], self.val_src_2_opn_tri[opn_tri_idx_epoch]))
                self.val_dst = np.concatenate((self.val_dst_cls_tri[cls_tri_idx_epoch], self.val_dst_opn_tri[opn_tri_idx_epoch]))
                self.val_ts = np.concatenate((self.val_ts_cls_tri[cls_tri_idx_epoch], self.val_ts_opn_tri[opn_tri_idx_epoch]))
                self.val_idx = np.concatenate((self.val_edge_idx_cls_tri[cls_tri_idx_epoch], self.val_edge_idx_opn_tri[opn_tri_idx_epoch]))
            elif self.interpretation_type == 2:
                cls_tri_idx_epoch = np.random.choice(len(self.val_src_1_cls_tri), int(self.size_val / 2), replace=False)
                opn_tri_idx_epoch = np.random.choice(len(self.val_src_1_opn_tri), self.size_val - int(self.size_val / 2), replace=False)
                wedge_idx_epoch = np.random.choice(len(self.val_src_1_wedge), self.size_val, replace=False)

                self.val_src_1 = np.concatenate((self.val_src_1_cls_tri[cls_tri_idx_epoch], self.val_src_1_opn_tri[opn_tri_idx_epoch], self.val_src_1_wedge[wedge_idx_epoch]))
                self.val_src_2 = np.concatenate((self.val_src_2_cls_tri[cls_tri_idx_epoch], self.val_src_2_opn_tri[opn_tri_idx_epoch], self.val_src_2_wedge[wedge_idx_epoch]))
                self.val_dst = np.concatenate((self.val_dst_cls_tri[cls_tri_idx_epoch], self.val_dst_opn_tri[opn_tri_idx_epoch], self.val_dst_wedge[wedge_idx_epoch]))
                self.val_ts = np.concatenate((self.val_ts_cls_tri[cls_tri_idx_epoch], self.val_ts_opn_tri[opn_tri_idx_epoch], self.val_ts_wedge[wedge_idx_epoch]))
                self.val_idx = np.concatenate((self.val_edge_idx_cls_tri[cls_tri_idx_epoch], self.val_edge_idx_opn_tri[opn_tri_idx_epoch], self.val_edge_idx_wedge[wedge_idx_epoch]))
            elif self.interpretation_type == 3:
                wedge_idx_epoch = np.random.choice(len(self.val_src_1_wedge), self.size_val, replace=False)
                nega_idx_epoch = np.random.choice(len(self.val_src_1_neg), self.size_val, replace=False)

                self.val_src_1 = np.concatenate((self.val_src_1_wedge[wedge_idx_epoch], self.val_src_1_neg[nega_idx_epoch]))
                self.val_src_2 = np.concatenate((self.val_src_2_wedge[wedge_idx_epoch], self.val_src_2_neg[nega_idx_epoch]))
                self.val_dst = np.concatenate((self.val_dst_wedge[wedge_idx_epoch], self.val_dst_neg[nega_idx_epoch]))
                self.val_ts = np.concatenate((self.val_ts_wedge[wedge_idx_epoch], self.val_ts_neg[nega_idx_epoch]))
                self.val_idx = np.concatenate((self.val_edge_idx_wedge[wedge_idx_epoch], self.val_edge_idx_neg[nega_idx_epoch]))
            elif self.interpretation_type == 4:
                cls_tri_idx_epoch = np.random.choice(len(self.val_src_1_cls_tri), self.size_val, replace=False)
                wedge_idx_epoch = np.random.choice(len(self.val_src_1_wedge), self.size_val, replace=False)

                self.val_src_1 = np.concatenate((self.val_src_1_cls_tri[cls_tri_idx_epoch], self.val_src_1_wedge[wedge_idx_epoch]))
                self.val_src_2 = np.concatenate((self.val_src_2_cls_tri[cls_tri_idx_epoch], self.val_src_2_wedge[wedge_idx_epoch]))
                self.val_dst = np.concatenate((self.val_dst_cls_tri[cls_tri_idx_epoch], self.val_dst_wedge[wedge_idx_epoch]))
                self.val_ts = np.concatenate((self.val_ts_cls_tri[cls_tri_idx_epoch], self.val_ts_wedge[wedge_idx_epoch]))
                self.val_idx = np.concatenate((self.val_edge_idx_cls_tri[cls_tri_idx_epoch], self.val_edge_idx_wedge[wedge_idx_epoch]))
        elif self.ablation_type == 1: #abla
            opn_tri_idx_epoch = np.random.choice(len(self.val_src_1_opn_tri), self.size_val, replace=False)
            wedge_idx_epoch = np.random.choice(len(self.val_src_1_wedge), self.size_val, replace=False)

            self.val_src_1 = np.concatenate((self.val_src_1_opn_tri[opn_tri_idx_epoch], self.val_src_1_wedge[wedge_idx_epoch]))
            self.val_src_2 = np.concatenate((self.val_src_2_opn_tri[opn_tri_idx_epoch], self.val_src_2_wedge[wedge_idx_epoch]))
            self.val_dst = np.concatenate((self.val_dst_opn_tri[opn_tri_idx_epoch], self.val_dst_wedge[wedge_idx_epoch]))
            self.val_ts = np.concatenate((self.val_ts_opn_tri[opn_tri_idx_epoch], self.val_ts_wedge[wedge_idx_epoch]))
            self.val_idx = np.concatenate((self.val_edge_idx_opn_tri[opn_tri_idx_epoch], self.val_edge_idx_wedge[wedge_idx_epoch]))
        elif self.time_prediction_type > 0:
            if self.time_prediction_type == 1:
                cls_tri_idx_epoch = np.random.choice(len(self.val_src_1_cls_tri), self.size_val, replace=False)
                self.val_src_1 = self.val_src_1_cls_tri[cls_tri_idx_epoch]
                self.val_src_2 = self.val_src_2_cls_tri[cls_tri_idx_epoch]
                self.val_dst = self.val_dst_cls_tri[cls_tri_idx_epoch]
                self.val_ts = self.val_ts_cls_tri[cls_tri_idx_epoch]
                self.val_idx = self.val_edge_idx_cls_tri[cls_tri_idx_epoch]
                self.val_time_gt = np.float32(self.val_endtime_cls_tri[cls_tri_idx_epoch] - self.val_ts_cls_tri[cls_tri_idx_epoch])
            elif self.time_prediction_type == 2:
                opn_tri_idx_epoch = np.random.choice(len(self.val_src_1_opn_tri), self.size_val, replace=False)
                self.val_src_1 = self.val_src_1_opn_tri[opn_tri_idx_epoch]
                self.val_src_2 = self.val_src_2_opn_tri[opn_tri_idx_epoch]
                self.val_dst = self.val_dst_opn_tri[opn_tri_idx_epoch]
                self.val_ts = self.val_ts_opn_tri[opn_tri_idx_epoch]
                self.val_idx = self.val_edge_idx_opn_tri[opn_tri_idx_epoch]
                self.val_time_gt = np.float32(self.val_endtime_opn_tri[opn_tri_idx_epoch] - self.val_ts_opn_tri[opn_tri_idx_epoch])
            elif self.time_prediction_type == 3:
                wedge_idx_epoch = np.random.choice(len(self.val_src_1_wedge), self.size_val, replace=False)
                self.val_src_1 = self.val_src_1_wedge[wedge_idx_epoch]
                self.val_src_2 = self.val_src_2_wedge[wedge_idx_epoch]
                self.val_dst = self.val_dst_wedge[wedge_idx_epoch]
                self.val_ts = self.val_ts_wedge[wedge_idx_epoch]
                self.val_idx = self.val_edge_idx_wedge[wedge_idx_epoch]
                self.val_time_gt = np.float32(self.val_endtime_wedge[wedge_idx_epoch] - self.val_ts_wedge[wedge_idx_epoch])
        else:
            cls_tri_idx_epoch = np.random.choice(len(self.val_src_1_cls_tri), self.size_val, replace=False)
            opn_tri_idx_epoch = np.random.choice(len(self.val_src_1_opn_tri), self.size_val, replace=False)
            wedge_idx_epoch = np.random.choice(len(self.val_src_1_wedge), self.size_val, replace=False)
            nega_idx_epoch = np.random.choice(len(self.val_src_1_neg), self.size_val, replace=False)

            self.val_src_1 = np.concatenate((self.val_src_1_cls_tri[cls_tri_idx_epoch], self.val_src_1_opn_tri[opn_tri_idx_epoch], self.val_src_1_wedge[wedge_idx_epoch], self.val_src_1_neg[nega_idx_epoch]))
            self.val_src_2 = np.concatenate((self.val_src_2_cls_tri[cls_tri_idx_epoch], self.val_src_2_opn_tri[opn_tri_idx_epoch], self.val_src_2_wedge[wedge_idx_epoch], self.val_src_2_neg[nega_idx_epoch]))
            self.val_dst = np.concatenate((self.val_dst_cls_tri[cls_tri_idx_epoch], self.val_dst_opn_tri[opn_tri_idx_epoch], self.val_dst_wedge[wedge_idx_epoch], self.val_dst_neg[nega_idx_epoch]))
            self.val_ts = np.concatenate((self.val_ts_cls_tri[cls_tri_idx_epoch], self.val_ts_opn_tri[opn_tri_idx_epoch], self.val_ts_wedge[wedge_idx_epoch], self.val_ts_neg[nega_idx_epoch]))
            self.val_idx = np.concatenate((self.val_edge_idx_cls_tri[cls_tri_idx_epoch], self.val_edge_idx_opn_tri[opn_tri_idx_epoch], self.val_edge_idx_wedge[wedge_idx_epoch], self.val_edge_idx_neg[nega_idx_epoch]))

        self.idx = 0
        np.random.shuffle(self.val_idx_list)
    
    def initialize_test(self):
        if self.interpretation_type > 0:
            if self.interpretation_type == 1:
                cls_tri_idx_epoch = np.random.choice(len(self.test_src_1_cls_tri), self.size_test, replace=False)
                opn_tri_idx_epoch = np.random.choice(len(self.test_src_1_opn_tri), self.size_test, replace=False)
                
                self.test_src_1 = np.concatenate((self.test_src_1_cls_tri[cls_tri_idx_epoch], self.test_src_1_opn_tri[opn_tri_idx_epoch]))
                self.test_src_2 = np.concatenate((self.test_src_2_cls_tri[cls_tri_idx_epoch], self.test_src_2_opn_tri[opn_tri_idx_epoch]))
                self.test_dst = np.concatenate((self.test_dst_cls_tri[cls_tri_idx_epoch], self.test_dst_opn_tri[opn_tri_idx_epoch]))
                self.test_ts = np.concatenate((self.test_ts_cls_tri[cls_tri_idx_epoch], self.test_ts_opn_tri[opn_tri_idx_epoch]))
                self.test_idx = np.concatenate((self.test_edge_idx_cls_tri[cls_tri_idx_epoch], self.test_edge_idx_opn_tri[opn_tri_idx_epoch]))
            elif self.interpretation_type == 2:
                cls_tri_idx_epoch = np.random.choice(len(self.test_src_1_cls_tri), int(self.size_test / 2), replace=False)
                opn_tri_idx_epoch = np.random.choice(len(self.test_src_1_opn_tri), self.size_test - int(self.size_test / 2), replace=False)
                wedge_idx_epoch = np.random.choice(len(self.test_src_1_wedge), self.size_test, replace=False)

                self.test_src_1 = np.concatenate((self.test_src_1_cls_tri[cls_tri_idx_epoch], self.test_src_1_opn_tri[opn_tri_idx_epoch], self.test_src_1_wedge[wedge_idx_epoch]))
                self.test_src_2 = np.concatenate((self.test_src_2_cls_tri[cls_tri_idx_epoch], self.test_src_2_opn_tri[opn_tri_idx_epoch], self.test_src_2_wedge[wedge_idx_epoch]))
                self.test_dst = np.concatenate((self.test_dst_cls_tri[cls_tri_idx_epoch], self.test_dst_opn_tri[opn_tri_idx_epoch], self.test_dst_wedge[wedge_idx_epoch]))
                self.test_ts = np.concatenate((self.test_ts_cls_tri[cls_tri_idx_epoch], self.test_ts_opn_tri[opn_tri_idx_epoch], self.test_ts_wedge[wedge_idx_epoch]))
                self.test_idx = np.concatenate((self.test_edge_idx_cls_tri[cls_tri_idx_epoch], self.test_edge_idx_opn_tri[opn_tri_idx_epoch], self.test_edge_idx_wedge[wedge_idx_epoch]))
            elif self.interpretation_type == 3:
                wedge_idx_epoch = np.random.choice(len(self.test_src_1_wedge), self.size_test, replace=False)
                nega_idx_epoch = np.random.choice(len(self.test_src_1_neg), self.size_test, replace=False)

                self.test_src_1 = np.concatenate((self.test_src_1_wedge[wedge_idx_epoch], self.test_src_1_neg[nega_idx_epoch]))
                self.test_src_2 = np.concatenate((self.test_src_2_wedge[wedge_idx_epoch], self.test_src_2_neg[nega_idx_epoch]))
                self.test_dst = np.concatenate((self.test_dst_wedge[wedge_idx_epoch], self.test_dst_neg[nega_idx_epoch]))
                self.test_ts = np.concatenate((self.test_ts_wedge[wedge_idx_epoch], self.test_ts_neg[nega_idx_epoch]))
                self.test_idx = np.concatenate((self.test_edge_idx_wedge[wedge_idx_epoch], self.test_edge_idx_neg[nega_idx_epoch]))
            elif self.interpretation_type == 4:
                cls_tri_idx_epoch = np.random.choice(len(self.test_src_1_cls_tri), self.size_test, replace=False)
                wedge_idx_epoch = np.random.choice(len(self.test_src_1_wedge), self.size_test, replace=False)

                self.test_src_1 = np.concatenate((self.test_src_1_cls_tri[cls_tri_idx_epoch], self.test_src_1_wedge[wedge_idx_epoch]))
                self.test_src_2 = np.concatenate((self.test_src_2_cls_tri[cls_tri_idx_epoch], self.test_src_2_wedge[wedge_idx_epoch]))
                self.test_dst = np.concatenate((self.test_dst_cls_tri[cls_tri_idx_epoch], self.test_dst_wedge[wedge_idx_epoch]))
                self.test_ts = np.concatenate((self.test_ts_cls_tri[cls_tri_idx_epoch], self.test_ts_wedge[wedge_idx_epoch]))
                self.test_idx = np.concatenate((self.test_edge_idx_cls_tri[cls_tri_idx_epoch], self.test_edge_idx_wedge[wedge_idx_epoch]))
        elif self.ablation_type == 1:
            opn_tri_idx_epoch = np.random.choice(len(self.test_src_1_opn_tri), self.size_test, replace=False)
            wedge_idx_epoch = np.random.choice(len(self.test_src_1_wedge), self.size_test, replace=False)

            self.test_src_1 = np.concatenate((self.test_src_1_opn_tri[opn_tri_idx_epoch], self.test_src_1_wedge[wedge_idx_epoch]))
            self.test_src_2 = np.concatenate((self.test_src_2_opn_tri[opn_tri_idx_epoch], self.test_src_2_wedge[wedge_idx_epoch]))
            self.test_dst = np.concatenate((self.test_dst_opn_tri[opn_tri_idx_epoch], self.test_dst_wedge[wedge_idx_epoch]))
            self.test_ts = np.concatenate((self.test_ts_opn_tri[opn_tri_idx_epoch], self.test_ts_wedge[wedge_idx_epoch]))
            self.test_idx = np.concatenate((self.test_edge_idx_opn_tri[opn_tri_idx_epoch], self.test_edge_idx_wedge[wedge_idx_epoch]))
        elif self.time_prediction_type > 0:
            if self.time_prediction_type == 1:
                cls_tri_idx_epoch = np.random.choice(len(self.test_src_1_cls_tri), self.size_test, replace=False)
                self.test_src_1 = self.test_src_1_cls_tri[cls_tri_idx_epoch]
                self.test_src_2 = self.test_src_2_cls_tri[cls_tri_idx_epoch]
                self.test_dst = self.test_dst_cls_tri[cls_tri_idx_epoch]
                self.test_ts = self.test_ts_cls_tri[cls_tri_idx_epoch]
                self.test_idx = self.test_edge_idx_cls_tri[cls_tri_idx_epoch]
                self.test_time_gt = np.float32(self.test_endtime_cls_tri[cls_tri_idx_epoch] - self.test_ts_cls_tri[cls_tri_idx_epoch])
            elif self.time_prediction_type == 2:
                opn_tri_idx_epoch = np.random.choice(len(self.test_src_1_opn_tri), self.size_test, replace=False)
                self.test_src_1 = self.test_src_1_opn_tri[opn_tri_idx_epoch]
                self.test_src_2 = self.test_src_2_opn_tri[opn_tri_idx_epoch]
                self.test_dst = self.test_dst_opn_tri[opn_tri_idx_epoch]
                self.test_ts = self.test_ts_opn_tri[opn_tri_idx_epoch]
                self.test_idx = self.test_edge_idx_opn_tri[opn_tri_idx_epoch]
                self.test_time_gt = np.float32(self.test_endtime_opn_tri[opn_tri_idx_epoch] - self.test_ts_opn_tri[opn_tri_idx_epoch])
            elif self.time_prediction_type == 3:
                wedge_idx_epoch = np.random.choice(len(self.test_src_1_wedge), self.size_test, replace=False)
                self.test_src_1 = self.test_src_1_wedge[wedge_idx_epoch]
                self.test_src_2 = self.test_src_2_wedge[wedge_idx_epoch]
                self.test_dst = self.test_dst_wedge[wedge_idx_epoch]
                self.test_ts = self.test_ts_wedge[wedge_idx_epoch]
                self.test_idx = self.test_edge_idx_wedge[wedge_idx_epoch]
                self.test_time_gt = np.float32(self.test_endtime_wedge[wedge_idx_epoch] - self.test_ts_wedge[wedge_idx_epoch])
        else:
            cls_tri_idx_epoch = np.random.choice(len(self.test_src_1_cls_tri), self.size_test, replace=False)
            opn_tri_idx_epoch = np.random.choice(len(self.test_src_1_opn_tri), self.size_test, replace=False)
            wedge_idx_epoch = np.random.choice(len(self.test_src_1_wedge), self.size_test, replace=False)
            nega_idx_epoch = np.random.choice(len(self.test_src_1_neg), self.size_test, replace=False)

            self.test_src_1 = np.concatenate((self.test_src_1_cls_tri[cls_tri_idx_epoch], self.test_src_1_opn_tri[opn_tri_idx_epoch], self.test_src_1_wedge[wedge_idx_epoch], self.test_src_1_neg[nega_idx_epoch]))
            self.test_src_2 = np.concatenate((self.test_src_2_cls_tri[cls_tri_idx_epoch], self.test_src_2_opn_tri[opn_tri_idx_epoch], self.test_src_2_wedge[wedge_idx_epoch], self.test_src_2_neg[nega_idx_epoch]))
            self.test_dst = np.concatenate((self.test_dst_cls_tri[cls_tri_idx_epoch], self.test_dst_opn_tri[opn_tri_idx_epoch], self.test_dst_wedge[wedge_idx_epoch], self.test_dst_neg[nega_idx_epoch]))
            self.test_ts = np.concatenate((self.test_ts_cls_tri[cls_tri_idx_epoch], self.test_ts_opn_tri[opn_tri_idx_epoch], self.test_ts_wedge[wedge_idx_epoch], self.test_ts_neg[nega_idx_epoch]))
            self.test_idx = np.concatenate((self.test_edge_idx_cls_tri[cls_tri_idx_epoch], self.test_edge_idx_opn_tri[opn_tri_idx_epoch], self.test_edge_idx_wedge[wedge_idx_epoch], self.test_edge_idx_neg[nega_idx_epoch]))

        self.idx = 0
        np.random.shuffle(self.test_idx_list)

    def get_size(self):
        return self.num_class * self.size
    
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
            print("train error")
        batch_idx = self.train_idx_list[s_idx:e_idx]
        src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut = self.train_src_1[batch_idx], self.train_src_2[batch_idx], self.train_dst[batch_idx], self.train_ts[batch_idx], self.train_idx[batch_idx]
        if self.time_prediction_type > 0:
            label_cut = self.train_time_gt[batch_idx]
        else:
            label_cut = self.train_label_t[batch_idx]
        self.idx += 1
        return src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut, label_cut
    
    def train_samples_baselines(self):
        s_idx = self.idx * self.bs
        e_idx = min(self.get_size(), s_idx + self.bs)
        if s_idx == e_idx:
            s_idx = 0
            e_idx = self.bs
            self.idx = 0
            print("train error")
        batch_idx = self.train_idx_list[s_idx:e_idx]
        if self.time_prediction_type > 0:
            label_cut = self.train_time_gt[batch_idx]
        else:
            label_cut = self.train_label_t[batch_idx]
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
            print("val error")
        
        batch_idx = self.val_idx_list[s_idx:e_idx]
        src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut = self.val_src_1[batch_idx], self.val_src_2[batch_idx], self.val_dst[batch_idx], self.val_ts[batch_idx], self.val_idx[batch_idx]
        if self.time_prediction_type > 0:
            label_cut = self.val_time_gt[batch_idx]
        else:
            label_cut = self.val_label[batch_idx]
        self.idx += 1
        return src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut, label_cut
    
    def val_samples_baselines(self, bs = None):
        if bs == None:
            bs = self.bs
        s_idx = self.idx * bs
        e_idx = min(self.get_val_size(), s_idx + bs)
        if s_idx == e_idx:
            s_idx = 0
            e_idx = bs
            self.idx = 0
            print("val error")
        
        batch_idx = self.val_idx_list[s_idx:e_idx]
        e_l_cut = self.val_idx[batch_idx]
        if self.time_prediction_type > 0:
            label_cut = self.val_time_gt[batch_idx]
        else:
            label_cut = self.val_label[batch_idx]
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
        src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut = self.test_src_1[batch_idx], self.test_src_2[batch_idx], self.test_dst[batch_idx], self.test_ts[batch_idx], self.test_idx[batch_idx]
        if self.time_prediction_type > 0:
            label_cut = self.test_time_gt[batch_idx]
        else:
            label_cut = self.test_label[batch_idx]
        self.idx += 1
        return src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut, label_cut
    
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
            label_cut = self.test_label[batch_idx]
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

class NegTripletSampler(object):
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
