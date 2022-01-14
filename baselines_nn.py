import pandas as pd
from log import set_up_logger
from parser import *
# from eval import *
from utils import *
import utils
# from train import *
import os
from module import HIT

from module import finalClassifier
from module import finalClassifier_time_prediction
from module import finalClassifier_inter

from graph import NeighborFinder
import resource
from sklearn.preprocessing import scale
from histogram import plot_hist
import module

import torch
import numpy as np
from tqdm import tqdm
import math
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

def eval_one_epoch(hint, model, dataset, val_flag='val', interpretation=False, time_prediction=False, src_1_emb_cut=None, src_2_emb_cut=None, dst_emb_cut=None, device=None):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    y_true, y_pred, y_score, y_one_hot_np = None, None, None, None
    dataset.reset()
    # model.test = True
    # device = model.n_feat_th.data.device
    if interpretation:
        roc_auc_score = utils.roc_auc_score_single
    else:
        roc_auc_score = utils.roc_auc_score_multi
    if val_flag == 'train':
        num_test_instance = dataset.get_size()
        get_sample = dataset.train_samples_baselines
        # dataset.initialize()
    elif val_flag == 'val':
        num_test_instance = dataset.get_val_size()
        get_sample = dataset.val_samples_baselines
        # dataset.initialize_val()
    elif val_flag == 'test':
        num_test_instance = dataset.get_test_size()
        get_sample = dataset.test_samples_baselines
        # dataset.initialize_test()
        # print("mintime", min(dataset.test_time_gt))
    with torch.no_grad():
        model = model.eval()
        TEST_BATCH_SIZE = dataset.bs
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        num_class = dataset.num_class
        walk_pattern = None
        walk_score = None
        walk_pattern_total = None
        walk_pattern_label_total = None
        walk_score_total = None
        NLL_total = None
        MSE_total = None
        MAE_total = None
        time_predicted_total = None
        time_gt_total = None
        loop_num = num_test_batch
        for k in tqdm(range(loop_num)):
            batch_idx, true_label = get_sample()
            src_1_l_cut = src_1_emb_cut[batch_idx]
            src_2_l_cut = src_2_emb_cut[batch_idx]
            dst_l_cut = dst_emb_cut[batch_idx]
            if time_prediction:
                true_label_torch = torch.from_numpy(true_label).to(device)
                # _NLL_score, _ = model.contrast(src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut, endtime_pos=true_label_torch)
                # print(true_label_torch)
                _NLL_score = model(src_1_l_cut, src_2_l_cut, dst_l_cut, true_label_torch)
                ave_mae_t, ave_log_t, NLL_score, time_list = _NLL_score
                # we compare the log t distribution
                time_predicted = time_list[0].detach().cpu().numpy()
                time_gt = time_list[1].detach().cpu().numpy()
                if time_predicted_total is None:
                    time_predicted_total = time_predicted
                    time_gt_total = time_gt
                else:
                    time_predicted_total = np.concatenate([time_predicted_total, time_predicted])
                    time_gt_total = np.concatenate([time_gt_total, time_gt])
            else:
                # pred_score, _ = model.contrast(src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut)
                pred_score = model(src_1_l_cut, src_2_l_cut, dst_l_cut)
            if time_prediction:
                if NLL_total is None:
                    NLL_total = NLL_score
                    MSE_total = ave_log_t
                    MAE_total = ave_mae_t
                else:
                    NLL_total += NLL_score
                    MSE_total += ave_log_t
                    MAE_total += ave_mae_t
            else:
                pred_label = torch.argmax(pred_score, dim=1).cpu().detach().numpy()
                pred_score = torch.nn.functional.softmax(pred_score, dim=1).cpu().numpy()
                y_one_hot = torch.nn.functional.one_hot(torch.from_numpy(true_label).long(), num_classes=num_class).float().cpu().numpy()
                if y_pred is None:
                    y_pred = np.copy(pred_label)
                    y_true = np.copy(true_label)
                    y_score = np.copy(pred_score)
                    y_one_hot_np = y_one_hot
                else:
                    y_pred = np.concatenate((y_pred, pred_label))
                    y_true = np.concatenate((y_true, true_label))
                    y_score = np.concatenate((y_score, pred_score))
                    y_one_hot_np = np.concatenate((y_one_hot_np, y_one_hot))
                
                
                val_acc.append((pred_label == true_label).mean())
                val_ap.append(1)

    logger.info(val_flag)

    if time_prediction:
        logger.info("NLL Loss  " + str(NLL_total / num_test_instance))
        logger.info("MSE Loss  " + str(MSE_total / num_test_instance))
        logger.info("MAE Loss  " + str(MAE_total / num_test_instance))
        # print("min time_predicted_total", min(time_predicted_total))
        # print("min time_gt_total", min(time_gt_total))
        return NLL_total, num_test_instance, time_predicted_total, time_gt_total
    else:
        cm = confusion_matrix(y_true, y_pred)
        print('Confusion Matrix')
        print(cm)
        if (interpretation) and (val_flag == 'test'):
            _, _, result = process_pattern(walk_pattern_total, walk_score_total, pattern_dict=model.position_encoder.pattern, non_idx=model.num_layers*2, pattern_label=walk_pattern_label_total)
            print('result')
            print(result)
            print('walk pattern')
            print(model.position_encoder.pattern)
            print('walk pattern number:', len(np.unique(walk_pattern_total)))
        val_auc = roc_auc_score(y_one_hot_np, y_score)

        return np.mean(val_acc), np.mean(val_ap), None, val_auc, cm
        

def train_val(dataset, model, mode, bs, epochs, criterion, optimizer, early_stopper, logger, interpretation=False, time_prediction=False, device=None):
    # partial_ngh_finder, full_ngh_finder = ngh_finders
    # device = model.n_feat_th.data.device
    num_instance = dataset.get_size()
    num_batch = math.ceil(num_instance / bs)
    dataset.set_batch_size(bs)
    
    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    # model.test = False
    

    if interpretation:
        roc_auc_score = utils.roc_auc_score_single
    else:
        roc_auc_score = utils.roc_auc_score_multi

    for epoch in range(epochs):
        # model.update_ngh_finder(partial_ngh_finder)
        dataset.initialize()
        dataset.reset()
        print(len(src_l))
        src_1_emb_cut, src_2_emb_cut, dst_emb_cut = dealData(src_l, dst_l, ts_l, dataset.train_src_1, dataset.train_src_2, dataset.train_dst, dataset.train_ts)
        src_1_emb_cut, src_2_emb_cut, dst_emb_cut = torch.from_numpy(src_1_emb_cut).to(device).float(), torch.from_numpy(src_2_emb_cut).to(device).float(), torch.from_numpy(dst_emb_cut).to(device).float()

        acc, ap, f1, auc, m_loss = [], [], [], [], []
        logger.info('start {} epoch'.format(epoch))
        NLL_total = None; MSE_total = None; MAE_total = None
        y_true, y_pred, y_one_hot_np = None, None, None

        for k in tqdm(range(int(num_batch))):
        # for k in tqdm(range(int(1))):
            batch_idx, true_label = dataset.train_samples_baselines()
            src_1_l_cut = src_1_emb_cut[batch_idx]
            src_2_l_cut = src_2_emb_cut[batch_idx]
            dst_l_cut =  dst_emb_cut[batch_idx]
            
            model.train()
            optimizer.zero_grad()
            if time_prediction:
                true_label_torch = torch.from_numpy(true_label).to(device)
                # _pred_score, _ = model.contrast(src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut, endtime_pos=true_label_torch)   # the core training code
                _pred_score = model(src_1_l_cut, src_2_l_cut, dst_l_cut, true_label_torch)
                # print("_pred_score", _pred_score, "true_label_torch", true_label_torch)
                ave_mae_t, ave_log_t, pred_score, _ = _pred_score
            else:
                true_label_torch = torch.from_numpy(true_label).long().to(device)
                # pred_score, _ = model.contrast(src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut)   # the core training code
                pred_score = model(src_1_l_cut, src_2_l_cut, dst_l_cut)            

            if time_prediction:
                loss = pred_score
            else:
                loss = criterion(pred_score, true_label_torch)
            
            # print("loss", loss)
            loss.backward()
            optimizer.step()
            
            # collect training results
            with torch.no_grad():
                model.eval()
                if time_prediction:
                    if NLL_total is None:
                        NLL_total = pred_score
                        MSE_total = ave_log_t
                        MAE_total = ave_mae_t
                    else:
                        NLL_total += pred_score
                        MSE_total += ave_log_t
                        MAE_total += ave_mae_t
                else:
                    pred_label = torch.argmax(pred_score, dim=1).cpu().detach().numpy()
                    acc.append((pred_label == true_label).mean())
                    ap.append(1)
                    m_loss.append(loss.item())
                    y_one_hot = torch.nn.functional.one_hot(torch.from_numpy(true_label).long(), num_classes=num_class).float().cpu().numpy()
            
                    if y_pred is None:
                        y_pred = np.copy(pred_label)
                        y_true = np.copy(true_label)
                        y_one_hot_np = y_one_hot
                        pred_score_np = torch.nn.functional.softmax(pred_score, dim=1).cpu().numpy()
                    else:
                        y_pred = np.concatenate((y_pred, pred_label))
                        y_true = np.concatenate((y_true, true_label))
                        y_one_hot_np = np.concatenate((y_one_hot_np, y_one_hot))
                        pred_score_np = np.concatenate((pred_score_np, torch.nn.functional.softmax(pred_score, dim=1).cpu().numpy()))
        if time_prediction:
            logger.info("train")
            logger.info("NLL " + str(NLL_total/dataset.get_size()))
            logger.info("MSE " + str(MSE_total/dataset.get_size()))
            logger.info("MAE " + str(MAE_total/dataset.get_size()))
        else:
            print("train")
            cm = confusion_matrix(y_true, y_pred)
            print(cm)
            logger.info('confusion matrix: ')
            logger.info(', '.join(str(r) for r in cm.reshape(1,-1)))

            acc = np.mean(acc)
            auc = roc_auc_score(y_one_hot_np, pred_score_np)

        dataset.initialize_val()
        src_1_emb_cut, src_2_emb_cut, dst_emb_cut = dealData(src_l, dst_l, ts_l, dataset.val_src_1, dataset.val_src_2, dataset.val_dst, dataset.val_ts)
        src_1_emb_cut, src_2_emb_cut, dst_emb_cut = torch.from_numpy(src_1_emb_cut).to(device).float(), torch.from_numpy(src_2_emb_cut).to(device).float(), torch.from_numpy(dst_emb_cut).to(device).float()
        
        if time_prediction:
            # print(len(src_l))
            NLL_loss, num, time_predicted_total, time_gt_total = eval_one_epoch('val for {} nodes'.format(mode), model, dataset, val_flag='val',interpretation=interpretation, time_prediction=time_prediction, src_1_emb_cut=src_1_emb_cut, src_2_emb_cut=src_2_emb_cut, dst_emb_cut=dst_emb_cut, device=device)
            logger.info('val NLL: {}  Number: {}'.format(NLL_loss, num))
            val_auc = -NLL_loss.cpu().numpy()
        else:
            val_acc, val_ap, val_f1, val_auc, cm = eval_one_epoch('val for {} nodes'.format(mode), model, dataset, val_flag='val',interpretation=interpretation, time_prediction=time_prediction, src_1_emb_cut=src_1_emb_cut, src_2_emb_cut=src_2_emb_cut, dst_emb_cut=dst_emb_cut, device=device)
            logger.info('confusion matrix: ')
            logger.info(', '.join(str(r) for r in cm.reshape(1,-1)))
        
        # model.update_ngh_finder(full_ngh_finder)

        dataset.initialize_test()
        src_1_emb_cut, src_2_emb_cut, dst_emb_cut = dealData(src_l, dst_l, ts_l, dataset.test_src_1, dataset.test_src_2, dataset.test_dst, dataset.test_ts)
        src_1_emb_cut, src_2_emb_cut, dst_emb_cut = torch.from_numpy(src_1_emb_cut).to(device).float(), torch.from_numpy(src_2_emb_cut).to(device).float(), torch.from_numpy(dst_emb_cut).to(device).float()
        
        if time_prediction:
            test_NLL, num, time_predicted_total, time_gt_total = eval_one_epoch('test for {} nodes'.format(mode), model, dataset, val_flag='test',interpretation=interpretation, time_prediction=time_prediction, src_1_emb_cut=src_1_emb_cut, src_2_emb_cut=src_2_emb_cut, dst_emb_cut=dst_emb_cut, device=device)
            time_predicted_total = np.exp(time_predicted_total)
            time_gt_total = np.exp(time_gt_total)
            # file_addr = './Histogram/'+dataset.DATA+'-'+str(dataset.time_prediction_type)+'/'
            # if not os.path.exists(file_addr):
            #     os.makedirs(file_addr)
            
            # with open(file_addr+'time_prediction_histogram'+str(epoch), 'wb') as f:
            #     np.save(f, np.array([time_predicted_total, time_gt_total]))
            # histogram.plot_hist_multi([time_predicted_total, time_gt_total], bins=50, figure_title='Time Prediction Histogram'+str(epoch), file_addr=file_addr, label=['Ours', 'Groundtruth'])
            
            logger.info('test NLL: {}'.format(test_NLL))
        else:
            val_acc_t, val_ap_t, val_f1_t, val_auc_t, cm = eval_one_epoch('val for {} nodes'.format(mode), model, dataset, val_flag='test',interpretation=interpretation, time_prediction=time_prediction, src_1_emb_cut=src_1_emb_cut, src_2_emb_cut=src_2_emb_cut, dst_emb_cut=dst_emb_cut, device=device)
            logger.info('confusion matrix: ')
            logger.info(', '.join(str(r) for r in cm.reshape(1,-1)))
            logger.info('epoch: {}:'.format(epoch))
            logger.info('epoch mean loss: {}'.format(np.mean(m_loss)))
            logger.info('train acc: {}, val acc: {}, test acc: {}'.format(np.mean(acc), val_acc, val_acc_t))
            logger.info('train auc: {}, val auc: {}, test auc: {}'.format(np.mean(auc), val_auc, val_auc_t))
            logger.info('train ap: {}, val ap: {}, test ap: {}'.format(np.mean(ap), val_ap, val_ap_t))

        # # early stop check and checkpoint saving
        if early_stopper.early_stop_check(val_auc):
            logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            # best_checkpoint_path = model.get_checkpoint_path(early_stopper.best_epoch)
            best_checkpoint_path = get_checkpoint_path(early_stopper.best_epoch)
            model.load_state_dict(torch.load(best_checkpoint_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            model.eval()
            break
        else:
            # pass
            torch.save(model.state_dict(), get_checkpoint_path(epoch))




# utilize

def dealData(src_l, dst_l, ts_l, src_1, src_2, dst, ts_cut):

    nodeTimeDict = {}
    nodeTimePairDict = {}

    src_1_emb = np.zeros((len(src_1), embed_size))
    src_2_emb = np.zeros((len(src_1), embed_size))
    dst_emb = np.zeros((len(src_1), embed_size))

    # srcMap = [[0,0] for _ in range(len(src_l))]
    # dstMap = [[0,0] for _ in range(len(src_l))]
    # ts = np.append(ts, np.max(ts) + 1)
    ts_cut_order_idx = np.argsort(ts_cut)
    ts_cut_order_idx = np.append(ts_cut_order_idx, -1)
    
    src_l = np.append(src_l, np.max(src_l) + 1)
    dst_l = np.append(dst_l, np.max(dst_l) + 1)
    ts_l = np.append(ts_l, np.max(ts_l) + 1)
    ts_l_order_idx = np.argsort(ts_l)

    # for i in range(len(src_l)):
    i = 0 # ts_l_idx
    j = 0 # ts_cut_idx
    ts_cut_idx = ts_cut_order_idx[j]
    while (i < len(src_l)):
        # ts_l_order_idx[i] is the current time stamp link;
        link_idx = ts_l_order_idx[i]

        if ts_l[link_idx] > ts_cut[ts_cut_idx]:
            # deal with all targets whose ts smaller than current timestamp in the graph.
            while (j < len(ts_cut)) and (ts_cut[ts_cut_idx] < ts_l[link_idx]):
                src_1_emb[ts_cut_idx] = src_dst_emb[nodeTimeDict[src_1[ts_cut_idx]]]
                src_2_emb[ts_cut_idx] = src_dst_emb[nodeTimeDict[src_2[ts_cut_idx]]]
                dst_emb[ts_cut_idx] = src_dst_emb[nodeTimeDict[dst[ts_cut_idx]]]

                j += 1
                ts_cut_idx = ts_cut_order_idx[j]

            if j == len(ts_cut):
                break

        nodeTimeDict[src_l[link_idx]] = link_idx
        # nodeTimePairDict[(src_l[link_idx], ts_l[link_idx])] = link_idx

        nodeTimeDict[dst_l[link_idx]] = link_idx + n_size
        # nodeTimePairDict[(dst_l[link_idx], ts_l[link_idx])] = link_idx + n_size

        i += 1
    return src_1_emb, src_2_emb, dst_emb

# main

# INITIALIZE PARAMETERS
# parser = argparse.ArgumentParser()
# parser.add_argument('--data', required=True, help='Network name')
# parser.add_argument('--model', default='tgat', choices=['jodie', 'nhp', 'TGN', 'tgat'], help="Model name")
# parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
# parser.add_argument('--epoch', default=50, type=int, help='Epoch id to load')
# parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions')
# parser.add_argument('--train_proportion', default=0.8, type=float, help='Proportion of training interactions')
# parser.add_argument('--state_change', default=True, type=bool, help='True if training with state change of users in addition to the next interaction prediction. False otherwise. By default, set to True. MUST BE THE SAME AS THE ONE USED IN TRAINING.') 

# parser.add_argument('--interpretation', action='store_true', default=False, help='Interpretation or not')
# parser.add_argument('--interpretation_type', type=int, default=0, help='Interpretation type: For interpretation, we have 4 tasks. 1: closure vs trianlge; 2: triangle + closure vs wedge; 3: wedge and edge; 4: closure and wedge; Default 0 means no interpretation')
# parser.add_argument('--test_path', type=str, default=None, help='Best model File Path')
# parser.add_argument('--time_prediction', action='store_true', default=False, help='Time prediction task')
# parser.add_argument('--time_prediction_type', type=int, default=0, help='Interpretation type: For time_prediction, we have 3 tasks. 1 for closure; 2 for triangle; 3 for wedge;  Default 0 means no time_prediction')
# parser.add_argument('--debug', action='store_true', default=False, help='Time prediction task')

args, sys_argv = get_args()
args.test_baselines = True
GPU = args.gpu
DATA = args.data
device = torch.device('cuda:{}'.format(GPU))
LEARNING_RATE = args.lr

interpretation = args.interpretation
interpretation_type = args.interpretation_type
time_prediction = args.time_prediction
time_prediction_type = args.time_prediction_type
time_window_factor, time_start_factor = 0.10, 0.4

# load train/val/test cls_tri, opn_tri, wedge, nega
file_path = './saved_triplets/'+DATA+'/'+DATA+'_'+str(time_start_factor)+'_'+str(time_window_factor)
test = 0
if os.path.exists(file_path) and (test==0):
    with open(file_path+'/triplets.npy', 'rb') as f:
        x = np.load(f, allow_pickle=True)
        cls_tri, opn_tri, wedge, nega, set_all_nodes = x[0], x[1], x[2], x[3], x[4]
        print("close tri", len(cls_tri[0]))
        print("open tri", len(opn_tri[0]))
        print("wedge", len(wedge[0]))
        print("nega", len(nega[0]))

# deal with data
# find and store the (node, t)'s corresponding embedding
# # Load data and sanity check
g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))
e_feat = np.load('./processed/ml_{}.npy'.format(DATA))
n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))
src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values

if DATA == 'congress-bills':
    pass
elif DATA == 'DAWN':
    pass
# elif DATA == 'NDC-substances':
#     pass
else:
    pass
    _time = 1
    while _time < max(ts_l):
        _time = _time * 10
    if time_prediction:
        ts_l = ts_l * 1.0 / (_time * 1e-7)    
    else:
        ts_l = ts_l * 1.0 / (_time * 1e-7)
ts_l = ts_l - min(ts_l) 

# t_max = ts_l.max()
# t_min = ts_l.min()
# time_window = time_window_factor * (t_max - t_min)
# time_start = t_min + time_start_factor * (t_max - t_min)
# time_end = t_max - time_window_factor * (t_max - t_min)
# randomly choose 70% as training, 15% as validating, 15% as testing
ts1 = time_start_factor + 0.7 * (1 - time_start_factor - time_window_factor)
ts2 = time_start_factor + 0.85 * (1 - time_start_factor - time_window_factor)
ts_start = (ts_l.max() - ts_l.min()) * time_start_factor + ts_l.min()
ts_end = ts_l.max() - (ts_l.max() - ts_l.min()) * time_window_factor
ts_train = (ts_end - ts_start) * 0.7 + ts_start
ts_val = (ts_end - ts_start) * 0.85 + ts_start

# load embeddings from different models
# modelName = args.model

src_emb = np.load('./embedding_output/{}_{}_embedding_src.npy'.format(args.data, args.model))
dst_emb = np.load('./embedding_output/{}_{}_embedding_dst.npy'.format(args.data, args.model))
n_size = src_emb.shape[0]
embed_size = src_emb.shape[1]
src_dst_emb = np.concatenate((src_emb, dst_emb))
# src_dst_emb = np.zeros((n_size * 2, embed_size))
# src_dst_emb = np.random.rand(n_size * 2, embed_size)



if interpretation:
    num_class = 2
elif time_prediction:
    num_class = 1
else:
    num_class = 4

if interpretation:
    interpretation = True
    model = finalClassifier_inter(embed_size, embed_size, embed_size, embed_size, num_class).to(device)
elif time_prediction:
    interpretation = False
    model = finalClassifier_time_prediction(embed_size, embed_size, embed_size, embed_size, num_class).to(device)
else:
    interpretation = False
    model = finalClassifier(embed_size, embed_size, embed_size, embed_size, num_class).to(device) # cls_tri, opn_tri, wedge, neg

dataset = TripletSampler(cls_tri, opn_tri, wedge, nega, ts_start, ts_train, ts_val, ts_end, set_all_nodes, DATA, args.interpretation_type, time_prediction_type=args.time_prediction_type)

bs = 128
dataset.set_batch_size(bs)
dataset.initialize()
dataset.reset()



# print(src_1_emb_cut)
# print(src_2_emb_cut)
# print(dst_emb_cut)

logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys_argv)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
early_stopper = EarlyStopMonitor(tolerance=1e-3)

train_val(dataset, model, args.mode, args.bs, args.n_epoch, criterion, optimizer, early_stopper, logger, interpretation=interpretation, time_prediction=time_prediction, device=device)
