import pandas as pd
from log import set_up_logger
from parser import *
from eval import *
from utils import *
from train import *
from find_pattern import *
import os
from module import HIT
from graph import NeighborFinder
import resource
from sklearn.preprocessing import scale
from histogram import plot_hist
import time

args, sys_argv = get_args()

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
ATTN_NUM_HEADS = args.attn_n_head
DROP_OUT = args.drop_out
GPU = args.gpu
USE_TIME = args.time
ATTN_AGG_METHOD = args.attn_agg_method
ATTN_MODE = args.attn_mode
DATA = args.data
NUM_LAYER = args.n_layer
if args.time_prediction_type > 0:
    LEARNING_RATE = 1e-3
else:
    LEARNING_RATE = args.lr

POS_ENC = args.pos_enc
POS_DIM = args.pos_dim
TIME_DIM = args.time_dim
WALK_N_HEAD = args.walk_n_head
WALK_MUTUAL = args.walk_mutual
TOLERANCE = args.tolerance
CPU_CORES = args.cpu_cores
NGH_CACHE = args.ngh_cache
VERBOSITY = args.verbosity
interpretation = args.interpretation
interpretation_type = args.interpretation_type
time_prediction = args.time_prediction
time_prediction_type = args.time_prediction_type
ablation = args.ablation
ablation_type = args.ablation_type
WALK_POOL = args.walk_pool
walk_linear_out = args.walk_linear_out
test_path = args.test_path

AGG = args.agg
SEED = args.seed
assert(CPU_CORES >= -1)
set_random_seed(SEED)
logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys_argv)
if args.debug == False:
    if interpretation:
        print("Interpretation used spd, walk Linear out, mean pooling for walk")
        POS_ENC = 'spd'
        walk_linear_out = True
        WALK_POOL = 'sum'
        best_model_path = './interpretation_output/{}_{}_{}.pth'.format(str(time.time()), DATA, interpretation_type)
        fout = open('./interpretation_output/{}_{}_{}.txt'.format(str(time.time()), DATA, interpretation_type), 'w')
        sys.stdout = fout
    elif ablation:
        print("Ablation study: wedge and open triangle")
        # best_model_path = './ablation_output/{}_{}.pth'.format(DATA, ablation_type)
        # fout = open('./interpretation_output/{}_{}.txt'.format(DATA, interpretation_type), 'w')
        # POS_ENC = 'lp'
        # sys.stdout = fout
    elif time_prediction:
        print("Time prediction used lp")
        """
        for time prediction task, remember to plus 1 for the delta t.
        Since delta may be 1, thus log 1 becomes 0, eventually lead to nan.
        """
        POS_ENC = 'lp'
        best_model_path = './time_prediction_output/{}_{}_{}.pth'.format(str(time.time()), DATA, time_prediction_type)
        fout = open('./time_prediction_output/{}_{}_{}.txt'.format(str(time.time()), DATA, time_prediction_type), 'w')
        sys.stdout = fout

# Load data and sanity check
g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))
# e_feat = np.load('./processed/ml_{}.npy'.format(DATA))
# n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))
src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values

# scale the time
# congress-bills and DAWN dont have to do the scale
if DATA == 'congress-bills':
    pass
elif DATA == 'DAWN':
    pass
# elif DATA == 'NDC-substances':
#     pass
else:
    pass
    _time = 1
    while _time < max(ts_l): # 0 - x * 10^7
        _time = _time * 10
    if time_prediction:
        ts_l = ts_l * 1.0 / (_time * 1e-7)    
    else:
        ts_l = ts_l * 1.0 / (_time * 1e-7)
    
    print(_time)
print(max(ts_l), min(ts_l))
ts_l = ts_l - min(ts_l) 
print(max(ts_l), min(ts_l))
max_idx = max(src_l.max(), dst_l.max())
print(max_idx, np.unique(np.stack([src_l, dst_l])).shape[0])
assert(np.unique(np.stack([src_l, dst_l])).shape[0] == max_idx)  # all nodes except node 0 should appear and be compactly indexed

import pickle

# find possitive and negative triplet
time_window_factor, time_start_factor = 0.10, 0.4

file_path = './saved_triplets/'+DATA+'/'+DATA+'_'+str(time_start_factor)+'_'+str(time_window_factor)
test = 0
if os.path.exists(file_path+'/triplets.npy') and (test==0):
    if DATA == 'threads_ask_ubuntu':
        with open(file_path+'/triplets.npy', 'rb') as f: 
            x = pickle.load(f)
    else:
        with open(file_path+'/triplets.npy', 'rb') as f:
            x = np.load(f, allow_pickle=True)

    cls_tri, opn_tri, wedge, nega, set_all_nodes = x[0], x[1], x[2], x[3], x[4]
    logger.info(f"close tri {len(cls_tri[0])}")
    logger.info(f"open tri {len(opn_tri[0])}")
    logger.info(f"wedge {len(wedge[0])}")
    logger.info(f"edge {len(nega[0])}")

else:
    cls_tri, opn_tri, wedge, nega, set_all_nodes = preprocess_dataset(ts_list=ts_l, src_list=src_l, dst_list=dst_l, node_max=max_idx, edge_idx_list=e_idx_l, 
                                                                      label_list=label_l, time_window_factor=time_window_factor, time_start_factor=time_start_factor, logger=logger)
    if not(os.path.exists(file_path)):
        os.makedirs(file_path)
    with open(file_path+'/triplets.npy', 'wb') as f:
        x = np.array([cls_tri, opn_tri, wedge, nega, set_all_nodes])
        np.save(f, x)

# triangle closure
# choose 70% as training, 15% as validating, 15% as testing
ts1 = time_start_factor + 0.7 * (1 - time_start_factor - time_window_factor)
ts2 = time_start_factor + 0.85 * (1 - time_start_factor - time_window_factor)
ts_start = (ts_l.max() - ts_l.min()) * time_start_factor + ts_l.min()
ts_end = ts_l.max() - (ts_l.max() - ts_l.min()) * time_window_factor
ts_train = (ts_end - ts_start) * 0.7 + ts_start
ts_val = (ts_end - ts_start) * 0.85 + ts_start

# create two neighbor finders to handle graph extraction.
# for training and validating use the partial one
# while test phase still always uses the full one
# no need now, in case we have transductive and inductive settings in the future

full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, bias=args.bias, use_cache=NGH_CACHE, sample_method=args.pos_sample)

partial_adj_list = [[] for _ in range(max_idx + 1)]
idx_partial = ts_l <= ts_val
for src, dst, eidx, ts in zip(src_l[idx_partial], dst_l[idx_partial], e_idx_l[idx_partial], ts_l[idx_partial]):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))    
partial_ngh_finder = NeighborFinder(partial_adj_list, bias=args.bias, use_cache=NGH_CACHE, sample_method=args.pos_sample)

print("Finish build HIT")
ngh_finders = partial_ngh_finder, full_ngh_finder

# multiprocessing memory setting
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (200*args.bs, rlimit[1]))

# model initialization
device = torch.device('cuda:{}'.format(GPU))
hit = HIT(agg=AGG,
          num_layers=NUM_LAYER, use_time=USE_TIME, attn_agg_method=ATTN_AGG_METHOD, attn_mode=ATTN_MODE,
          n_head=ATTN_NUM_HEADS, drop_out=DROP_OUT, time_dim=TIME_DIM, pos_dim=POS_DIM, pos_enc=POS_ENC, walk_pool=WALK_POOL, 
          num_neighbors=NUM_NEIGHBORS, walk_n_head=WALK_N_HEAD, walk_mutual=WALK_MUTUAL, walk_linear_out=walk_linear_out,
          cpu_cores=CPU_CORES, verbosity=VERBOSITY, get_checkpoint_path=get_checkpoint_path, interpretation=interpretation, 
          interpretation_type=interpretation_type, time_prediction=time_prediction, ablation=ablation, ablation_type=ablation_type, device=device)
hit.to(device)

# dataset initialization
dataset = TripletSampler(cls_tri, opn_tri, wedge, nega, ts_start, ts_train, ts_val, ts_end, set_all_nodes, DATA, 
                         interpretation_type, time_prediction_type=time_prediction_type, ablation_type=ablation_type)

logger.info(interpretation and (not os.path.exists(best_model_path)))

# if ((not interpretation) and test_path is None) or (interpretation and (not os.path.exists(best_model_path))): # change only for interpretation
if (not os.path.exists(best_model_path)):
    optimizer = torch.optim.Adam(hit.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    early_stopper = EarlyStopMonitor(tolerance=TOLERANCE)

    # start train and val phases
    train_val(dataset, hit, args.mode, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, ngh_finders, logger, interpretation=interpretation, time_prediction=time_prediction)
else:
    logger.info("load model")
    dataset.set_batch_size(BATCH_SIZE)
    hit.load_state_dict(torch.load(test_path))
    
dic1 = {}
dic2 = {}
# final testing
hit.update_ngh_finder(full_ngh_finder)  # remember that testing phase should always use the full neighbor finder

"""
comment for interpretation
"""
if time_prediction:
    NLL_total, num_test_instance, time_predicted_total, time_gt_total = eval_one_epoch('test for {} nodes'.format(args.mode), hit, dataset, val_flag='test', interpretation=interpretation, time_prediction=time_prediction)    
    print('Testing NLL: ', NLL_total, 'num instances: ', num_test_instance)
    file_addr = './Histogram/'+DATA+'/'
    print("time_predicted_total", time_predicted_total)
    print("time_gt_total", time_gt_total)
    if not os.path.exists(file_addr):
            os.makedirs(file_addr)
    histogram.plot_hist_multi([time_predicted_total, time_gt_total], bins=50, figure_title='time_prediction_histogram', file_addr=file_addr, label=['Ours', 'Groundtruth'])
else:
    test_acc, test_ap, test_f1, test_auc, cm = eval_one_epoch('test for {} nodes'.format(args.mode), hit, dataset, val_flag='test', interpretation=interpretation, time_prediction=time_prediction)
    print('Test statistics: {} all nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_acc, test_auc, test_ap))
    # print("Confusion matrix\n", cm)
    logger.info(', '.join(str(r) for r in cm.reshape(1,-1)))
    logger.info('Test statistics: {} all nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_acc, test_auc, test_ap))

# save model
if test is None:
    logger.info('Saving hit model')
    torch.save(hit.state_dict(), best_model_path)
    logger.info('hit models saved')
