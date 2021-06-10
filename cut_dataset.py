# cut the large dataset
import pickle

file_path = './saved_triplets/threads-ask-ubuntu/threads-ask-ubuntu_0.4_0.1'

with open(file_path+'/triplets_ori.npy', 'rb') as f: 
    x = pickle.load(f)

cls_tri, opn_tri, wedge, nega, set_all_nodes = x[0], x[1], x[2], x[3], x[4]

print("close tri", len(cls_tri[0]))
print("open tri", len(opn_tri[0]))
print("wedge", len(wedge[0]))
print("nega", len(nega[0]))

n = len(nega[0])

import numpy as np

p = 1000
idx = np.random.choice(range(n), int(n/p), replace=False)
idx_sorted = np.sort(idx, kind='quicksort')

nega_new = [nega[0][idx], nega[1][idx], nega[2][idx], nega[3][idx], nega[4][idx]]

with open(file_path+'/triplets.npy', 'wb') as f:
    x = np.array([cls_tri, opn_tri, wedge, nega_new, set_all_nodes])
    np.save(f, x)

