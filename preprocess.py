# change the simplex version to the HIT version
"""
The code is used to expand the hypergraph to the traditional graph.
Changed the file_name here to process the simplex data.
The original data can be downloaded from https://www.cs.cornell.edu/~arb/data/
In our paper, we use tags-sx-math, tags-ask-ubuntu, congress-bills, DAWN, NDC-substances
We also provide tags-sx-math as a example.
"""
import numpy as np
import argparse
import sys

# Load data and sanity check
def get_args():
    parser = argparse.ArgumentParser('PNAS Baseline')

    # select dataset and training mode
    parser.add_argument('-d', '--data', type=str, help='data sources to use, DAWN, tags-ask-ubuntu, tags-math-sx, NDC-substances, congress-bills',
                        choices=['DAWN', 'tags-ask-ubuntu', 'tags-math-sx', 'NDC-substances', 'congress-bills', 'threads-ask-ubuntu'],
                        default='tags-math-sx')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv

args, sys_argv = get_args()
DATA = args.data

file_name = DATA

fout = open('./processed/ml_' + file_name + '.csv', 'w')
fout.write(',u,i,ts,label,idx\n')
file_addr = './data/' + file_name + '/' + file_name

fin_nverts = open(file_addr + '-nverts.txt', 'r')
fin_simplices = open(file_addr + '-simplices.txt', 'r')
fin_times = open(file_addr + '-times.txt', 'r')

nverts = []
simplices = []
times = []
node2idx = {}
for i in fin_nverts:
    nverts.append(int(i))
count = 1
for i in fin_simplices:
    simplices.append(int(i))

last_time = -1
idx = -1
for i in fin_times:
    idx += 1
    if int(i) >= last_time:
        last_time = int(i)
    else:
        pass
        # print("Time not monotune", last_time, int(i), nverts[idx])
    times.append(int(i))

print("First")

times = np.array(times)
y = np.argsort(times) 
print(y)

nvertsList = np.array(nverts)
print("Average Size: ", np.mean(nvertsList[nvertsList>1]))
print("Total size", np.sum(nvertsList), "total hyperedges",len(nvertsList), "total nodes hyperedges > 1",np.sum(nvertsList[nvertsList>1]))

simplices_i = 0
edge_idx = 0
exist_solo = len(node2idx.keys())
node_bool = {}
node_list_total = []
for idx_nverts, nverts_i in enumerate(nverts):
    node_list_total.append(simplices[simplices_i: simplices_i + nverts_i])
    if nverts_i == 1: # there may be 1 simplex, which means doesn't have edge with other nodes, so remove them
        simplices_i += 1
        continue
    
    for i in simplices[simplices_i: simplices_i + nverts_i]:
        if not(i in node2idx):
            node2idx[i] = count
            count += 1
    
    simplices_i += nverts_i

simplex_idx = -1
for idx_y in y:
    node_list = node_list_total[idx_y]
    if len(node_list) == 1:
        continue
    simplex_idx += 1
    for idx_st, st in enumerate(node_list[:-1]):
        for ed in node_list[idx_st+1:]:
            node_bool[node2idx[st]] = 1
            node_bool[node2idx[ed]] = 1
            fout.write("%s,%s,%s,%s,%s,%s\n" %(edge_idx, node2idx[st], node2idx[ed], times[idx_y], simplex_idx, edge_idx + 1))
            
            edge_idx += 1

fout.close()
fin_times.close()
fin_simplices.close()
fin_nverts.close()

# since we do not have  node feature and edge feature, we use all zeros matrix
# the dimension is 172
rand_feat = np.zeros((count, 172))
np.save('./processed/ml_'+ file_name + '_node.npy', rand_feat)
rand_feat = np.zeros((edge_idx, 172))
np.save('./processed/ml_'+ file_name + '.npy', rand_feat)

print("total nodes ", len(node2idx))
print("ave link stream intensity ", edge_idx * 1.0 / len(node2idx) / (max(times) - min(times)))
