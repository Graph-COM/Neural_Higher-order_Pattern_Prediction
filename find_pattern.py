import numpy as np
from tqdm import tqdm
from itertools import permutations

# preprocess dataset
def preprocess_dataset(ts_list, src_list, dst_list, node_max, edge_idx_list, label_list, time_window_factor=0.1, time_start_factor=0.4, logger=None):
    """
    we turn the original hypergraph into a graph by clique expansion to preprocess it.
    ts_list: numpy list, timestamp for the edge
    src_list: numpy list, source list
    dst_list: numpy list, dst list
    node_max: max id of node, indicate the node number
    edge_idx_list: indicates idx of the corresponding edge
    label_list: the hyperedge idx, for those edges with the same label_list, they come from one hyperedge
    time_window_factor: time window, default 0.1
    time_start_factor: when to start, default 0.4

    This function is used to calculate when the node appears for the first time, and if the new hyperedge forms a new closure(cls_tri).
    After calculate these, the function will call the find_triangle_closure to find the other patterns.

    return cls_tri, opn_tri, wedge, edge, set_all_node
    for cls_tri, opn_tri, wedge, edge, we record all the information, including the nodes, hyperedges idx, timestamp of hyperedges
    """

    t_max = ts_list.max()
    t_min = ts_list.min()
    time_window = time_window_factor * (t_max - t_min)
    time_end = t_max - time_window_factor * (t_max - t_min)
    edges = {}  # edges dict: all the edges
    edges_idx = {}  # edges index dict: all the edges index, corresponding with edges
    adj_list = {}
    node_idx = {}
    node_simplex = {}
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
            # TODO: permutation
            # this is to check whether the simplex contains a cls_tri(closure)
            perm = permutations(list_simplex, 3)
            # i.e., 
            # for _i in list(list_simplex):
            #     for _j in list(list_simplex):
            #         for _l in list(list_simplex):
            for _perm in perm :
                # if three nodes already in the closure
                if _perm in node_simplex:
                    continue

                _i, _j, _l = _perm
                
                if ((_i, _j) in edges) and (edges[(_i, _j)] <= time_end) and (node_first_time[_l] < edges[(_i, _j)]):  #TODO: dict.get()
                # assume w first appear at the same time as edge(u,v), then no previous information, no way to predict
                    timing = ts_list[i-1] - edges[(_i, _j)] 
                    if (timing > 0) and (timing < time_window):
                    # timing = 0 indicates that three nodes appear for the first time in one closure. This doesn't make sense.
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
                # shouldn't run this, still write here because in case the timestamp is not in order
                print("wrong timestamp in find_pattern.py, the ts should be increasing")
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
        # can be improved if we set for all (i,j) in edges, we all have i < j
        # but also need to change the following code
        src = dst_list[i]
        tgt = src_list[i]
        if (src, tgt) in edges:            
            if edges[(src, tgt)] > ts:
                # shouldn't run this, still write here because in case the timestamp is not in order
                print("wrong timestamp in find_pattern.py, the ts should be increasing")
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
    return find_triangle_closure(ts_list, node_max, edges, adj_list, edges_idx, node_simplex, simplex_ts, node_first_time, node2simplex, time_window_factor, time_start_factor, logger)

def find_triangle_closure(ts_list, node_max, edges, adj_list, edges_idx, node_simplex, simplex_ts, node_first_time, node2simplex, time_window_factor, time_start_factor=0.4, logger=None):
    """
    find triangle closure
    ts_list: time stamp list
    node_max: the max node id, aka, the node number
    edges: dict of first edges appear between node pair (i,j) at time t, i.e., {(i,j)}=t
    adj_list: adjacency list
    edges_idx: record the corresponding edge idx in edges
    node_simplex: dict, store all closure. Used to check if the open triangle is already a closure.
    simplex_ts: dict, the correponding time of each closure(i,j,k), where idx_i < idx_j < idx_k, {(i,j,k)} = t
    node_first_time: dict, the first time the node appear
    node2simplex: no use
    time_window_factor: time_window_factor * total_time = time_window_size, default 0.1
    time_start_factor: time_start_factor * total_time = time_start, default 0.4

    The general idea of this is to find the first edge, then the third node. To avoid repeat counting, we assume x1 < x2 < x3, where x1, x2, x3 are the time of three edges.
    If x1=x2, then we assume the edge_idx of x1 < the edge_idx of x2

    return:
    cls_tri(closure), opn_tri(triangle), wedge(wedge), edge(edge), set_all_node
    for cls_tri, opn_tri, wedge, edge, we record all the information, including the nodes, hyperedges idx, timestamp of hyperedges
    """
    node_max = int(node_max)
    t_max = ts_list.max()
    t_min = ts_list.min()
    time_window = time_window_factor * (t_max - t_min)
    time_start = t_min + time_start_factor * (t_max - t_min)
    time_end = t_max - time_window_factor * (t_max - t_min)
    
    # closure
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

    # edge(only one edge between the first two nodes in three nodes)
    src_1_edge = []
    src_2_edge = []
    dst_edge = []
    ts_edge_1 = []
    edge_idx_edge_1 = []
    count_edge = 0 # <a,b>

    set_all_node = set(adj_list.keys())
    print(len(list(set_all_node)))

    dict_processed_bool = {}

    

    for edge_i in edges.keys(): # first edge
        i, j = edge_i

        # second edge (j,l) 
        x1 = edges[edge_i]
        if (x1 < time_start) or (x1 > time_end):
            continue
        x1_idx = edges_idx[edge_i]        

        set_edge = None
        """
        Edge:
        deal with edge(no interaction with the third nodes)
        set_all_nodes - {(i, x)} - {(j,x)}
        calculate the original situation (only one link between the first two nodes)
        """
        if not ((i,j) in dict_processed_bool):
            dict_processed_bool[(i,j)] = 1
            dict_processed_bool[(j,i)] = 1
            set_edge = list(set_all_node - set(adj_list[j]) - set(adj_list[i])) # set_edge: the set of nodes which construct an edge with (i,j)

        for l in adj_list[j]:
            if (l==j) or (l==i) or (node_first_time[l] >= x1):
                continue

            x2 = edges[(j,l)]
            x2_idx = edges_idx[(j,l)]
            
            # Edge:
            # although can at least form wedge, out of time window, we consider it as edge(edge);
            if (x2 - x1 > time_window):
                if set_edge is None:
                    set_edge = [l]
                else:
                    set_edge.append(l)
                continue
            
            # to decrease dimension we assume x1 <= x2 <= x3; with the same timestamp, we have the edge_idx in ascending order
            if (x1 > x2) or (x1 == x2 and x1_idx > x2_idx):
                continue

            l3 = 0             
            if (l,i) in edges:
                x3 = edges[(l,i)] # since we record all edges in edges dict, so both (l,i) and (i,l) are okay
                x3_idx = edges_idx[(l,i)]
                # wedge: although can form a triangle, but out of window, we still consider it as wedge(wedge)
                # it can't be an edge, since we already handle the edge cases in the previous part
				# However, since this is not a real wedge, we can also just ignore it. The performance will not change much.
                if x3 - x1 > time_window:
                    pass
                # either be a triangle / closure
                elif ((x3 > x2) or (x3 == x2 and x3_idx > x2_idx)) and (x3 - x1 < time_window) and (x3 - x1 > 0):
                    l3 = 1

            # since the timestamp of edge is in order, the node idx order may be different
            # a better way to do this is also record the the node_simplex in order and order (i,j,l) to check

            l1 = (i, j, l) in node_simplex 
            # l1 indicates closure
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

                # total closure(cls_tri)
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

                # total triangle(opn_tri)
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

                    # total wedge(wedge)
                    count_wedge += 1
        
        if not(set_edge is None):
            set_edge = np.array(list(set_edge)).astype(int)
            l = len(set_edge)
            sample_length = min(100, l) # since too many edges, remove some to make it simple
            idx = np.random.choice(l, sample_length, replace=False)
            set_edge = set_edge[idx]
            for l in set_edge:
                # (i,j,l)
                if node_first_time[l] <= x1:
                    src_1_edge.append(i)
                    src_2_edge.append(j)
                    dst_edge.append(l)
                    ts_edge_1.append(x1)
                    edge_idx_edge_1.append(x1_idx)

                    # total edge(edge)
                    count_edge += 1
           
    cls_tri = [np.array(src_1_cls_tri), np.array(src_2_cls_tri), np.array(dst_cls_tri), np.array(ts_cls_tri_1), np.array(ts_cls_tri_2), np.array(ts_cls_tri_3), np.array(edge_idx_cls_tri_1), np.array(edge_idx_cls_tri_2), np.array(edge_idx_cls_tri_3)]
    opn_tri = [np.array(src_1_opn_tri), np.array(src_2_opn_tri), np.array(dst_opn_tri), np.array(ts_opn_tri_1), np.array(ts_opn_tri_2), np.array(ts_opn_tri_3), np.array(edge_idx_opn_tri_1), np.array(edge_idx_opn_tri_2), np.array(edge_idx_opn_tri_3)]
    wedge = [np.array(src_1_wedge), np.array(src_2_wedge), np.array(dst_wedge), np.array(ts_wedge_1), np.array(ts_wedge_2), np.array(edge_idx_wedge_1), np.array(edge_idx_wedge_2)]
    edge = [np.array(src_1_edge), np.array(src_2_edge), np.array(dst_edge), np.array(ts_edge_1), np.array(edge_idx_edge_1)]

    logger.info(f"Total sample number:  Cls Tri:  {count_cls_tri}, Opn Tri:  {count_opn_tri}, Wedge:  {count_wedge}, Edge:  {count_edge}")
    return cls_tri, opn_tri, wedge, edge, set_all_node

