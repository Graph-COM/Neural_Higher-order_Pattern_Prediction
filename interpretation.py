import numpy as np
# import torch
"""
Process the walk pattern for the interpretation.
pattern: np.array, walk pattern, [batch_size, walks, pattern]
score: tensor, walk score, [batch_size, walks, 4] totally 4 classes. cls, opn, wedge, none; before softmax
source: int, indicate whether they are the paths from source(1) or target(0). Currently not used.
topk: int, top k patterns do distinguish from different classes

return:
result: np.array, for each class pair(directed), the topk patterns and lowk patterns(mean score),
        from the lowest to the highest, [num_class, num_class, 2 * topk]

"""
def process_pattern(pattern, score, pattern_dict, source=None, topk=5, non_idx=4, file=None, pattern_label=None): 
    
    
    # pattern = np.array([1,2,3,1,2,3])
    # score = np.array([[1,2], [1,3], [1,4], [1,2], [1,3], [1,4]])
    print("check ", len(pattern_label[pattern_label == 1]), len(pattern_label[pattern_label == 0]))
    pattern = pattern.reshape(-1)
    num_class = score.shape[-1] # usually 4
    # score = score.view(-1, num_class).detach().cpu().numpy() # change score to numpy
    # print(num_class)
    
    result = np.zeros([num_class, num_class, 2 * topk]) # return
    _idx = np.zeros([num_class, num_class, 2 * topk], dtype=np.int16) # return

    # score = torch.softmax(score)
    num_walk = score.shape[0] #* score.shape[1]
    num_walk_batch = 1000
    
    """
    non_idx = 6
    """
    # non_idx = 6
    idx = [] # idx[l]: the idx of pattern l
    # pattern_set = np.unique(pattern)
    pattern_set = []
    non_idx = str(non_idx)
    non_str = '[{} {} {}]'.format(non_idx, non_idx, non_idx)
    # print(non_str)
    # print(len(list(pattern_dict.keys())))
    dict_pattern2str = {}
    for i in pattern_dict.keys():
        # print(i)
        if non_str not in i:
            pattern_set.append(pattern_dict[i])
            dict_pattern2str[pattern_dict[i]] = i
    # TODO: deal with pattern, eliminate all [4,4,4] and all pattern less than 200
    # print(pattern_set)
    pattern_set_new = []
    # dict_pattern2str_new = {}
    for i in pattern_set:
        # print(pattern[pattern == i])
        # print(list(pattern[pattern == i]), len(list(pattern[pattern == i])))
        # print(i)
        #  and (non_str not in i)
        # comment to test if all patterns are correct
        """
        """
        if (len(list(pattern[pattern == i])) > 200):
            pattern_set_new.append(i)
        # pattern_set_new.append(i)
            # dict_pattern2str_new[i] = 
    pattern_set = np.array(pattern_set_new)

    print(pattern_set)
    num_pattern_set = len(pattern_set)
    # print(num_pattern_set)
    score_bucket = np.zeros([num_pattern_set, num_class])
    var_bucket = np.zeros([num_pattern_set, num_class])
    count_bucket = np.zeros([num_pattern_set], dtype=np.int32)
    # process_score = np.zeros([num_class, num_class, num_pattern_set])
    
    print("total ratio")
    count_for_class = np.zeros([num_pattern_set, num_class])
    for idx_pattern, l in enumerate(pattern_set): # l: pattern idx
        idx_pattern_l = pattern == l # idx for pattern = l
        idx.append(idx_pattern_l)
        score_bucket[idx_pattern] = score[idx_pattern_l].mean(0)
        var_bucket[idx_pattern] = np.var(score[idx_pattern_l], axis=0)
        count_bucket[idx_pattern] = len(score[idx_pattern_l])
        for i in range(num_class):
            count_for_class[idx_pattern][i] = len(score[(idx_pattern_l) * (pattern_label == i)])
    
        print(dict_pattern2str[l], count_for_class[idx_pattern][0] * 100.0/ (count_for_class[idx_pattern][0] + count_for_class[idx_pattern][1]), count_for_class[idx_pattern])

    s1, s2 = 0, 0
    for i in count_for_class:
        s1 += i[0]
        s2 += i[1]
    print("check ", s1, s2)
    for i in range(num_class):
        for j in range(num_class):
            if i == j:
                continue

            # calculate score[i]-score[j]
            # note: score[i] - score[j] and score[j] - score[i] is different,
            # indicating the walk can distinguish class i and class j
            # print(score[:,i] )
            # process_score[i][j] = score[:,i] - score[:,j]
    
            # mean_bucket = np.zeros([num_pattern_set])
            # choose the top 3 pattern with max
            # for l in range(num_pattern_set):
            #     mean_bucket[l] = np.mean(score[idx[l]])
            mean_bucket = score_bucket[:,i] - score_bucket[:,j]


            sorted_idx = np.argsort(mean_bucket)
            _idx[i][j][0:topk] = sorted_idx[0:topk]
            result[i][j][0:topk] = pattern_set[sorted_idx[0:topk]]
            _idx[i][j][topk:2 * topk] = sorted_idx[-topk:]
            result[i][j][topk:2 * topk] = pattern_set[sorted_idx[-topk:]]

    # print(result)
    # f = open()
    print('mean bucket')
    print(score_bucket)
    print('variance bucket')
    print(var_bucket)
    print('count bucket')
    print(count_bucket)
    print('count for classes')
    print(count_for_class)
    for i in range(num_class):
        for j in range(num_class):
            if i == j:
                continue
            for l in range(2 * topk):
                # print(_idx[i][j][l])
                print(result[i][j][l])
                print(dict_pattern2str[result[i][j][l]])
                print('mean', score_bucket[_idx[i][j][l]], 'var', var_bucket[_idx[i][j][l]], 'count', count_bucket[_idx[i][j][l]], 'count for bucket', count_for_class[_idx[i][j][l]])
                
                # print()
                print("=========================")
                print()
    
    # whole information


    return mean_bucket, pattern_set, result #, score_bucket, var_bucket, count_bucket

# process_pattern(None, None, None, 2)
