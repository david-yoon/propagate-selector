#-*- coding: utf-8 -*-

"""
re-implement wang's code
https://github.com/shuohangwang/SeqMatchSeq/blob/master/util/utils.lua
"""
import numpy as np

"""
y_true  : label
y_score: predicted value
must -> len(y_true) == len(y_score)
"""
def MAP(y_true, y_score):

    ground_label = y_true
    predict_label = []

    map = float(0)
    map_idx = 0
    extracted = []
    
    # check duplicate
    tmp_check = set(y_score)
    if (len(y_score) - len(tmp_check)) > 3:
        print ("noti same predictions")

    for i in range( len(ground_label) ):
        if ground_label[i] != 0 :
            extracted.append(i)

    # for resolving same number
    sss = sorted( y_score, reverse=True )
    #sss_rank = [ list( y_score).index(x) for x in sss]
    
    # for resolving same number
    sss_rank = []
    for x in sss:
        list_index = ([i for i, val in enumerate(y_score) if val==x])
        if( len(list_index) == 1 ):
            sss_rank.append( list_index[0] )
        else:
            for i in list_index:
                if i not in sss_rank:
                    sss_rank.append(i)
                    break    
    
    predict_label = np.asarray(sss_rank)
    
#     print (extracted)
#     print (predict_label)
    
    for i in range( len(predict_label) ) :
        if predict_label[i] in extracted :
            map_idx = map_idx + 1
            map = map + map_idx / float(i+1)

    if map_idx != 0:
        map = map / float(map_idx)
    else:
        map = 0
    
    '''
    with open('./TEST_log_map_results.txt', 'a') as f:

        # label index
        str_extracted = [str(x) for x in extracted]
        f.write( ' '.join(str_extracted) )

        # map
        f.write('\t[MAP: ' + str(map) + ']')
        f.write('\n')

        # prediction (rank)
        str_predict_label = [str(x) for x in predict_label]
        f.write( ' '.join(str_predict_label) )
        f.write('\n')

        # prediction (prob.)
        str_y_score = [str(x) for x in y_score]
        f.write( ' '.join(str_y_score) )
        f.write('\n\n')
    '''
    
    return map



def MRR(y_true, y_score):

    ground_label = y_true
    predict_label = []

    mrr = float(0)
    mrr_idx = 0
    extracted = []

    for i in range( len(ground_label) ):
        if ground_label[i] != 0 :
            extracted.append(i)

    sss = sorted( y_score, reverse=True )
    #sss_rank = [ list( y_score).index(x) for x in sss]
    
    # for resolving same number
    sss_rank = []
    for x in sss:
        list_index = ([i for i, val in enumerate(y_score) if val==x])
        if( len(list_index) == 1 ):
            sss_rank.append( list_index[0] )
        else:
            for i in list_index:
                if i not in sss_rank:
                    sss_rank.append(i)
                    break    
    
    
    predict_label = np.asarray(sss_rank)
    
    for i in range( len(predict_label) ) :
        if predict_label[i] in extracted :
            mrr = 1.0 / float(i+1)
            break

    return mrr