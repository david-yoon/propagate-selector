#-*- coding: utf-8 -*-
"""
what    : detecting supporting sentences
data    : hotpot-qa
"""
from tensorflow.core.framework import summary_pb2
from random import shuffle
import numpy as np
from tqdm import tqdm

from measure_QA import MAP
from measure_QA import MRR

"""
    desc  : 
    
    inputs: 
        sess  : tf session
        model : model for test
        data  : such as the dev_set, test_set...
            
    return:
        sum_batch_ce : sum cross_entropy
        accr         : accuracy
        
"""
def run_test(sess, model, batch_gen, data):
    
    MAP_rst = 0
    MRR_rst = 0
    
    list_preds     = []
    list_labels    = []
    list_loss      = []
    list_seq_num_s = []
    
    preds, labels = [], []
    loss = 0
    
    max_loop  = int( len(data) / model.params.batch_size )

    # run 1 more loop for the remaining
    for test_itr in tqdm( range(max_loop + 1) ):
        
        list_q, list_s, list_g, list_l, list_len_q, list_len_s, list_num_s = batch_gen.get_batch(
            data=data,
            batch_size=model.params.batch_size,
            is_test=True,
            start_index= (test_itr* model.params.batch_size)
        )
        
        # prepare data which will be push from pc to placeholder
        input_feed = {}

        input_feed[model.batch_q]     = list_q
        input_feed[model.batch_s]     = list_s
        input_feed[model.batch_g]     = list_g
        input_feed[model.batch_l]     = list_l

        input_feed[model.batch_len_q] = list_len_q
        input_feed[model.batch_len_s] = list_len_s

        input_feed[model.batch_num_s] = list_num_s

        input_feed[model.dr_prob]     = 1.0
        input_feed[model.dr_rnn_prob] = 1.0
        
        try:
            preds, labels, loss = sess.run([model.predic_prob, model.y_masked, model.loss], input_feed)
        except Exception as e:
            print ("excepetion occurs in valid step : ", e)
            pass
        
        list_preds.extend( preds )
        list_labels.extend( labels )
        list_loss.append( loss )
        list_seq_num_s.extend( list_num_s )
    
    list_preds     = list_preds[:len(data)]
    list_labels    = list_labels[:len(data)]
    list_loss      = list_loss[:len(data)]
    list_seq_num_s = list_seq_num_s[:len(data)]
    
    # make sure it is padded
    for i in range( len(list_preds) ):
        list_preds[i][ list_seq_num_s[i]: ] = 0
    
    score_MAP = []
    score_MRR = []
   
    for y, y_hat, seq in zip(list_labels, list_preds, list_seq_num_s):
        score_MAP.append( MAP(y_true=y[:seq], y_score=y_hat[:seq]) )
        score_MRR.append( MRR(y_true=y[:seq], y_score=y_hat[:seq]) )

                
    MAP_rst = np.mean(score_MAP)
    MRR_rst = np.mean(score_MRR)
    
    
    if model.params.IS_SAVE_RESULTS_TO_FILE:
        print('save results as files')
        
        with open('./TEST_results.txt', 'w') as f:
            for y_hat, seq in zip(list_preds, list_seq_num_s):
                '''
                for idx, prob in enumerate(y_hat[:seq]):
                    if prob > 0.50:
                        f.write(str(idx)+' ')
                '''
                tmp = y_hat[:seq].argsort()[-10:][::-1]
                tmp = [str(x) for x in tmp]
                f.write( ' '.join(tmp) )
                f.write('\n')
                
        with open('./TEST_results_prob.txt', 'w') as f:
            for y_hat, seq in zip(list_preds, list_seq_num_s):
                tmp = y_hat[:seq].argsort()[-10:][::-1]
                tmp_prob = [y_hat[x] for x in tmp]
                tmp_prob = ['{:.4f}'.format(x) for x in tmp_prob]
                f.write( ' '.join(tmp_prob) )
                f.write('\n')    
            
    
    value1 = summary_pb2.Summary.Value(tag="dev_loss", simple_value= np.mean( list_loss ))
    value2 = summary_pb2.Summary.Value(tag="dev_MAP", simple_value=MAP_rst )
    value3 = summary_pb2.Summary.Value(tag="dev_MRR", simple_value=MRR_rst )                                   
    summary = summary_pb2.Summary(value=[value1, value2, value3])
    
    return np.mean(loss), MAP_rst, MRR_rst, summary

