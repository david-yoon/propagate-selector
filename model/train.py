#-*- coding: utf-8 -*-
"""
what    : detecting supporting sentences
data    : hotpot-qa
"""
import tensorflow as tf
import os
import time
import argparse
import datetime
import random
from tqdm import tqdm

from model import *
from process_data import *
from evaluation import *

from params import *

# for training         
def train_step(sess, model, batch_gen):
    
    
    list_q, list_s, list_g, list_l, list_len_q, list_len_s, list_num_s = batch_gen.get_batch(
                                data=batch_gen.train,
                                batch_size=model.params.batch_size,
                                is_test=False
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

    input_feed[model.dr_prob]     = model.params.dr
    input_feed[model.dr_rnn_prob] = model.params.dr_rnn
    
    _, summary = sess.run([model.optimizer, model.summary_op], input_feed)
    
    return summary

    
def train_model(model, params, batch_gen, num_train_steps, valid_freq, is_save=0, graph_dir_name='default', pre_model=""):
    
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    summary = None
    val_summary = None
    
    with tf.Session(config=config) as sess:

        writer = tf.summary.FileWriter('./graph/'+graph_dir_name, sess.graph)
        sess.run(tf.global_variables_initializer())
                
        early_stop_count = params.MAX_EARLY_STOP_COUNT
        
        # if exists check point, starts from the check point
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pre_model))
        if ckpt and ckpt.model_checkpoint_path:
            print ('[load] pre_model check point!!!')
            saver.restore(sess, ckpt.model_checkpoint_path)

        initial_time = time.time()
        
        max_MAP = 0.0
        max_MRR = 0.0
        
        best_test_MAP = 0.0
        best_test_MRR = 0.0
        
        train_MAP_at_best_dev= 0.0
        
        
        for index in range(num_train_steps):

            try:
                # run train 
                summary = train_step(sess, model, batch_gen)
                writer.add_summary( summary, global_step=model.global_step.eval() )
                
            except Exception as e:
                print ("excepetion occurs in train step", e)
                pass
                
            
            # run validation
            if (index + 1) % valid_freq == 0:
                
                dev_loss, dev_MAP, dev_MRR, dev_summary = run_test(sess=sess,
                                                         model=model, 
                                                         batch_gen=batch_gen,
                                                         data=batch_gen.dev)
                
                writer.add_summary( dev_summary, global_step=model.global_step.eval() )
                

                end_time = time.time()

                if index > params.CAL_ACCURACY_FROM:

                    if ( dev_MAP > max_MAP ):
                        max_MAP = dev_MAP
                        max_MRR = dev_MRR

                        # save best result
                        if is_save is 1:
                            saver.save(sess, 'save/' + graph_dir_name + '/', model.global_step.eval() )
                            
                        elif dev_MAP > float(params.QUICK_SAVE_THRESHOLD):
                            saver.save(sess, 'save/' + graph_dir_name + '/', model.global_step.eval() )
                            
                        
#                         test_loss, test_MAP, test_MRR, _ = run_test(sess=sess,
#                                                                     model=model,
#                                                                     batch_gen=batch_gen,
#                                                                     data=batch_gen.test)
                        test_loss, test_MAP, test_MRR = 0, 0, 0
                        
#                         train_loss, train_MAP, train_MRR, _ = run_test(sess=sess,
#                                                                      model=model,
#                                                                      batch_gen=batch_gen,
#                                                                      data=batch_gen.train)
                        train_loss, train_MAP, train_MRR = 0, 0, 0

                        
                        early_stop_count = params.MAX_EARLY_STOP_COUNT
                        
                        if test_MAP > best_test_MAP: 
                            best_test_MAP = test_MAP
                            best_test_MRR = test_MRR
                            
                        train_MAP_at_best_dev = train_MAP
                        
                    else:
                        # early stopping
                        if early_stop_count == 0:
                            print ("early stopped")
                            break
                             
                        test_MAP = 0.0
                        test_MRR = 0.0
                        train_MAP= 0.0
                        early_stop_count = early_stop_count -1
                        
                        
                    print (str( int((end_time - initial_time)/60) ) + " mins" + \
                      " step/seen/itr: " + str( model.global_step.eval() ) + "/ " + \
                      str( model.global_step.eval() * model.params.batch_size ) + "/" + \
                      str( round( model.global_step.eval() * model.params.batch_size / float(len(batch_gen.train)), 2)  ) + \
                      "\tdev: " + '{:.3f}'.format(dev_MAP) + \
                      " " + '{:.3f}'.format(dev_MRR) + \
                      "  test: " + '{:.3f}'.format(test_MAP) + \
                      " " + '{:.3f}'.format(test_MRR) + \
                      "  train: " + '{:.3f}'.format(train_MAP) + \
                      "  loss(dev): " + '{:.4f}'.format(dev_loss))
                
        writer.close()

        
        train_loss, train_MAP, train_MRR, _ = run_test(sess=sess,
                                                       model=model,
                                                       batch_gen=batch_gen,
                                                       data=batch_gen.train)
        
        print ("best result (MAP/MRR) \n" + \
               "dev   : " +\
                    str('{:.3f}'.format(max_MAP)) + '\t'+ \
                    str('{:.3f}'.format(max_MRR)) + '\n'+ \
               "test  : " +\
                    str('{:.3f}'.format(best_test_MAP)) + '\t' + \
                    str('{:.3f}'.format(best_test_MRR)) + '\n' + \
               "train : " +\
                    str('{:.3f}'.format(train_MAP)) + '\t' + \
                    str('{:.3f}'.format(train_MRR)) + \
                    '\n')

    
        # result logging to file
        with open('./TEST_run_result.txt', 'a') as f:
            f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + '\t' + \
                    str( int((end_time - initial_time)/60) ) + "-mins" + '\t' + \
                    batch_gen.data_path.split('/')[-2] + '\t' + \
                    graph_dir_name + '\t' + \
                    str('{:.3f}'.format(max_MAP)) + '\t'+ \
                    str('{:.3f}'.format(max_MRR)) + '\t'+ \
                    str('{:.3f}'.format(best_test_MAP)) + '\t' + \
                    str('{:.3f}'.format(best_test_MRR)) + '\t' + \
                    str('{:.3f}'.format(train_MAP)) + '\t' + \
                    str('{:.3f}'.format(train_MRR)) + \
                    '\n')
    

def create_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        
        
def main(params):
    
    create_dir('save/')
    
    if params.is_save is 1:
        create_dir('save/'+ params.graph_prefix )
    
    create_dir('graph/')
    create_dir('graph/'+ params.graph_prefix)

    batch_gen = ProcessData(params)
    
    model = GraphSentModel(
                            dic_size=batch_gen.dic_size,
                            params = params
                        )

    model.build_graph()
    
    valid_freq = int( len(batch_gen.train) * params.EPOCH_PER_VALID_FREQ / float(params.batch_size)  ) + 1
    
    print ("[Info] valid freq: ",        str(valid_freq))
    print ('[Info] MAP from wang: ',     params.USE_WANG_MAP)
    print ('[Info] use RNN: ',         params.IS_RNN_EMBEDDING)
    print ('[Info] embedding train: ',   params.EMBEDDING_TRAIN)
    
    train_model(model, params, batch_gen, params.num_train_steps, valid_freq, params.is_save, params.graph_prefix, params.pre_model)
    
if __name__ == '__main__':
    
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', type=str, default='hotpot')
    
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--dr', type=float, default=1.0)
    p.add_argument('--dr_rnn', type=float, default=1.0)

    p.add_argument('--sdim', type=int, default=100)
    p.add_argument('--hop', type=int, default=1)
    
    p.add_argument('--num_train_steps', type=int, default=10000)
    p.add_argument('--is_save', type=int, default=0)
    p.add_argument('--graph_prefix', type=str, default="default")
    p.add_argument('--pre_model', type=str, default="default")
    
    
    args = p.parse_args()
    
    if args.corpus == ('hotpot'): 
        params    = Params()
        params.data_path = '../data/target_hotpot/'
        graph_name = 'HOTPOT'
    
    elif args.corpus == ('hotpot_small'): 
        params    = Params_small()
        params.data_path = '../data/target_hotpot_small/'
        graph_name = 'HOTPOT_S'
        
        
    params.batch_size = args.batch_size
    params.lr              = args.lr
    params.dr             = args.dr
    params.dr_rnn        = args.dr_rnn
    params.DIM_SENTENCE_EMBEDDING = args.sdim
    params.hop           = args.hop
    params.num_train_steps = args.num_train_steps
    params.is_save       = args.is_save
    params.gaph_prefix = args.graph_prefix
    params.pre_model  = args.pre_model

    embed_fix = 0
    if params.EMBEDDING_TRAIN == False:
        embed_fix = 1
    
    graph_name = args.graph_prefix + \
                    '_b' + str(params.batch_size) + \
                    '_fix' + str(embed_fix) + \
                    '_s-dim' + str(params.DIM_SENTENCE_EMBEDDING) + \
                    '_hop' + str(params.hop) + \
                    '_dr' + str(params.dr) + \
                    '_dr-rnn' + str(params.dr_rnn)
    
    graph_name = graph_name + '_' + datetime.datetime.now().strftime("%m-%d-%H-%M")
    params.graph_prefix = graph_name
    
    print('[INFO] data:\t\t\t', params.data_path)
    print('[INFO] batch:\t\t\t', params.batch_size)
    print('[INFO] hop:\t\t\t', params.hop)
    print('[INFO] w-dim:\t\t\t', params.DIM_WORD_EMBEDDING)
    print('[INFO] s-dim:\t\t\t', params.DIM_SENTENCE_EMBEDDING)
    
    print('[INFO] IS_RNN_EMBEDDING:\t', params.IS_RNN_EMBEDDING)
    print('[INFO] USE_PRE_NODE_INFO:\t', params.USE_PRE_NODE_INFO)
    print('[INFO] PRJ_MLP:\t\t\t', params.PRJ_MLP)
    
    print('[INFO] EDGE_PASSAGE_PASSAGE:\t', params.EDGE_PASSAGE_PASSAGE)
    print('[INFO] EDGE_SENTENCE_QUESTION:\t', params.EDGE_SENTENCE_QUESTION)
    print('[INFO] EDGE_SELF:\t\t', params.EDGE_SELF)
    print('[INFO] EDGE_WITHIN_PASSAGE:\t', params.EDGE_WITHIN_PASSAGE)
    
    print('[INFO] lr:\t\t\t', params.lr)
    print('[INFO] dr:\t\t\t', params.dr)
    print('[INFO] dr_rnn:\t\t\t', params.dr_rnn)
    print('[INFO] IS_DEBUG:\t\t', params.IS_DEBUG)
    print('[INFO] is_save:\t\t\t', params.is_save)
    
    main(
        params  = params
        )