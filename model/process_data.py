#-*- coding: utf-8 -*-
"""
what    : detecting supporting sentences
data    : hotpot-qa
"""
import numpy as np
import pickle
import random
import time

from bilm import Batcher, TokenBatcher

class ProcessData:

    def __init__(self, params):

        self.data_path = params.data_path
        self.params = params

        if params.IS_DEBUG:
            print('debug mode')
            # load data for debugging
            self.train = self.load_data( self.data_path + self.params.DATA_DEBUG )
            self.dev   = self.load_data( self.data_path + self.params.DATA_DEBUG )
            self.test  = self.load_data( self.data_path + self.params.DATA_DEBUG )
        
        else:
            # load data
            self.train = self.load_data( self.data_path + self.params.DATA_TRAIN )
            self.dev   = self.load_data( self.data_path + self.params.DATA_DEV )
            self.test  = self.load_data( self.data_path + self.params.DATA_TEST )
       
        # batcher for ELMo
        if self.params.USE_CHAR_ELMO:
            print('[INFO] character-level ELMo')
            self.batcher      = Batcher( self.data_path + self.params.DIC, 50 )
        else:
            print('[INFO] cached-token-level ELMo')
            self.batcher      = TokenBatcher( self.data_path + self.params.DIC )

        self.dic_size = 0
        with open( self.data_path + self.params.DIC, 'r' ) as f:
            self.dic      = f.readlines()
            self.dic      = [ x.strip() for x in self.dic ]
            self.dic_size = len( self.dic )
            
        print ('[completed] load data, dic_size: ', self.dic_size)
        
        
    def load_data(self, file_path):
     
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
            
        print ('load data : ', file_path, len(dataset))

        return dataset
        
        
    def get_glove(self):
        
        print ('[load glove] ' + self.params.GLOVE)
        return np.load( self.data_path + self.params.GLOVE )

    
    """
        inputs: 
            data         : data to be processed (train/dev/test)
            batch_size   : mini-batch size
            is_test      : True, inference stage (ordered input)  (default : False)
            start_index  : start index of mini-batch (will be used when is_test==True)

        return:
            list_q       : [batch, time_step(==MAX_LENGTH_Q)], questions
            list_s       : [batch, MAX_SENTENCES, time_step(==MAX_LENGTH_S)], sentences
            list_graph   : [batch, MAX_SENTENCES+1, MAX_SENTENCES+1], adjacency matrix of graph [question ; sentecens]
            list_l       : [batch], labels
            
            list_len_q   : [batch]. vaild sequecne length
            list_len_s   : [batch, MAX_SENTENCES]. vaild sequecne length
            list_num_s   : [batch], valid number of sentences
    """
    def get_batch(self, data, batch_size, is_test=False, start_index=0):

        list_q, list_s, list_graph, list_l = [], [], [], []        
        list_len_q, list_len_s , list_num_s = [], [], []
        
        
        index = start_index
        
        # Get a random batch of encoder and encoderR inputs from data,
        # pad them if needed

        for _ in range(batch_size):
            
            tmp_list_s, tmp_list_len_s, tmp_list_l = [], [], []
            tmp_list_graph = np.zeros([self.params.MAX_SENTENCES+1, self.params.MAX_SENTENCES+1], dtype=np.int32)

            if is_test is False:
                # train case -  random sampling
                q, s, i, l = random.choice(data)
                s = s[:self.params.MAX_SENTENCES]
                i = [x for x in i if x< self.params.MAX_SENTENCES ]
                
            else:
                if index >= len(data):
                    # dummy data ( use index 0 data )
                    q, s, i, l = data[0]  # dummy for batch - will not be evaluated
                    s = s[:self.params.MAX_SENTENCES]
                    i = [x for x in i if x< self.params.MAX_SENTENCES ]
                    index += 1
                else:
                    # real data
                    q, s, i, l = data[index]
                    s = s[:self.params.MAX_SENTENCES]
                    i = [x for x in i if x< self.params.MAX_SENTENCES ]
                    index += 1

            tmp_q = q.copy()
            tmp_q = tmp_q[:(self.params.MAX_LENGTH_Q-3)]     # [make room] elmo will add <S>, 0 (last padding), we added <\S>
            tmp_q.append('<\\S>')
            
            list_q.append( tmp_q )
            list_len_q.append(  min(len(tmp_q)-1,self.params.MAX_LENGTH_Q) )  # ignore special token </S>
            
            # add data as many as MAX_ANSWERS
            for tmp_i in range(self.params.MAX_SENTENCES):

                # real data
                if tmp_i < len(s):
                    # Add pad to data & Calculate seq_length (for later use)
                    # negative case will not generate pad array
                    
                    tmp_s = s[tmp_i].copy()
                    tmp_s = tmp_s[:(self.params.MAX_LENGTH_S-3)]     # elmo will add <S>, 0 (last padding), we added <\S>
                    tmp_s.append('<\\S>')
                    
                    tmp_list_s.append( tmp_s )
                    tmp_list_len_s.append( min(len(tmp_s)-1,self.params.MAX_LENGTH_S) )  # ignore special token </S>  
                    
                    tmp_list_l.append( int(l[tmp_i]) )
                
                else:
                    # Add dummy data (data from index 0)
                    tmp_s = s[0].copy()
                    tmp_s = tmp_s[:(self.params.MAX_LENGTH_S-3)]     # elmo will add <S>, 0 (last padding), we added <\S>
                    tmp_s.append('<\\S>')
                    
                    tmp_list_s.append( tmp_s )
                    #tmp_list_len_s.append( min(len(tmp_s)-1,self.params.MAX_LENGTH_S) )  # ignore special token </S>  
                    tmp_list_len_s.append( 0 )  # ignore special token </S>  
                    tmp_list_l.append( int(l[0]) )
            
            # build graph adj matrix [question;sentences]

            # edge btw question and each sentence ( +1 for question )
            # [ max_sentence +1, max_sentence +1 ] 
            tmp_list_graph[0][:len(s)+1] = 1
            q_offset = 1
           
            i.append( len(s) )         # i = index of starting sentence in passage  <- append total length of the sentence
            start_s, end_s = -1, -1
            
            for sen_index in i:
                start_s = end_s
                end_s   = sen_index

                # skipping initial condition 
                if(start_s != -1):

                    tmp_same_passage = []                                                           # for checking the index of sentence in the same passage
                    # edge btw sentences in the same passage
                    for tmp_i in range(start_s, end_s):

                        if self.params.EDGE_SENTENCE_QUESTION:
                            tmp_list_graph[tmp_i + q_offset][0] = 1                            # edge with question
                            
                        if self.params.EDGE_SELF:
                            tmp_list_graph[tmp_i + q_offset][tmp_i+ q_offset] = 1        # self edge

                        # edge with neighbor within passage
                        if self.params.EDGE_WITHIN_PASSAGE == 0:
                            if (tmp_i+1 != end_s):
                                tmp_list_graph[tmp_i + q_offset][tmp_i+1 + q_offset] = 1   # edge with neighbor
                                tmp_list_graph[tmp_i+1 + q_offset][tmp_i + q_offset] = 1   # edge with neighbor
                                
                        tmp_same_passage.append(tmp_i + q_offset)

                        
                    # edge fully-connected within passage
                    if self.params.EDGE_WITHIN_PASSAGE == 1:
                        for sent_idx in tmp_same_passage:
                            copy_tmp_same_passage = list(tmp_same_passage)
                            copy_tmp_same_passage.remove(sent_idx)                     # self-connection is defined from params.EDGE_SELF
                            tmp_list_graph[sent_idx][copy_tmp_same_passage] = 1   # q_offset is already applied
                            
                            
                    # edge fully-connected among first sentence of the passage
                    if self.params.EDGE_PASSAGE_PASSAGE:
                        tmp_passage_index = list(i)[:-1]                                                     # remove last index
                        tmp_passage_index = [ (x+q_offset) for x in tmp_passage_index ]    # q offset
                        
                        for passage_idx in tmp_passage_index:
                            copy_tmp_passage_index = list(tmp_passage_index)
                            copy_tmp_passage_index.remove(passage_idx)                          # self-connection is defined from params.EDGE_SELF
                            tmp_list_graph[passage_idx][copy_tmp_passage_index] = 1        # q_offset is already applied
                            
                            
            list_graph.append(tmp_list_graph)
            list_s.append(tmp_list_s)
            list_len_s.append(tmp_list_len_s)
            list_l.append(tmp_list_l)
            list_num_s.append(len(s))
            
        list_s_reshape = np.reshape( list_s, (self.params.batch_size * self.params.MAX_SENTENCES) )

        elmo_list_q = self.batcher.batch_sentences(list_q)
        elmo_list_s = self.batcher.batch_sentences(list_s_reshape) 
           
        return elmo_list_q, elmo_list_s, list_graph, list_l, list_len_q, list_len_s, list_num_s