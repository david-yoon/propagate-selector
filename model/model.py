#-*- coding: utf-8 -*-
"""
what    : detecting supporting sentences
data    : hotpot-qa
"""
import tensorflow as tf
from tensorflow.contrib.distributions import Categorical
from layers import add_GRU

from tensorflow.core.framework import summary_pb2
import numpy as np

from bilm import Batcher, BidirectionalLanguageModel, weight_layers, dump_token_embeddings

class GraphSentModel:
    
    def __init__(self,
                 dic_size,
                 params=None
                ):

        self.dic_size       = dic_size
        self.params         = params
        
        self.data_path = self.params.data_path
        
        self.encoder_inputs = []
        self.encoder_seq_length =[]
        self.y_labels =[]
        
        self.embed_dim = self.params.DIM_WORD_EMBEDDING
        
        # for global counter
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    

    def _create_placeholders(self):
        print ('[launch] placeholders')
        with tf.name_scope('text_placeholder'):
            
            if self.params.USE_CHAR_ELMO:
                # [batch, word_step], question
                self.batch_q     = tf.placeholder(tf.int32, shape=[self.params.batch_size, None, 50], name="batch_Q")

                # [batch,time_step, #sentence, word_step], sentences
                self.batch_s     = tf.placeholder(tf.int32, shape=[self.params.batch_size * self.params.MAX_SENTENCES, None, 50], name="batch_S")
            else:
                # [batch, word_step], question
                self.batch_q     = tf.placeholder(tf.int32, shape=[self.params.batch_size, None], name="batch_Q")

                # [batch,time_step, #sentence, word_step], sentences
                self.batch_s     = tf.placeholder(tf.int32, shape=[self.params.batch_size * self.params.MAX_SENTENCES, None], name="batch_S")
                
                
            # [batch, #sentences+1, #sentences+1], graph adj matrix (question; sentence)
            self.batch_g     = tf.placeholder(tf.float32, shape=[self.params.batch_size, self.params.MAX_SENTENCES+1, self.params.MAX_SENTENCES+1], name="batch_G")

            # [batch, #sentence], label
            self.batch_l     = tf.placeholder(tf.float32, shape=[self.params.batch_size, self.params.MAX_SENTENCES], name="batch_label")
            
            
            # [batch] - valid word step
            self.batch_len_q = tf.placeholder(tf.float32, shape=[self.params.batch_size], name="batch_len_Q")
            
            # [batch, #sentences] - valid word step
            self.batch_len_s = tf.placeholder(tf.float32, shape=[self.params.batch_size, self.params.MAX_SENTENCES], name="batch_len_S")
            
            # [batch], valid sentneces
            self.batch_num_s  = tf.placeholder(tf.int32, shape=[self.params.batch_size], name="batch_num_S")

            # drop out
            self.dr_prob     = tf.placeholder(tf.float32, name="dropout")
            self.dr_rnn_prob = tf.placeholder(tf.float32, name="dropout-rnn")
            
             # for using pre-trained embedding  ( dic_size -1 : ignore pad )
            self.embedding_placeholder = tf.placeholder(tf.float32, shape=[self.dic_size-1, self.embed_dim], name="embedding_placeholder")

           
    def _embed_ids(self):
        print ('[launch] embed_ids, use_ELMO')
        with tf.name_scope('text_embedding_layer'):
            
            # Build the biLM graph.
            if self.params.USE_CHAR_ELMO:
                bilm = BidirectionalLanguageModel(
                                                options_file = self.data_path + self.params.ELMO_OPTIONS,
                                                weight_file  = self.data_path + self.params.ELMO_WEIGHTS,
                                                max_batch_size = self.params.batch_size * self.params.MAX_SENTENCES
                                                )
            else:
                bilm = BidirectionalLanguageModel(
                                                options_file = self.data_path + self.params.ELMO_OPTIONS,
                                                weight_file  = self.data_path + self.params.ELMO_WEIGHTS,
                                                use_character_inputs=False,
                                                embedding_weight_file = self.data_path + self.params.ELMO_TOKEN,
                                                max_batch_size = self.params.batch_size * self.params.MAX_SENTENCES
                                                )

            # question
            self.embed_q_op = bilm(self.batch_q)
            self.elmo_q_output = weight_layers('output', self.embed_q_op, l2_coef=0.0)
            self.embed_q_inter = self.elmo_q_output['weighted_op']

            '''
            self.q_len_to_pad = self.params.MAX_LENGTH_Q - tf.reduce_max( self.batch_len_q ) -1
            self.q_len_to_pad = tf.maximum(self.q_len_to_pad, 0)
            self.embed_q = tf.pad( self.embed_q_inter, [[0,0], [0, self.q_len_to_pad], [0,0]] )
            '''
            self.embed_q = self.embed_q_inter
            
            # sentence
            self.embed_s_op = bilm(self.batch_s)
            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                self.elmo_s_output = weight_layers(
                    'output', self.embed_s_op, l2_coef=0.0
                )    
            self.embed_s_inter = self.elmo_s_output['weighted_op']
            
            self.s_len_to_pad = self.params.MAX_SENTENCES - tf.reduce_max( self.batch_len_s ) -1
            self.s_len_to_pad = tf.maximum(self.s_len_to_pad, 0)
            #self.embed_s = tf.pad( self.embed_s_inter, [[0,0], [0, self.s_len_to_pad], [0,0]] )
            
            # [batch_size, max_len (data dependent), elmo_embedding]
            self.embed_q = self.embed_q_inter
            
            # [batch_size, MAX_SENTENCES, max_len (data dependent), elmo_embedding]
            self.embed_s = tf.reshape( self.embed_s_inter, [self.params.batch_size, self.params.MAX_SENTENCES, -1, self.params.DIM_WORD_EMBEDDING] )
            
            
    def _preprocessing(self):
        print ('[launch] Preprocessing')
        with tf.name_scope('sentence_embedding_layer'):
            
            # sentece embedding for question
            self.embed_sent_q = tf.reduce_sum( self.embed_q , axis=1 )
            
            # sentece embedding for each sentence
            self.embed_sent_s = tf.reduce_sum( self.embed_s , axis=2)
            
            # sequence masking
            batch_num_s_mask = tf.sequence_mask( self.batch_num_s, self.params.MAX_SENTENCES, dtype=tf.float32 )
            self.embed_sent_s = tf.multiply( self.embed_sent_s, tf.reshape(batch_num_s_mask, [self.params.batch_size, self.params.MAX_SENTENCES, 1]) )
            
            # concat [q;list_s]
            self.embed_sent_q_s = tf.concat( [ tf.reshape(self.embed_sent_q, [self.params.batch_size, 1, self.embed_dim]) , self.embed_sent_s], axis=1 )
       
    
    def _preprocessing_RNN(self):
        print ('[launch] Preprocessing-RNN')
        with tf.name_scope('sentence_embedding_layer-RNN'):
            
            self.outputs_q, self.states_q = add_GRU(
                                                                        inputs= self.embed_q,
                                                                        inputs_len=self.batch_len_q,
                                                                        hidden_dim = self.params.DIM_SENTENCE_EMBEDDING,
                                                                        layers = 1,
                                                                        scope = 'encoding_RNN',
                                                                        reuse = False,
                                                                        dr_input_keep_prob  = self.dr_rnn_prob,
                                                                        dr_output_keep_prob = 1.0,
                                                                        is_bidir    = False,
                                                                        is_bw_reversed = True,
                                                                        is_residual = False
                                                                        )
            
            # shape [ batch_size, 1(question only), hidden_dim ]
            self.encoded_q = tf.reshape( self.states_q, [self.params.batch_size, 1, self.params.DIM_SENTENCE_EMBEDDING] )
                                      
            
            self.embed_s_flat     = tf.reshape( self.embed_s, [self.params.batch_size * self.params.MAX_SENTENCES, -1, self.embed_dim] )
            self.batch_len_s_flat = tf.reshape( self.batch_len_s, [self.params.batch_size * self.params.MAX_SENTENCES] )

            self.outputs_s, self.states_s = add_GRU(
                                                                        inputs = self.embed_s_flat,
                                                                        inputs_len = self.batch_len_s_flat,
                                                                        hidden_dim = self.params.DIM_SENTENCE_EMBEDDING,
                                                                        layers = 1,
                                                                        scope = 'encoding_RNN',
                                                                        reuse = True,
                                                                        dr_input_keep_prob  = self.dr_rnn_prob,
                                                                        dr_output_keep_prob = 1.0,
                                                                        is_bidir    = False,
                                                                        is_bw_reversed = True,
                                                                        is_residual = False
                                                                        )
            
            # shape [ batch_size, max_sent_len, hidden_dim ]
            self.encoded_s = tf.reshape( self.states_s, [self.params.batch_size, self.params.MAX_SENTENCES, self.params.DIM_SENTENCE_EMBEDDING] )
            
            
            self.embed_sent_q = self.encoded_q
            # concat [q;list_s]
            self.embed_sent_q_s = tf.concat( [self.encoded_q, self.encoded_s], axis=1 )
    
    
    def _attention(self, node, reuse=None, scope="attn_scope", v_name="default-attn"):
        
        with tf.variable_scope(name_or_scope=scope, reuse=reuse, initializer=tf.contrib.layers.xavier_initializer()):

            # edge:1, no-edge:  - float32.max
            mask_value = -tf.ones_like( self.batch_g ) * tf.float32.max
            mask_value = tf.multiply( mask_value, ( 1- self.batch_g ) )

            # luong attention with W matrix   A^T  W  A
            w_attn = tf.get_variable(
                'w-'+v_name,
                shape=[self.params.DIM_SENTENCE_EMBEDDING, self.params.DIM_SENTENCE_EMBEDDING],
                trainable = True
            )

            w_attn = tf.nn.dropout( w_attn, keep_prob=self.dr_prob )
            b_attn = tf.get_variable(
                'bias-'+v_name,
                shape=self.params.DIM_SENTENCE_EMBEDDING,
                initializer=tf.constant_initializer(0.),
                trainable = True
            )

            flat_node = tf.reshape( node, [self.params.batch_size * (self.params.MAX_SENTENCES+1), self.params.DIM_SENTENCE_EMBEDDING])
            tmp_attend= tf.matmul( flat_node, w_attn ) + b_attn
            tmp_attend= tf.reshape( tmp_attend, [self.params.batch_size, self.params.MAX_SENTENCES+1, self.params.DIM_SENTENCE_EMBEDDING] )

            attend = tf.matmul( tmp_attend, tf.transpose( node, perm=[0,2,1]) )

            # compute attention weight
            attend_mask = tf.multiply( attend, self.batch_g )
            attend_mask = tf.add( attend_mask, mask_value )
            attend_mask_norm = tf.nn.softmax( attend_mask, dim=2 )

            return attend_mask_norm

        
    '''
    matrix:  [batch, num_sent, in_dim]
    output: [batch, num_sent, out_dim]
    '''
    def _projection(self, matrix, in_dim, out_dim, reuse=None, scope="prj_scope", v_name="default-prj"):
        print ('[launch] W matrix prj (reuse, scope): ', reuse, scope)
        with tf.variable_scope(name_or_scope=scope, reuse=reuse, initializer=tf.contrib.layers.xavier_initializer()):

            w= tf.get_variable(
                'w-'+v_name,
                shape=[in_dim, out_dim],
                trainable = True
                )

            #w = tf.nn.dropout( w, keep_prob=self.dr_prob )
            b = tf.Variable(tf.zeros([out_dim]), name='bias-'+scope)

            flat_node = tf.reshape( matrix, [self.params.batch_size * (self.params.MAX_SENTENCES+1), in_dim])
            tmp_matmul= tf.matmul( flat_node, w ) + b
            tmp_matmul= tf.reshape( tmp_matmul, [self.params.batch_size, self.params.MAX_SENTENCES+1, out_dim] )

            
            tmp_matmul = tf.nn.tanh(tmp_matmul)
            
            return tmp_matmul
            
            
    def _MLP(self, in_tensor, out_dim, reuse=None, scope="mlp_scope", v_name="default-mlp"):
        print ('[launch] MLP (reuse, scope): ', reuse, scope)
        with tf.name_scope(scope):
            with tf.variable_scope(name_or_scope=scope, reuse=reuse, initializer=tf.contrib.layers.xavier_initializer()):

                initializer = tf.contrib.layers.xavier_initializer(
                                                                uniform=True,
                                                                seed=None,
                                                                dtype=tf.float32
                                                                )

                node_mlp_L1 = tf.contrib.layers.fully_connected( 
                                                            inputs = in_tensor,
                                                            num_outputs =  out_dim,
                                                            activation_fn = tf.nn.tanh,
                                                            normalizer_fn=None,
                                                            normalizer_params=None,
                                                            weights_initializer=initializer,
                                                            weights_regularizer=None,
                                                            biases_initializer=tf.zeros_initializer(),
                                                            biases_regularizer=None,
                                                            trainable=True,
                                                            reuse=reuse,
                                                            scope=scope+'mlp1'
                                                        )

                return node_mlp_L1
            
            
    def _add_aggregation(self, num_hop):

        # initial node value
        self.final_node = self.embed_sent_q_s
        
        # for analysis
        self.attn_analysis = []
        
        for i in range(num_hop):
            hop = i+1
            
            print ('[launch] aggregate information from neighbors hop:', hop)
            with tf.name_scope('graph_aggregation_hop_' + str(hop)):

                node = self.final_node

                # compute attention
                node_attention = self._attention( node, reuse=False, scope='L'+str(hop), v_name='attn' )

                # message passing from neighborhood
                node_next = tf.matmul( node_attention, node )
                node_next = tf.nn.tanh( node_next )

                if self.params.USE_PRE_NODE_INFO:
                    # contains original data
                    self.final_node = tf.concat( [self.final_node, node_next], axis=2)
                    
                    if self.params.PRJ_MLP:
                        print('[INFO] projection - MLP')
                        self.final_node = self._MLP( self.final_node, 
                                                self.params.DIM_SENTENCE_EMBEDDING, 
                                                reuse=False, 
                                                scope='L'+str(hop),
                                                v_name='prj-MLP'
                                               )
                    else:
                        print('[INFO] projection - W matrix')
                        self.final_node = self._projection( self.final_node,
                                                           self.params.DIM_SENTENCE_EMBEDDING * 2, 
                                                           self.params.DIM_SENTENCE_EMBEDDING,
                                                           reuse=False,
                                                           scope='L'+str(hop),
                                                           v_name='prj'
                                                          )
                    
                else:
                    self.final_node = node_next

                self.final_node_attention = node_attention
    
                self.attn_analysis.append( node_attention ) 
    
    """
    _input     : [batch, dim]
    _batch_seq : valid seq [batch]
    _max_len   : add pad till max_len
    """
    def masked(self, _input, _batch_seq, _max_len) :
        mask = tf.sequence_mask( lengths=_batch_seq, maxlen=_max_len, dtype=tf.float32 )
        mask_value = -tf.ones_like( mask ) * tf.float32.max
        mask_value = tf.multiply( mask_value, ( 1- mask ) )
        return _input + mask_value
    
    
    def _create_output_layers(self):
        print ('[launch] create output projection layer')
        
        with tf.name_scope('text_output_layer') as scope:
    
            # slice out the question part:  [ batch, [q;sent], dim]  -->  [ batch, [sent], dim ] 
            self.final_node_without_q = tf.slice( self.final_node, [0,1,0], [self.params.batch_size,self.params.MAX_SENTENCES, self.params.DIM_SENTENCE_EMBEDDING] )

            
            ##########################
            # W matrix compare Q^T W A
            ##########################
            if 1:
                # last q
                self.final_q = tf.slice( self.final_node, [0,0,0], [self.params.batch_size, 1, self.params.DIM_SENTENCE_EMBEDDING] )
                q = tf.reshape( self.final_q, [self.params.batch_size, self.params.DIM_SENTENCE_EMBEDDING, 1] )
            else:
                # original q
                q = tf.reshape( self.embed_sent_q, [self.params.batch_size, self.params.DIM_SENTENCE_EMBEDDING, 1] )
            
            w_output = tf.get_variable(
                'w-output',
                shape=[self.params.DIM_SENTENCE_EMBEDDING, self.params.DIM_SENTENCE_EMBEDDING],
                trainable = True
            )

            #w_output = tf.nn.dropout( w_attn, keep_prob=self.dr_prob )
            b_output = tf.get_variable(
                'bias-output',
                shape=self.params.DIM_SENTENCE_EMBEDDING,
                initializer=tf.constant_initializer(0.),
                trainable = True
            )
            
            flat_node = tf.reshape( self.final_node_without_q, [self.params.batch_size * (self.params.MAX_SENTENCES),self.params.DIM_SENTENCE_EMBEDDING])
            tmp_matmul= tf.matmul( flat_node, w_output ) + b_output
            tmp_matmul= tf.reshape( tmp_matmul, [self.params.batch_size, self.params.MAX_SENTENCES, self.params.DIM_SENTENCE_EMBEDDING] )
    
            sim_node_q = tf.matmul( tmp_matmul, q )
            
            self.final_output= tf.squeeze( sim_node_q )
            
            
        with tf.name_scope('loss') as scope:
            
            # label loss
            # Cross-entropy
            mask   = tf.sequence_mask( lengths=self.batch_num_s, maxlen=self.params.MAX_SENTENCES, dtype=tf.float32 )
            self.predic_masked = tf.multiply( mask, self.final_output )
            
            self.sigmoid_predic_masked = tf.sigmoid( self.predic_masked )

            self.y_masked      = self.masked(self.batch_l, self.batch_num_s, self.params.MAX_SENTENCES)

            self.loss_label = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.batch_l, logits=self.predic_masked)
            self.batch_loss_label = tf.multiply( mask, self.loss_label )
            self.rsum_batch_loss_label = tf.reduce_sum( self.batch_loss_label )

            
            # attn loss
            q_only = tf.squeeze(tf.slice( self.final_node_attention, [0,0,1], [self.params.batch_size, 1, self.params.MAX_SENTENCES] ))
            mask   = tf.sequence_mask( lengths=self.batch_num_s, maxlen=self.params.MAX_SENTENCES, dtype=tf.float32 )
            
            self.sigmoid_q_only = tf.sigmoid( q_only )
            
            self.loss_attn = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.batch_l, logits=q_only)
            self.masked_loss_attn = tf.multiply( self.loss_attn, mask)
            self.rsum_masked_loss_attn = tf.reduce_sum( self.masked_loss_attn )

            
            # Regularizer
            self.vars     = tf.trainable_variables()
            self.lossL2   = tf.add_n([ tf.nn.l2_loss(v) for v in self.vars if 'bias' not in v.name ]) * self.params.L2_LOSS_RATIO
            
            
            
            # TEST
            self.predic_prob = (self.sigmoid_predic_masked + self.sigmoid_q_only) / 2
            
            self.loss = self.rsum_batch_loss_label                                                                       # label loss
            self.loss = self.loss + self.params.LOSS_ATTN_RATIO*self.rsum_masked_loss_attn     # attn loss
            #self.loss = self.loss + self.lossL2                                                                               # L2 W loss
            
            
            
    def _create_optimizer(self):
        print ('[launch] create optimizer')
        
        with tf.name_scope('text_optimizer') as scope:
            
            opt_func = tf.train.AdamOptimizer(learning_rate=self.params.lr)
            gradients = opt_func.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_norm(grad, 5.0), var) for grad, var in gradients]
            self.optimizer = opt_func.apply_gradients(grads_and_vars=capped_gvs, global_step=self.global_step)

    
    def _create_summary(self):
        print ('[launch] create summary')
        
        with tf.name_scope('summary'):
            tf.summary.scalar('batch_loss', self.loss)
            self.summary_op = tf.summary.merge_all()
    
    
    def build_graph(self):
        
        self._create_placeholders()
        
        self._embed_ids()
        
        if (self.params.IS_RNN_EMBEDDING):
            print('[INFO] sentence dim != word dim', self.params.DIM_SENTENCE_EMBEDDING, self.params.DIM_WORD_EMBEDDING)
            self._preprocessing_RNN()
        else:
            self.params.DIM_SENTENCE_EMBEDDING = self.params.DIM_WORD_EMBEDDING
            print('[INFO] sentence dim = word dim', self.params.DIM_SENTENCE_EMBEDDING, self.params.DIM_WORD_EMBEDDING)
            self._preprocessing()

        self._add_aggregation(self.params.hop)

        self._create_output_layers()
        self._create_optimizer()
        self._create_summary()
        