
# coding: utf-8

# ## create ELMO embedding

# In[2]:


import tensorflow as tf
import os
import pickle
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, dump_token_embeddings


# In[3]:


# # small 

# try:
#     file = '../data/processed/hotpot/ELMO_options.json'
#     if os.lstat(file):
#         os.remove(file)
#         os.system('ln -s ../../ELMO_pretrain/elmo_2x1024_128_2048cnn_1xhighway_options.json ../data/processed/hotpot/ELMO_options.json')
# except:
#     os.system('ln -s ../../ELMO_pretrain/elmo_2x1024_128_2048cnn_1xhighway_options.json ../data/processed/hotpot/ELMO_options.json')
    
# try:
#     file = '../data/processed/hotpot/ELMO_weights.hdf5'
#     if os.lstat(file):
#         os.remove(file)
#         os.system('ln -s ../../ELMO_pretrain/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5 ../data/processed/hotpot/ELMO_weights.hdf5')
# except:
#     os.system('ln -s ../../ELMO_pretrain/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5 ../data/processed/hotpot/ELMO_weights.hdf5')


# In[10]:


# origianl 5.5B

try:
    file = '../data/processed/hotpot/ELMO_options.json'
    if os.lstat(file):
        os.remove(file)
        os.system('ln -s ../../ELMO_pretrain/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json ../data/processed/hotpot/ELMO_options.json')
except:
    os.system('ln -s ../../ELMO_pretrain/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json ../data/processed/hotpot/ELMO_options.json')
    
try:
    file = '../data/processed/hotpot/ELMO_weights.hdf5'
    if os.lstat(file):
        os.remove(file)
        os.system('ln -s ../../ELMO_pretrain/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5 ../data/processed/hotpot/ELMO_weights.hdf5')
except:
    os.system('ln -s ../../ELMO_pretrain/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5 ../data/processed/hotpot/ELMO_weights.hdf5')


# In[ ]:





# In[ ]:





# In[11]:


with open('../data/processed/hotpot/vocab.txt') as f:
    voca = f.readlines()
    voca = [x.strip() for x in voca]
print ('original dic size = ', len(voca))


# In[12]:


with open('../data/processed/hotpot/vocab-elmo.txt', 'w') as f:
    for tok in voca:
        f.write( tok + '\n')
    f.write( '<S>' + '\n')
    f.write( '<\S>' + '\n') 


# In[ ]:





# In[13]:


# Location of pretrained LM.  Here we use the test fixtures.
datadir = '../data/processed/hotpot/'
options_file = os.path.join(datadir, 'ELMO_options.json')
weight_file = os.path.join(datadir, 'ELMO_weights.hdf5')
vocab_file =  os.path.join(datadir, 'vocab-elmo.txt')
token_embedding_file =  os.path.join(datadir, 'ELMO_token_embeddings.hdf5')


# In[14]:


# Dump the token embeddings to a file. Run this once for your dataset.
dump_token_embeddings(
    vocab_file, options_file, weight_file, token_embedding_file
)
tf.reset_default_graph()


# In[ ]:





# In[ ]:





# In[ ]:




