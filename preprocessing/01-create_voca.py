
# coding: utf-8

# In[1]:


import ujson as json
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
import file_util
import pickle


# In[2]:


IS_LOWERCASE = True
DIC_MINCUT_FREQ = 12    # less equal than this frequency will not be considered


# In[ ]:





# In[3]:


with open('../data/raw/hotpot/hotpot_train_v1.1.json', 'rb') as f:
    data_train = json.load(f)
    
with open('../data/raw/hotpot/hotpot_dev_distractor_v1.json', 'rb') as f:
    data_dev_distractor = json.load(f)

with open('../data/raw/hotpot/hotpot_dev_fullwiki_v1.json', 'rb') as f:
    data_dev_wiki = json.load(f)


# In[ ]:





# #### nltk

# In[4]:


def add_sent_to_dic(dic, sent):
    list_tokent = word_tokenize(sent)
    for token in list_tokent:
        if IS_LOWERCASE:
            token = token.lower().strip()
        else:
            token = token.strip() 
            
        if token in dic:
            dic[token] += 1
        else:
            dic[token] = 1


# #### spacy

# In[5]:


# import spacy
# nlp = spacy.load('en', disable=['tagger', 'parser', 'ner', 'textcat'])

# def add_sent_to_dic(dic, sent):
#     list_tokent = nlp(sent)
#     for token in list_tokent:
#         if IS_LOWERCASE:
#             token = token.text.lower().strip()
#         else:
#             token = token.text.strip() 
            
#         if token in dic:
#             dic[token] += 1
#         else:
#             dic[token] = 1


# In[ ]:





# In[6]:


def create_dic(dic, data):
    
    for sample in tqdm(data):
        add_sent_to_dic(dic, sample['question'])

        for context in sample['context']:
            add_sent_to_dic(dic, context[0])    # title of the passage

            for sentence in context[1]:
                add_sent_to_dic(dic, sentence)  # sentence in the passage


# In[ ]:





# In[ ]:


dic = {}
create_dic(dic, data_train)
create_dic(dic, data_dev_distractor)
create_dic(dic, data_dev_wiki)
print('dic size:' + str(len(dic)))


# In[ ]:


file_util.create_folder('../data/processed/hotpot')


# In[ ]:





# ## reducing dictionary

# In[ ]:


dic_ori = dic


# In[ ]:


from nlp_util import apply_mincut_lessequal_than
dic_mincut = apply_mincut_lessequal_than(dic_ori, DIC_MINCUT_FREQ)


# #### nltk

# In[20]:


with open('../data/processed/hotpot/vocab.txt', 'w') as f:
    f.write('_PAD_' + '\n')
    f.write('_UNK_' + '\n')
    
    for key in dic_mincut.keys():
        f.write(key + '\n')
        
with open('../data/processed/hotpot/vocab.txt', 'r') as f:
    read_voca = f.readlines()
print('voca size including _PAD_ _UNK_: ' + str(len(read_voca)))


# #### spacy

# In[21]:


# with open('../data/processed/hotpot/vocab.txt', 'w') as f:
#     f.write('_PAD_' + '\n')
#     f.write('_UNK_' + '\n')
    
#     for key in dic_mincut.keys():
#         f.write(key + '\n')
        
# with open('../data/processed/hotpot/vocab-spacy.txt', 'r') as f:
#     read_voca = f.readlines()
# print('voca size including _PAD_ _UNK_: ' + str(len(read_voca)))


# In[ ]:





# In[22]:


print('completed')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




