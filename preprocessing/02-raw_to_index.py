#!/usr/bin/env python
# coding: utf-8

# ## 01 - text to index

# In[1]:


from tqdm import tqdm
import ujson as json
import nlp_vocab
import pickle
import file_util


# In[2]:


IS_LOWERCASE = True


# In[ ]:





# In[3]:


with open('../data/raw/hotpot/hotpot_train_v1.1.json', 'rb') as f:
    train = json.load(f)
    
with open('../data/raw/hotpot/hotpot_dev_distractor_v1.json', 'rb') as f:
    dev_distractor = json.load(f)

with open('../data/raw/hotpot/hotpot_dev_fullwiki_v1.json', 'rb') as f:
    dev_wiki = json.load(f)


# In[ ]:





# In[5]:


# nltk
import nltk
from nltk.tokenize import word_tokenize

def sent2text(sent):
    sent = word_tokenize(sent.strip())
    if IS_LOWERCASE:
        sent = [x.lower().strip() for x in sent]
    else:
        sent = [x.strip() for x in sent]
    return sent


# In[ ]:





# In[6]:


def raw2text(raw_data):
    list_data = []

    for data in tqdm(raw_data):
        question = ''
        sentence = []
        label = [0] * 200
        passage_index = []

        question = data['question']
        question = sent2text(question.strip())

        for context in data['context']:

            passage_index.append( len(sentence) )
            
            # check supporting facts
            for sf in data['supporting_facts']:
                if context[0] == sf[0]:
                    if int(sf[1]) > 90:
                        ("")
                    else:
                        label[ (len(sentence)+int(sf[1])) ] = 1
                    
            # add sentence from passage
            sentence.extend( context[1] )

        sentence = [sent2text(x) for x in sentence]

        list_data.append( [question, sentence, passage_index, label] )
        
    return list_data


# In[ ]:





# In[7]:


dev_output = raw2text(dev_distractor)

with open('../data/processed/hotpot/dev.pkl', 'wb') as f:
    pickle.dump(dev_output, f)


# In[8]:


train_output = raw2text(train)

with open('../data/processed/hotpot/train.pkl', 'wb') as f:
    pickle.dump(train_output, f)


# In[9]:


# dev_wiki_output = raw2text(dev_wiki)

# with open('../data/processed/hotpot/dev-wiki.pkl', 'wb') as f:
#     pickle.dump(dev_wiki_output, f)


# In[ ]:





# In[ ]:





# In[10]:


with open('../data/processed/hotpot/debug.pkl', 'wb') as f:
    pickle.dump(dev_output[:200], f)


# In[ ]:





# In[ ]:




