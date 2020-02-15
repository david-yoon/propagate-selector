class Params ():

    def __init__(self):
        print('HotpotQA')
        self.name = "HotpotQA"
    
    
    ################################
    #     dataset
    ################################     
    data_path = ''
    
    DATA_TRAIN  = 'train.pkl'
    DATA_DEV     = 'dev.pkl'
    DATA_TEST    = 'dev.pkl'
    DATA_DEBUG = 'debug.pkl'

    DIC          = 'vocab-elmo.txt'
    
    ELMO_TOKEN   = 'ELMO_token_embeddings.hdf5'
    ELMO_OPTIONS = 'ELMO_options.json'
    ELMO_WEIGHTS = 'ELMO_weights.hdf5'

    
    ################################
    #     training
    ################################
    
    batch_size             = 128    
    lr                     = 0.001
    dr                     = 1.0
    dr_rnn                 = 1.0
    hop                  = 1
    num_train_steps      = 10000
    is_save              = 0
    graph_prefix       = "default"
    pre_model          = "default"
    
    EMBEDDING_TRAIN        = True     # True == (fine-tuning)
    CAL_ACCURACY_FROM      = 0         # run iteration without excuting validation
    MAX_EARLY_STOP_COUNT   = 4
    EPOCH_PER_VALID_FREQ   = 0.2
    
    QUICK_SAVE_THRESHOLD = 0.70
    
    ################################
    #     model
    ################################
    # train wordQ/wordS/Sent  max 108/394/144  avg 17.9/22.3/40.9    std 9.5/10.9/11.2
    # dev   wordQ/wordS/Sent  max  46/318/147  avg 15.8/22.4/41.3    std 5.5/10.9/11.2 
    MAX_LENGTH_Q = 66   # 110
    MAX_LENGTH_S = 76   # 250
    MAX_SENTENCES= 95   # 150  
    PAD_INDEX    = 0
    DIM_WORD_EMBEDDING = 1024
    DIM_SENTENCE_EMBEDDING = 1024
    
    IS_RNN_EMBEDDING  = True
    USE_PRE_NODE_INFO = True
    PRJ_MLP           = False
    
    USE_CHAR_ELMO     = False
    LOSS_ATTN_RATIO   = 1e+0
    L2_LOSS_RATIO     = 2e-4
    
        
    ################################
    #     topology
    ################################
    EDGE_PASSAGE_PASSAGE = True
    EDGE_SENTENCE_QUESTION = True
    EDGE_SELF = True
    EDGE_WITHIN_PASSAGE = 1            # 0: neighbor, 1:fully-connected
    
    ################################
    #     MEASRE
    ################################
    USE_WANG_MAP = True                # calculate MAP from wang's implementation


    ################################
    #     ETC
    ################################
    IS_DEBUG = False                    # load sample data for debugging
    IS_SAVE_RESULTS_TO_FILE = False
    
    
    
class Params_small (Params):
    
    def __init__(self):
        print('HotpotQA_small')
        self.name = "HotpotQA_small"
        
    ################################
    #     model
    ################################
    # train wordQ/wordS/Sent  max 108/394/144  avg 17.9/22.3/40.9    std 9.5/10.9/11.2
    # dev   wordQ/wordS/Sent  max  46/318/147  avg 15.8/22.4/41.3    std 5.5/10.9/11.2 
    MAX_LENGTH_Q = 66   # 110
    MAX_LENGTH_S = 76   # 250
    MAX_SENTENCES= 95   # 150  
    PAD_INDEX    = 0
    DIM_WORD_EMBEDDING = 256
    DIM_SENTENCE_EMBEDDING = 512
    
    IS_RNN_EMBEDDING  = True
    USE_PRE_NODE_INFO = True
    PRJ_MLP           = False
    
    USE_CHAR_ELMO     = False
    LOSS_ATTN_RATIO   = 1e+0
    L2_LOSS_RATIO     = 2e-4
    
        
    ################################
    #     topology
    ################################
    EDGE_PASSAGE_PASSAGE = True
    EDGE_SENTENCE_QUESTION = True
    EDGE_SELF = True
    EDGE_WITHIN_PASSAGE = 1            # 0: neighbor, 1:fully-connected
    
    
    ################################
    #     ETC
    ################################
    IS_DEBUG = False                    # load sample data for debugging
    IS_SAVE_RESULTS_TO_FILE = False