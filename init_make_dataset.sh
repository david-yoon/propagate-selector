########################################
# Envionments & raw dataset download
#######################################

pip install --upgrade pip
pip install -r requirements.txt
python -m nltk.downloader punkt

mkdir data
cd data

mkdir raw
cd raw


########################################
# download HotpotQA dataset
########################################

mkdir hotpot
cd hotpot

wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json

cd ../../../


########################################
# ELMO model & binary download
########################################

git clone git@github.com:allenai/bilm-tf.git

cd data
mkdir ELMO_pretrain
cd ELMO_pretrain

# small
#wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5
#wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json

# medium
# wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5
# wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json

# original 5.5B
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json

cd ../../


########################################
# Preprocessing
########################################

cd preprocessing
python 01-create_voca.py
python 02-raw_to_index.py
python 03-prepare-ELMO.py

echo completed
