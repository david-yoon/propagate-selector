propagate-selector
------------------------------------------------------------

This repository contains the source code & data corpus used in the following paper,

Requirements
-------------

> tensorflow==1.10 (tested on cuda-9.2, cudnn-7.6.5) <br>
> python==3.5 <br>
> tqdm==4.31.1 <br>
> nltk==3.3 <br>
> h5py==2.8.0 <br>
> ujson==1.35 <br>



download dataset & preprocessing
-------------

- download "HotpotQA" to ./data/raw/hotpot/ <br>
- clone ELMo repository and download pretrained models
```bash
sh init_make_dataset.sh
```
- processed file (train/dev.pkl)
- [#samples, 4]
	- 0: question [#token]
	- 1: list (sentences) [#sentence, #token]
	- 2: index of first sentence of each passage [#sentence]
	- 3: label [#sentence]


Training Phase
-------------
- run reference script in "./model" folder
- results will be displayed in console <br>
- results will be saved to "./model/TEST_run_result.txt" <br>
```bash
sh reference_script_train.sh
```


hyper parameters
-------------
- major parameters : edit from "./model/reference_script_train.sh" <br>
- other parameters : edit from "./model/params.py" <br>


cite
-------------
- Please cite our paper, when you use our code | dataset | model