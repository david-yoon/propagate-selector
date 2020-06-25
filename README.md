# propagate-selector

This repository contains the source code & data corpus used in the following paper,

**Propagate-Selector: Detecting Supporting Sentences for Question Answering via Graph Neural Networks**, LREC-20, <a href="https://www.aclweb.org/anthology/2020.lrec-1.664">[paper]</a>

-------------

### [requirements]
	tensorflow==1.10 (tested on cuda-9.2, cudnn-7.6.5)
	python==3.5
	tqdm==4.31.1
	nltk==3.3
	h5py==2.8.0
	ujson==1.35


### [download dataset & preprocessing]

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


-------------

### [training Phase]

- run reference script in "./model" folder
- results will be displayed in console <br>
- results will be saved to "./model/TEST_run_result.txt" <br>
```bash
sh reference_script_train.sh
```

### [hyper parameters]

- major parameters : edit from "./model/reference_script_train.sh" <br>
- other parameters : edit from "./model/params.py" <br>


-------------
### [cite]

- Please cite our paper, when you use our code | dataset | model
	> @inproceedings{yoon2020propagate,<br>
  title={Propagate-Selector: Detecting Supporting Sentences for Question Answering via Graph Neural Networks},<br>
  author={Yoon, Seunghyun and Dernoncourt, Franck and Kim, Doo Soon and Bui, Trung and Jung, Kyomin},<br>
  booktitle={Proceedings of The 12th Language Resources and Evaluation Conference},<br>
  pages={5400--5407},<br>
  year={2020}<br>
}
