# NLP Final Project: Text Classification

## Overview
- Dataset: BBC News (5 categories: business, entertainment, politics, sport, tech)
- BERT model: BERT base cased
- mcc: >95%

## How to run:
- mkdir cache
- download file weight trained put into cache dir: https://drive.google.com/file/d/1chCUtSJWTHGs_OwO3OHTtx3Nf5Ynpz1W/view?usp=sharing

- demo (https://youtu.be/nmOwSWgGeXU):
```
$ export FLASK=app.py
$ flask run
```

- train/evaluate:
```
$ python train.py
$ python evaluate.py
```

## Reference:
https://github.com/huggingface/transformers
https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04
