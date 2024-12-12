# Evolving Knowledge Distillation for Lightweight Neural Machine Translation

<a href="https://arxiv.org/abs/2402.03898"><img src="https://img.shields.io/badge/Paper-arXiv:2402.03898-Green"></a>


Official PyTorch implementation of **Evolving Knowledge Distillation**, as presented in our paper: \
\
**Evolving Knowledge Distillation for Lightweight Neural Machine Translation** \
*[Jongwoo Ko](https://sites.google.com/view/jongwooko), [Sungnyun Kim](https://sungnyunkim.notion.site/Sungnyun-Kim-4770a0182c47469ebdcd357cde97bd32), Tianyi Chen, Se-Young Yun* \
KAIST AI and Microsoft


# Runtime Environment

* [PyTorch](http://pytorch.org/) version >= 1.10.0
* Python version = 3.8
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* To install and develop locally:

``` bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
pip install sacremoses
pip install sacrebleu==1.5.1

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```

## Data acquisition and processing
Here is an example of data acquisition and processing for the IWSLT 2014 (German-English) dataset. For the acquisition and processing of other datasets, you can refer to this example. 
``` bash
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en
```


## Evolving Knowledge Distillation
We provide example commands for transformer_iwslt_de_en model. 


### Train base model on iwslt14-de-en dataset
train student model:
```bash
# default parameter
bash bin/student_model.sh
# manual parameter passing
bash bin/student_model.sh ${SAVE_DIR} ${LOG_FILE} ${DATA_SET_DIR} ${MODEL} ${MAX_EPOCH}
```

train junior teacher model:
```bash
# default parameter
bash bin/junior_teacher.sh
# manual parameter passing
bash bin/junior_teacher.sh ${SAVE_DIR} ${LOG_FILE} ${DATA_SET_DIR} ${MODEL} ${MAX_EPOCH}
```

train senior teacher model:
```bash
# default parameter
bash bin/senior_teacher.sh
# manual parameter passing
bash bin/senior_teacher.sh ${SAVE_DIR} ${LOG_FILE} ${DATA_SET_DIR} ${MODEL} ${MAX_EPOCH}
```

### Distillation

train  junior student model:
```bash
# default parameter
bash bin/junior_student.sh
# manual parameter passing
bash bin/junior_student.sh ${SAVE_DIR} ${LOG_FILE} ${DATA_SET_DIR} ${TEACHER_MODEL_PATH} ${MODEL} ${MAX_EPOCH}
```

train senior student model:
```bash
# default parameter
bash bin/senior_student.sh
# manual parameter passing
bash bin/senior_student.sh ${SAVE_DIR} ${LOG_FILE} ${DATA_SET_DIR} ${TEACHER_MODEL_PATH} ${MODEL} ${MAX_EPOCH}
```

train master student model:
```bash
# default parameter
bash bin/master_student.sh
# manual parameter passing
bash bin/master_student.sh ${SAVE_DIR} ${LOG_FILE} ${DATA_SET_DIR} ${JUNIOR_STUDENT_MODEL_PATH} ${TEACHER_MODEL_PATH} ${MODEL} ${MAX_EPOCH}
```

### Run Evaluation
```bash
# default parameter
bash bin/eval_model.sh
# manual parameter passing
bash bin/eval_model.sh ${BEST_MODEL_PATH} ${LOG_FILE} ${DATA_SET_DIR}
```

## TAKD
### Train assistant teacher model on iwslt14-de-en dataset
train teacher model:
```bash
# default parameter
bash bin/TAKD_assistant_teacher.sh
# manual parameter passing
bash bin/TAKD_assistant_teacher.sh ${SAVE_DIR} ${LOG_FILE} ${DATA_SET_DIR} ${TEACHER_MODEL_PATH} ${MODEL} ${MAX_EPOCH}
```

### Distillation
train student model:
```bash
# default parameter
bash bin/TAKD_student.sh
# manual parameter passing
bash bin/TAKD_student.sh ${SAVE_DIR} ${LOG_FILE} ${DATA_SET_DIR} ${TEACHER_MODEL_PATH} ${MODEL} ${MAX_EPOCH}
```


## Results
DistiLLM outperforms other KD baselines in terms of both generation performance and training speed for various model families such as GPT-2, OPT, and OpenLLaMA.
<p align="center">
<img width="1394" src="https://github.com/jongwooko/distillm/assets/59277369/19ddac5c-4cd6-4d81-99d8-32723a8e60d8">
</p>


## Acknowledgement
Our code is based on the code of ICLR2024 [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/pdf/2306.08543.pdf).


## BibTeX
If you find this repo useful for your research, please consider citing our paper:

```
@inproceedings{kodistillm,
  title={DistiLLM: Towards Streamlined Distillation for Large Language Models},
  author={Ko, Jongwoo and Kim, Sungnyun and Chen, Tianyi and Yun, Se-Young}
}
```

## Contact
- Jongwoo Ko: jongwoo.ko@kaist.ac.kr
