# DistiLLM: Towards Streamlined Distillation for Large Language Models (ICML 2024)

<a href="https://arxiv.org/abs/2402.03898"><img src="https://img.shields.io/badge/Paper-arXiv:2402.03898-Green"></a>


Official PyTorch implementation of **DistiLLM**, as presented in our paper: \
\
**DistiLLM: Towards Streamlined Distillation for Large Language Models** \
*[Jongwoo Ko](https://sites.google.com/view/jongwooko), [Sungnyun Kim](https://sungnyunkim.notion.site/Sungnyun-Kim-4770a0182c47469ebdcd357cde97bd32), Tianyi Chen, Se-Young Yun* \
KAIST AI and Microsoft


# Requirements and Installation

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

# Data
Refer to the [Fairseq Official Documentation](https://fairseq.readthedocs.io/en/latest/getting_started.html#data-pre-processing) for detailed steps on data preprocessing.
Follow the instructions provided in the documentation to prepare your data correctly, ensuring the model can load and train properly.


Alternatively, you can also change the `CKPT` variable in each script to the corresponding model name to enable Transformers to download the base models automatically. For example, set `CKPT="gpt2-large"` in `scripts/gpt2/sft/sft_large.sh` causes download of the gpt2-large base model from the HugginFace model hub.

# Train
We provide example commands for transformer_iwslt_de_en model. 

# Baselines
The final checkpoints are selected by the **BLEU** scores.

## Train base model on iwslt14-de-en dataset
train student model:
```bash
./bin/student_model.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

train junior teacher model:
```bash
./bin/junior_teacher.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

train senior teacher model:
```bash
./bin/senior_teacher.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

## Distillation

train  junior student model:
```bash
./bin/junior_student.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

train senior student model:
```bash
./bin/senior_student.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

train master student model:
```bash
./bin/master_student.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```


#### MiniLLM Baselines
```bash
bash scripts/gpt2/minillm/train_base_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/minillm/train_medium_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/minillm/train_large_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

#### GKD Baselines
```bash
bash scripts/gpt2/gkd/gkd_base_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/gkd/gkd_medium_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/gkd/gkd_large_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

### DistiLLM
The final checkpoints are selected by the **validation loss**.
```bash
bash scripts/gpt2/init/init_base.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/init/init_medium.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/init/init_large.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

The final checkpoints are selected by the **ROUGE-L** scores.
```bash
bash scripts/gpt2/distillm/train_base_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/distillm/train_medium_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/distillm/train_large_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

## Run Evaluation
```bash
bash scripts/gpt2/eval/run_eval.sh ${GPU_IDX} ${/PATH/TO/DistiLLM}
bash scripts/opt/eval/run_eval.sh ${GPU_IDX} ${/PATH/TO/DistiLLM} 
bash scripts/openllama2/eval/run_eval.sh ${GPU_IDX} ${/PATH/TO/DistiLLM} 
```

## Results
DistiLLM outperforms other KD baselines in terms of both generation performance and training speed for various model families such as GPT-2, OPT, and OpenLLaMA.
<p align="center">
<img width="1394" src="https://github.com/jongwooko/distillm/assets/59277369/19ddac5c-4cd6-4d81-99d8-32723a8e60d8">
</p>

## Checkpoints (OpenLLaMA-3B)
We share the LoRA weights for OpenLLaMA-3B in [google drive](https://drive.google.com/drive/folders/1Yun1aNpn-mz2h-IVH_VdJ1Jhzm0K55Bo?usp=sharing).

## Acknowledgement
Our code is based on the code of ICLR2024 [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/pdf/2306.08543.pdf).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=jongwooko/distillm&type=Date)](https://star-history.com/#jongwooko/distillm&Date)

## BibTeX
If you find this repo useful for your research, please consider citing our paper:

```
@inproceedings{kodistillm,
  title={DistiLLM: Towards Streamlined Distillation for Large Language Models},
  author={Ko, Jongwoo and Kim, Sungnyun and Chen, Tianyi and Yun, Se-Young},
  booktitle={Forty-first International Conference on Machine Learning}
}
```

## Contact
- Jongwoo Ko: jongwoo.ko@kaist.ac.kr
