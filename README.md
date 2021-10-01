# SPARTA
Re-Implementation of [SPARTA: Efficient Open-Domain Question Answering via Sparse Transformer Matching Retrieval](https://arxiv.org/abs/2009.13013). It is the re-implementation we used for [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663).

Also have a look at our BEIR repository: https://github.com/UKPLab/beir


Note: Sorry, this is just research code, it is not in the best shape. It is sadly also not well documented.

## Requirements

- Pytorch with at least version 1.6.0: https://pytorch.org/get-started/locally/
- `pip install sentence-transformers==1.2.1`

## Training

See `train_sparta_msmarco.py` how to train it on the [MSMARCO Passage Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking) dataset. Note, you find the needed training files there. Download them and put them in a `data/` folder.

## Evaluation

See `eval_msmarco.py` how to evaluate a SPARTA model on the MSMARCO Passage Ranking dataset. 


## Pretrained model

We provide a pre-trained model here: https://huggingface.co/BeIR/sparta-msmarco-distilbert-base-v1

## Evaluation
See [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663) how well our SPARTA implementation performs across several retrieval tasks.
