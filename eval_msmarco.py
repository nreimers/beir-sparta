import gzip
import torch
import json
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
import tqdm
import numpy as np
import sys
import pickle
import logging
from sentence_transformers import LoggingHandler
import os
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
import random




#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout



device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_passage_emb(tokenizer, bert_model, bert_input_emb, sparse_vec_size, passages):
    sparse_embeddings = []

    with torch.no_grad():
        tokens = tokenizer(passages, padding=True, truncation=True, return_tensors='pt', max_length=500).to(device)
        passage_embeddings = bert_model(**tokens).last_hidden_state
        for passage_emb in passage_embeddings:
            scores = torch.matmul(bert_input_emb, passage_emb.transpose(0, 1))
            max_scores = torch.max(scores, dim=-1).values
            relu_scores = torch.relu(max_scores) #Eq. 5
            final_scores = torch.log(relu_scores + 1)  # Eq. 6, final score

            top_results = torch.topk(final_scores, k=sparse_vec_size)
            tids = top_results[1].cpu().detach().tolist()
            scores = top_results[0].cpu().detach().tolist()
            passage_emb = []
            for tid, score in zip(tids, scores):
                if score > 0:
                    passage_emb.append((tid, score))
                else:
                    break

            sparse_embeddings.append(passage_emb)

    return sparse_embeddings


def main():
    model_name = sys.argv[1]
    corpus_max_size = int(sys.argv[2]) * 1000

    sparse_vec_size = 2000
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name)
    bert_model.to(device)
    bert_model.eval()

    bert_input_emb = bert_model.embeddings.word_embeddings(torch.tensor(list(range(0, len(tokenizer))), device=device))

    # Set Special tokens [CLS] [MASK] etc. to zero
    for special_id in tokenizer.all_special_ids:
        bert_input_emb[special_id] = 0 * bert_input_emb[special_id]

    dev_qids = set()
    needed_pids = set()
    needed_qids = set()

    questions = {}
    corpus = {}
    relevant = {}  ##qid => Set[cid]

    ########### load eval dataset
    dev_queries_file = 'data/queries.dev.small.tsv'

    with open(dev_queries_file) as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            dev_qids.add(qid)

    with open('data/qrels.dev.tsv') as fIn:
        for line in fIn:
            qid, _, pid, _ = line.strip().split('\t')

            if qid not in dev_qids:
                continue

            if qid not in relevant:
                relevant[qid] = set()
            relevant[qid].add(pid)

            needed_pids.add(pid)
            needed_qids.add(qid)

    with open(dev_queries_file) as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            if qid in needed_qids:
                questions[qid] = query.strip()

    with gzip.open('data/collection.tsv.gz', 'rt') as fIn:
        for line in fIn:
            pid, passage = line.strip().split("\t")

            if pid in needed_pids or corpus_max_size <= 0 or len(corpus) <= corpus_max_size:
                corpus[pid] = passage.strip()

    ########
    passage_pids = list(corpus.keys())
    passages = [corpus[pid] for pid in passage_pids]

    print("Questions:", len(questions))
    print("Passages:", len(passages))

    # Encode passages
    batch_size = 64

    num_elements = len(passages) * sparse_vec_size
    col = np.zeros(num_elements, dtype=np.int)
    row = np.zeros(num_elements, dtype=np.int)
    values = np.zeros(num_elements, dtype=np.float)


    sparse_idx = 0
    for start_idx in tqdm.trange(0, len(passages), batch_size):
        passage_embs = compute_passage_emb(tokenizer, bert_model, bert_input_emb, sparse_vec_size, passages[start_idx:start_idx + batch_size])

        for pid, emb in enumerate(passage_embs):
            for tid, score in emb:
                col[sparse_idx] = start_idx+pid
                row[sparse_idx] = tid
                values[sparse_idx] = score
                sparse_idx += 1



    logging.info("Create sparse matrix")
    sparse = csr_matrix((values, (row, col)), shape=(len(bert_input_emb), len(passages)), dtype=np.float)
    print("Scores:", sparse.shape)

    logging.info("Start scoring")
    # Compute MRR for questions
    mrr = []
    k = 10
    for qid, question in tqdm.tqdm(questions.items(), total=len(questions)):
        token_ids = tokenizer(question, add_special_tokens=False)['input_ids']

        # Get the candidate passages
        scores = np.asarray(sparse[token_ids, :].sum(axis=0)).squeeze(0)
        top_k_ind = np.argpartition(scores, -k)[-k:]
        hits = sorted([(pid, scores[pid]) for pid in top_k_ind], key=lambda x: x[1], reverse=True)

        mrr_score = 0
        for rank, hit in enumerate(hits):
            pid = passage_pids[hit[0]]
            if pid in relevant[qid]:
                mrr_score = 1 / (rank + 1)
                break
        mrr.append(mrr_score)

    print("MRR@10:", np.mean(mrr))

if __name__ == '__main__':
    main()



