"""
This examples show how to train a Bi-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)

Negative passage are hard negative examples, that were mined using different dense embedding methods and lexical search methods.
Each positive and negative passage comes with a score from a Cross-Encoder. This allows denoising, i.e. removing false negative
passages that are actually relevant for the query.

With a distilbert-base-uncased model, it should achieve a performance of about 33.79 MRR@10 on the MSMARCO Passages Dev-Corpus

Running this script:
python train_bi-encoder-v3.py
"""
import sys
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
from collections import defaultdict
from torch.utils.data import IterableDataset
import tqdm
from torch.utils.data import Dataset
import random
import pickle
import argparse

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=16, type=int)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--max_passages", default=0, type=int)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument("--negs_to_use", default=None, help="From which systems should negatives be used? Multiple systems seperated by comma. None = all")
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negs_per_system", default=1, type=int)
parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
parser.add_argument("--use_all_queries", default=False, action="store_true")
args = parser.parse_args()

print(args)

# The  model we want to fine-tune
model_name = 'distilroberta-base'

train_batch_size = args.train_batch_size           #Increasing the train batch size improves the model performance, but requires more GPU memory
max_seq_length = args.max_seq_length            #Max length for passages. Increasing it, requires more GPU memory
num_negs_per_system = args.num_negs_per_system         # We used different systems to mine hard negatives. Number of hard negatives to add from each system
num_epochs = args.epochs                 # Number of epochs we want to train


logging.info("Create new SBERT model")
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model_save_path = 'output/train_bi-encoder-mnrl-{}-{}'.format(model_name.replace("/", "-"), datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


### Now we read the MS Marco dataset
data_folder = 'dataset'

#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}         #dict in the format: passage_id -> passage. Stores all existent passages
collection_filepath = os.path.join(data_folder, 'dbpedia201510.tsv')

logging.info("Read corpus: collection.tsv")
with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        pid = pid
        corpus[pid] = passage


### Read the train queries, store in queries dict
queries = {}        #dict in the format: query_id -> query. Stores all training queries
queries_filepath = os.path.join(data_folder, 'queries.train')

with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        try:
            qid, query = line.strip().split("\t")
        except:
            print(line)
        queries[qid] = query



train_queries0={}

for file in os.listdir('dataset/triples'):
    with open('dataset/triples/'+file, 'rb') as f:
        train_queries0 = pickle.load(f)
        
    for qid in train_queries0:
        if len(train_queries0[qid]['neg']) > args.num_negs_per_system:
            selected=random.sample(train_queries0[qid]['neg'],args.num_negs_per_system)
            train_queries0[qid]['neg']=selected

    print(file)    
print('training creation done')

    
logging.info("Read hard negatives train file")
train_queries={}
negs_to_use = None

for qid in train_queries0:

        #Get the positive passage ids
        pos_pids = train_queries0[qid]['pos']

        #Get the hard negatives
        neg_pids = set()
        if negs_to_use is None:

            negs_to_use = train_queries0[qid]['neg']


        negs_added = 0
        for pid in train_queries0[qid]['neg']:
            if pid not in neg_pids:
                neg_pids.add(pid)
                negs_added += 1
                if negs_added >= num_negs_per_system:
                    break

        if (len(pos_pids) > 0 and len(neg_pids) > 0):
            train_queries[qid] = {'qid': qid, 'query': queries[qid], 'pos': pos_pids, 'neg': neg_pids}


# We create a custom MSMARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.
class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']
        qid = query['qid']

        if len(query['pos']) > 0:
            pos_id = query['pos'].pop(0)    #Pop positive and add at end
            pos_text = self.corpus[pos_id]
            query['pos'].append(pos_id)
        else:   #We only have negatives, use two negs
            pos_id = query['neg'].pop(0)    #Pop negative and add at end
            pos_text = self.corpus[pos_id]
            query['neg'].append(pos_id)

        #Get a negative passage
        neg_id = query['neg'].pop(0)    #Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query['neg'].append(neg_id)


        return InputExample(texts=[query_text, pos_text, neg_text])


    def __len__(self):
        return len(self.queries)

# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(train_queries, corpus=corpus)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=args.warmup_steps,
          use_amp=True,
          checkpoint_path=model_save_path,
          checkpoint_save_steps=len(train_dataloader),
          optimizer_params = {'lr': args.lr},
          show_progress_bar=True
          )

# Save the model
model.save(model_save_path)
