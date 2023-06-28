from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, models, losses, InputExample
import logging
from datetime import datetime
import os
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
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negs_per_system", default=1, type=int)
parser.add_argument("--model_name",default='distilbert-base-uncased')
parser.add_argument("--number_of_queries", default=50000, type=int)

args = parser.parse_args()
print(args)

# The  model we want to fine-tune
model_name = args.model_name
train_batch_size = args.train_batch_size           #Increasing the train batch size improves the model performance, but requires more GPU memory
max_seq_length = args.max_seq_length            #Max length for passages. Increasing it, requires more GPU memory
num_negs_per_system = args.num_negs_per_system         #  Number of  negatives to add from each system
num_epochs = args.epochs                 # Number of epochs we want to train


logging.info("Create new SBERT model")
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model_save_path = 'train/models/train_LaQuE-bi-encoder-mnrl-{}-{}'.format(model_name.replace("/", "-"), datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}         #dict in the format: passage_id -> passage. Stores all existent passages
collection_filepath = 'collection/dbpedia201510.tsv'

logging.info("Read corpus")
with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        pid = pid
        corpus[pid] = passage

### Read the train queries, store in queries dict
queries = {}        #dict in the format: query_id -> query. Stores all training queries
queries_filepath = 'queries/queries.train.tsv'

with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        try:
            qid, query = line.strip().split("\t")
        except:
            print(line)
        queries[qid] = query
      
train_queries={}

for file in os.listdir('train/triples'):
    with open('train/triples/'+file, 'rb') as f:
        train_queries_initial = pickle.load(f)
        
    for qid in train_queries_initial:
        if len(train_queries_initial[qid]['neg']) > args.num_negs_per_system:
            selected=random.sample(train_queries_initial[qid]['neg'],args.num_negs_per_system)
            train_queries_initial[qid]['neg']=selected
          
        #Get the positive passage ids
        pos_pids = train_queries_initial[qid]['pos']
        #Get the hard negatives
        neg_pids = set()
        negs_added = 0
        for pid in train_queries_initial[qid]['neg']:
            if pid not in neg_pids:
                neg_pids.add(pid)
                negs_added += 1
                if negs_added >= num_negs_per_system:
                    break

        if (len(pos_pids) > 0 and len(neg_pids) > 0):
            train_queries[qid] = {'qid': qid, 'query': queries[qid], 'pos': pos_pids, 'neg': neg_pids}
        
        if len(train_queries) > args.number_of_queries:
            break
    if len(train_queries) > args.number_of_queries:
        break
    

# We create a custom LaQuE dataset that returns triplets (query, positive, negative)
class LaQuEDataset(Dataset):
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

train_dataset = LaQuEDataset(train_queries, corpus=corpus)
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
