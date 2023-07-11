# LaQuE: Large-Scale Query Collection for Entity Search

First, please clone this repository and then follow the related sections for downloading the data, training, retrieving and etc.  

## Download the dataset
**Due to the large sizes of the files, we were not able to upload them in this Git repo. You may download all the necessary file from [here](https://www.dropbox.com/scl/fo/baib3l6cftkem1zhfj3oa/h?dl=0&rlkey=6vceu03w24iw7a6zvqtnki4gn).**

**We recommend following these commands to get them separately and step by step for specific purposes:**

First, you may download the main files including the cleaned collection, queries, different query subsets and the related entities (qrels):

```
wget https://www.dropbox.com/s/pgkzbcnch0kw606/Main_LaQuE.zip
unzip Main_LaQuE.zip
```

Once you unzip ```Main_LaQuE```, You will have the following files:

### Collection

- ```collection/dbpedia201510.tsv ```: This is the main collection including over 4.6 million entities with their DBpedia abstract in tab-separated format. 
For instance:

```
http://dbpedia.org/resource/Animation   "Animation is the process of creating the illusion of motion and shape change by means of the rapid display of a sequence of static images that minimally differ from each other. The illusion—as in motion pictures in general—is thought to rely on the phi phenomenon. Animators are artists who specialize in the creation of animation."@en
http://dbpedia.org/resource/Acid        "An acid (from the Latin acidus/acēre meaning sour) is a chemical substance whose aqueous solutions are characterized by a sour taste, the ability to turn blue litmus red, and the ability to react with bases and certain metals (like calcium) to form salts. Aqueous solutions of acids have a pH of less than 7. Non-aqueous acids are usually formed when an anion (negative ion) reacts with one or more positively charged hydrogen cations."@en
http://dbpedia.org/resource/Alkane      "In organic chemistry, an alkane, or paraffin (a historical name that also has other meanings), is a saturated hydrocarbon. Alkanes consist only of hydrogen and carbon atoms and all bonds are single bonds.   Alkanes (technically, always acyclic or open-chain compounds) have the general chemical formula CnH2n+2. For example, Methane is CH4, in which n=1 (n being the number of Carbon atoms)."@en
```

### Train/Dev/Test Queries and Related entities

- The queries and qrels for the three splits i.e., ```train/dev/test``` will be stored under ```queries``` and ```qrels``` directory.
```
queries/queries.dev.tsv        qrels/qrels.dev.tsv
queries/queries.test.tsv        qrels/qrels.test.tsv
queries/queries.train.tsv        qrels/qrels.train.tsv
```

Here are a few instances of queries in format of ```<query ID>\t<Query Text>```:

```
3982317        history of socialism in america
5244799        rbg cancer history
4978403        petroleum equipment
9615173        iron cross symbol
```

and related entities (qrel) files in trec format as ```<query ID> 0 <Related Entity ID> 1```: 

```
3982317 0 http://dbpedia.org/resource/History_of_the_socialist_movement_in_the_United_States 1
5244799 0 http://dbpedia.org/resource/Ruth_Bader_Ginsburg 1
4978403 0 http://dbpedia.org/resource/Petroleum 1
9615173 0 http://dbpedia.org/resource/Iron_Cross 1
```

### Other Query Subsets: Popularity and difficulty
In LaQue, we introduce a revised categorization of queries that takes into account both the popularity of the related entities and the difficulty of the queries themselves.
As such, you can find the two different query subsets:

We note that since these are all subsets of ```queries/queries.dev.tsv```, you can evaluate them using their related entities in ``` qrels/qrels.dev.tsv```. 

- **Popularity-based**: We split the over 100k queries in LaQuE dev set into 4 categories based on how popular their related entities are.  
```
query_subsets/popularity-based/query_splits/queries.high-pop.tsv
query_subsets/popularity-based/query_splits/queries.pop.tsv
query_subsets/popularity-based/query_splits/queries.somewhatpop.tsv
query_subsets/popularity-based/query_splits/queries.unpop.tsv
```

Additionally ```query_subsets\popularity-based\popularity.dev.tsv``` shows the page views for each entity in LaQuE dev set on Wikipedia from January 1, 2018, to December 31, 2022.


- **Diffuculty-based**: We also split the queries in LaQuE dev set based on their performance from BM25 into:

```
query_subsets\difficulty-based\query_splits\queries.easy.tsv
query_subsets\difficulty-based\query_splits\queries.med.tsv
query_subsets\difficulty-based\query_splits\queries.hard.tsv
query_subsets\difficulty-based\query_splits\queries.veryhard.tsv
```


## Retrieved results' runs

Although you can easily retrieve the queries in LaQuE dev and test set by any retriever of your choice, you may also download all the run files for the reported results in the paper including top-1000 retrieved results for all 8 dense retrievers as well as two sparse retrievers from [here](https://www.dropbox.com/scl/fo/mfnjbv7l9rjhtjjjmyghe/h?dl=0&rlkey=r08vblx0vaozxqtzdz5zj11qg). Due to the size of the run files, we split the run files into three files (e.a. ~8-10 GB) including the top-1000 retrieved entities for queries in LaQuE dev set by 
[sparse retrievers](https://www.dropbox.com/s/0ja0h67y7wuku8v/sparse.tar.gz?dl=0)
, [dense retrievers trained on MS MARCO](https://www.dropbox.com/s/va49py0iht2c12f/dense_msmarco.tar.gz?dl=0)
and [dense retrievers trained on LaQuE]().

```
cd runs/
wget https://www.dropbox.com/s/va49py0iht2c12f/dense_msmarco.tar.gz
wget https://www.dropbox.com/s/rnzv3x8l6wdrca7/dense_ours.tar.gz
wget https://www.dropbox.com/s/0ja0h67y7wuku8v/sparse.tar.gz
tar -xvf  sparse.tar.gz
tar -xvf  dense_ours.tar.gz
tar -xvf  dense_msmarco.tar.gz
```

This includes the following retrievers' results :
- BM25 (+rm3)
- QL (+rm3)
- Dense retriever with BERT-base-uncased - trained on LaQuE/MSMarco
- Dense retriever with DistilBERT - trained on LaQuE/MSMarco
- Dense retriever with MiniLM - trained on LaQuE/MSMarco
- Dense retriever with DistilRoBERTa - trained on LaQuE/MSMarco

## Training a Dense Retriever on LaQuE

In this GitHub repository, you will find an example of training a dense retriever using the [sentence-transformer](https://www.sbert.net/docs/pretrained-models/msmarco-v3.html) package. While it is possible to train any retriever using over 2 million queries and their associated entities from the LaQuE train set, we have specifically focused on this approach as an example of how LaQue could be leveraged as a valuable resource for tackling entity retrieval.

For training, we utilize the related entities from the train set as positive samples. Additionally, we randomly select a retrieved item from the top 1000 entities retrieved by BM25 as negative examples. Given a large number of queries in the train set, we have already divided them into 15 chunks and stored the resulting triples in pickle format.

Feel free to explore the repository and utilize the provided code and data for training your own dense retriever.

You may download all the queries with their paired positive and negative samples i.e., [triples for training, from here](https://www.dropbox.com/s/gbcxlq4tbbneam3/triples.tar.gz) using the following commands :

```
cd train/
wget https://www.dropbox.com/s/gbcxlq4tbbneam3/triples.tar.gz
tar -xvf  triples.tar.gz
```

Each pickle file in ```triples.tar.gz``` include a dictionary with the following information for more than 100K queries:

```
        triple_dic[qid]={}
        triple_dic[qid]['qid']=qid #<query id>
        triple_dic[qid]['query']=qtext #<query text>
        triple_dic[qid]['pos']=[] #list of related entities IDs
        triple_dic[qid]['neg']=[]# list of 100 unclicked retrieved entities with BM25 for this query 
```

After extracting the triples, we can train the model with the choice of your language model of interest as follows:
```
python train/train.py \
        --train_batch_size 16 \
        --max_seq_length 300   \
        --epoch 1 \
        --pooling mean \
        --warmup_steps 1000 \
        --lr 2e-5 \
        --num_negs_per_system 1  \
        --model_name distilbert-base-uncased  \
        --number_of_queries=50000  
```

You can also download the [trained models with default settings from here]().
```
cd train/
wget https://www.dropbox.com/s/jky6mkowgp1tzru/trained_models_on_laque.zip
unzip trained_models_on_laque.zip
```

## Summary:
|             Description            |           Filename          | File size |              Format             | Details                                                                                      |
|:----------------------------------:|:---------------------------:|----------:|:-------------------------------:|----------------------------------------------------------------------------------------------|
| Collection                         | [LaQuE_collection.tsv](https://www.dropbox.com/scl/fi/l0pb90j6hluwukrhy94mr/LaQuE_collection.tsv?rlkey=ituxv1k14orbaf1ybra3ea681&dl=0)        | 1.50 GB   | tsv: entity_id, entity abstract |                                                                                              |
| Queries                            | [queries.zip](https://www.dropbox.com/scl/fi/pkoso7sdgx3nz32ycjynl/queries.zip?rlkey=95l1yaeae4hz5fxk1pel8d1ue&dl=0)                 | 29.2 MB   | tsv: q_id, qury_text            | Include 3 files : queries.dev.tsv queries.test.tsv queries.train.tsv                         |
| Qrels                              | [qrels.zip](https://www.dropbox.com/scl/fi/vut22e343k5pn8eayed60/qrels.zip?rlkey=ewf3oltqmueh77t5yzbpblp7x&dl=0)                   | 25.1 MB   | TREC qrels format               | Include 3 files : qrels.dev.tsv qrels.test.tsv qrels.train.tsv                               |
| Train Triples                      | [triples.tar.gz](https://www.dropbox.com/s/gbcxlq4tbbneam3/triples.tar.gz?dl=0)              | 10.5 GB   | TREC run format                 | Include 16 pickle files - top1000 retrieved entities for each query in the train set by BM25 |
| Baselines -retrieved results-  Dev | 3 seperate compressed files | 24.03 GB  | TREC run format                 |                                                                                              |
| Trained models                     | [trained_models_on_laque.zip](https://www.dropbox.com/s/jky6mkowgp1tzru/trained_models_on_laque.zip?dl=0) |           | Pytorch Models                  |                                                                                              |
