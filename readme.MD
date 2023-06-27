# LaQuE: Large-Scale Query Collection for Entity Search

First, please clone this repository and then follow the related sections for downloading the data, training, retrieving and etc.  

## Download the dataset
Due to the large sizes of the files, we were not able to upload them in this Git repo. You may download all the necessary files including the cleaned collection, queries, different query subsets and the related entities (qrels) from [here](https://www.dropbox.com/s/tnbh2jiwlxbg9pp/Main_LaQuE.zip?dl=0) by following these commands:

```
wget https://www.dropbox.com/s/tnbh2jiwlxbg9pp/Main_LaQuE.zip
unzip Main_LaQuE.zip
```

Once you unzip ```Main_LaQuE```, You will have the following files:

### Collection

- ```collection/dbpedia201510.tsv ```: This is the main collection including over 4.6 million entities with their dbpedia abstract in tab seperated format. 
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

### Other Query Subsets: Popularity and Performance
In LaQue, we introduce a revised categorization of queries that takes into account both the popularity of the related entities and the performance of the queries themselves.
As such, you can find the two different query subsets:

- *Popularity-based* : **We** split the over 100k queries in LaQuE dev set into 4 categories based on how popular their related entities are.  
```
query_subsets/popularity-based/query_splits/queries.high-pop.tsv
query_subsets/popularity-based/query_splits/queries.pop.tsv
query_subsets/popularity-based/query_splits/queries.somewhatpop.tsv
query_subsets/popularity-based/query_splits/queries.unpop.tsv
```
Since these are all subsets of ```queries/queries.dev.tsv```, you can evaluate them using their related enities in ``` qrels/qrels.dev.tsv```. 

- Performance-based: WE also split the queries in LaQuE dev set based on their performance from BM25 into:

```
query_subsets\performance-based\query_splits\queries.easy.tsv
query_subsets\performance-based\query_splits\queries.med.tsv
query_subsets\performance-based\query_splits\queries.hard.tsv
query_subsets\performance-based\query_splits\queries.veryhard.tsv
```



## Train a dense retriever on LaQuE
In this GitHub repository, you will find an example of training a dense retriever using the sentence-transformer library. While it is possible to train any retriever using over 2 million queries and their associated entities from the LaQuE train set, we have specifically focused on this approach as an example of how LaQue could be leveraged as a valuable resource for tackling entity retrieval.

For training, we utilize the related entities from the train set as positive samples. Additionally, we randomly select a retrieved item from the top 1000 entities retrieved by BM25 as negative examples. Given a large number of queries in the train set, we have already divided them into 15 chunks and stored the resulting triples in pickle format.

Feel free to explore the repository and utilize the provided code and data for training your own dense retriever using the sentence-transformer library.

You may download all the tripels from [here]() using the following commands :
```
wget 
cd train/
unzip triples.zip
```

## Runs on Dev set 
Although you can easily replicate all the results on LaQuE dev and test set, you may also download all the runs including top-1000 retrieved results for all the 8 dense retrievers as well as two sparse retrievers from [here]()
```
wget 
cd runs/
tar  dev-runs.zip
```
This includes the following retrievers results :
- BM25 (+rm3)
- QL (+rm3_
- Dense retriever with BERT-base-uncased - trained on LaQuE/MSMarco
- Dense retriever with DistilBERT - trained on LaQuE/MSMarco
- Dense retriever with MiniLM - trained on LaQuE/MSMarco
- Dense retriever with DistilRoBERTa - trained on LaQuE/MSMarco
