# LaQuE: Large-Scale Query Collection for Entity Search

First, please clone this repository and then follow the related sections for downloading the data, training, retrieving and etc.  

## Download the dataset
Due to the large sizes of the files, we were not able to upload them in this Git repo. You may download all the necessary files including the cleaned collection, queries, different query subsets and the related entities (qrels) from [here](https://www.dropbox.com/s/sc1fvthma0t0jbp/Main_LaQuE.zip?dl=0) by following these commands:

```
wget https://www.dropbox.com/s/sc1fvthma0t0jbp/Main_LaQuE.zip
unzip Main_LaQuE.zip
```

Once you unzip ```Main_LaQuE```, You will have the following files:

- ```collection/dbpedia201510.tsv ```: This is the main collection including over 4.6 million entities with their dbpedia abstract in tab seperated format. 
For instance:

```
http://dbpedia.org/resource/Animation   "Animation is the process of creating the illusion of motion and shape change by means of the rapid display of a sequence of static images that minimally differ from each other. The illusion—as in motion pictures in general—is thought to rely on the phi phenomenon. Animators are artists who specialize in the creation of animation."@en
http://dbpedia.org/resource/Acid        "An acid (from the Latin acidus/acēre meaning sour) is a chemical substance whose aqueous solutions are characterized by a sour taste, the ability to turn blue litmus red, and the ability to react with bases and certain metals (like calcium) to form salts. Aqueous solutions of acids have a pH of less than 7. Non-aqueous acids are usually formed when an anion (negative ion) reacts with one or more positively charged hydrogen cations."@en
http://dbpedia.org/resource/Alkane      "In organic chemistry, an alkane, or paraffin (a historical name that also has other meanings), is a saturated hydrocarbon. Alkanes consist only of hydrogen and carbon atoms and all bonds are single bonds.   Alkanes (technically, always acyclic or open-chain compounds) have the general chemical formula CnH2n+2. For example, Methane is CH4, in which n=1 (n being the number of Carbon atoms)."@en
```

- The queries and qrels for the three splits i.e., ```train/dev/test``` will be stored under ```queries``` and ```qrels``` directory.
For instance:
```
queries/queries.dev.tsv        qrels/qrels.dev.tsv
queries/queries.test.tsv        qrels/qrels.test.tsv
queries/queries.train.tsv        qrels/qrels.train.tsv
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
