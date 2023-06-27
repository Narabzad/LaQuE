# LaQuE: Large-Scale Query Collection for Entity Search

First, please clone this repository and then follow the related sections for downloading the data, training, retrieving and etc.  

## Download the dataset
Due to the large sizes of the files, we were not able to upload them in this Git repo. You may download all the necessary files including the cleaned collection, queries and the qrels from [here](https://www.dropbox.com/s/sc1fvthma0t0jbp/Main_LaQuE.zip?dl=0) by following these commands:

```
wget https://www.dropbox.com/s/sc1fvthma0t0jbp/Main_LaQuE.zip
unzip Main_LaQuE.zip
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

