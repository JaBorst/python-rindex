# python-rindex
# RIModel

Creates a Random-Indexing-Model. There are function to add contexts for specific tokens and RIModel will build the Model from add

The most important function there are:

```r = RIModel( d=100, k=40) ``` 


__add_context__:
You can add a list of words and specify the index of the word you want to add. The rest will be treated as the context words:

```context(self, context = ["hello", "world", "whats", "up"], index = 0):```

__add_unit__:
You can add any unit you want by name and specify a list of words that is the the context of the unit. Optionally you can specify a list of Weights for the context. This method lets you named documents for example. You can add a document name as unit and specify all the words of the document as context:

```r.add_unit(self, unit="doc1", context=["hello", "world", "whats", "up"], weights=[0.5, 1.0, 1.0, 0.5]):```


__get_similarity__:
Additionally, after creating a Model you can query the similarity of two units by using

```r.get_similarity(self, word1 = "hello", word2 = "world", method = "cos"):```
Implemented methods are "cos", "ksd", and "jaccard" 
