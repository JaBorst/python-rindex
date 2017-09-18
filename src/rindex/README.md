#python-rindex
#RIModel

Creates a Random-Indexing-Model. There are function to add contexts for specific tokens and RIModel will build the Model from add

The most important function there are:
```r = RIModel( d=100, k=40) ``` 


__add_context__:
You can add a list of words and specify the index of the word you want to add. The rest will be treated as the context words:
context(self, context = [], index = 0):
