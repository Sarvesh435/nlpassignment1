# Assignment1
Group: Chimera

### Members:

1. Fatihah Ulya Hakiem (124734)
2. Sarvesh Kumar Singh (124773) 

### How to run:

1. Place ``train.jsonl`` and ``test.jsonl`` on the same folder as ``a1.py``.
2. Run the file using ``python a1.py``.

### Code Description:

--
In this task, we use ``CountVectorizer`` to tokenize the documents, ``TfIdfTransformer`` to get the tf-idf value for each token, and Multinomial Naive Bayes model ``MultinomialNB`` to classify and predict the document language.  

Using (1,1) n-gram gives an average of 98% accuracy and f1-score of 0.98. We also tried using (1,2) and (1,3) n-gram, but the results are not improving and the execution time also increase very significantly. Therefore, we decided on (1,1) n-gram for the final code. 

We also tried to use SVM model ``SVC`` but it gives an error when given Chinese-language documents and the training time is considerably larger. 

### References:
1. The reference documents and links given for this assignment
2. https://dbs.cs.uni-duesseldorf.de/lehre/bmarbeit/barbeiten/ba_panich.pdf
3. https://aclanthology.org/E14-4019.pdf