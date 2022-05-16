import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

# 1. Read the file
def open_jsonl(filename):
    sentences = []
    with open(filename,  encoding="utf8") as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        sentences.append(result)
    
    return pd.DataFrame(sentences)

train = open_jsonl('train.jsonl')
test = open_jsonl('test.jsonl')

# 2. Separate the text with the label
X_train = train['text']
y_train = train['lang']

# 3. Tokenize the corpus
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)

# 4. Get the tf-idf value for each token
tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train_cv)

# 5. Initialize the model and train it using the tf-idf values
clf = MultinomialNB().fit(X_train_tfidf, y_train)

# 6. Validate it using 5-fold cross-validation
scores = cross_val_score(clf, X_train_tfidf, y_train, cv=5, scoring='f1_macro')
print('Test scores: ', scores)
print('Average: ', np.average(scores))

# 7. Predict the test dataset
X_test = test['text']
X_test_cv = cv.transform(X_test)
X_test_tfidf = tfidf.transform(X_test_cv)

predicted = clf.predict(X_test_tfidf)

# 8. Concat the prediction result to ID of test items.
df_new = pd.concat([test.id, pd.DataFrame(predicted, columns=['lang'])], axis=1)
df_new.to_json('predictions.jsonl', orient='records', lines=True)