### Final Project ###


### 1 - Import libraries -----------------------------

import csv
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter
import string
from nltk.corpus import stopwords as sw
from nltk.stem.snowball import ItalianStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import IncrementalPCA
from sklearn import metrics
from sklearn.model_selection import cross_val_score

### 2 - Define some global variables ------------------

punctuation = string.punctuation + '\n'
stopwords = sw.words('italian')
# Remove some useful words from stopwords
stopwords.remove('non')
stopwords.remove('sono')
stopwords.remove("ma")
stopwords.remove('contro')

### 3 - Define functions -----------------------------

def preprocessing(texts):
    prep_texts = []
    regex = re.compile('[%s]' % re.escape(punctuation))
    for text in texts:
        # Remove numbers
        text = re.sub(r'\d+', ' ', text)
        # Remove foreign chars
        text = re.sub(r'[^a-zA-ZàèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ]', ' ', text)
        # remove all the special characters
        text = re.sub(r'\W', ' ', text)
        # Remove punctuation
        text = regex.sub(" ", text)
        # Remove characters length equal to 1
        text = re.sub(r"\b[a-zA-Zèé]\b", "", text)
        # Remove whitespaces
        text = text.strip()
        # Convert text to lowercase
        text = text.lower()
        prep_texts.append(text)
    return prep_texts


def count_words(texts):
    words = set()
    for row in texts:
        words.update(row.split())
    return len(words)


def top_n_words_occur(texts, n):
    final_text = " ".join(texts)
    word_counter = Counter(final_text.split())
    return word_counter.most_common(n)  # return a list of tuples


def divide_dev_pos_neg(texts):
    pos_reviews = []
    neg_reviews = []
    for i in range(len(dev_texts)):
        if dev_classes[i] == 'pos':
            pos_reviews.append(texts[i])
        else:
            neg_reviews.append(texts[i])
    return pos_reviews, neg_reviews

def bar_chart(list):
    height = [h[1] for h in list[:10]]
    print(height)
    x = np.arange(10)
    labels = [l[0] for l in list[:10]]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(x, height, tick_label=labels)
    plt.show()


def tokenization(reviews):
    tokens = []
    stemmer = ItalianStemmer()
    for review in reviews:
        t = word_tokenize(review, language='italian')
        result = [stemmer.stem(word) for word in t if ((word not in stopwords)
                  and (len(word) > 1) and len(word) < 18)]
        str_result = " ".join(result)
        tokens.append(str_result)
    return tokens


# Map classes to int (0, 1)

def map_classes(labels):
    y_truth = []
    for dc in labels:
        if dc == "pos":
            y_truth.append(0)
        elif dc == "neg":
            y_truth.append(1)
    return y_truth


### 4 - Load data and Data exploration --------------------------

# I choose to import data from csv file and save them in 2 lists

dev_texts, dev_classes = [], []
eval_texts = []

with open('development.csv', mode='r', encoding='utf-8') as dev_fp:
    dev_reader = csv.reader(dev_fp)
    columns = next(dev_reader)
    for row in dev_reader:
        if row[0] != "" and (row[1] == 'pos' or row[1] == 'neg'):
            dev_texts.append(row[0])
            dev_classes.append(row[1])

# Check for the length

print("List of reviews len =", len(dev_texts))
print("List of lebes len =", len(dev_classes))

if len(dev_texts) == len(dev_classes):
    print("Correct: two lists have the same length")
else:
    sys.exit("Error: two lists have not the same length")

y = map_classes(dev_classes)

texts = pd.Series(dev_texts)
classes = pd.Series(dev_classes)

df = pd.DataFrame({columns[0]: texts, columns[1]: classes})

with open('evaluation.csv', mode='r', encoding='utf-8') as eval_fp:
    eval_reader = csv.reader(eval_fp)
    next(eval_reader)
    for row in eval_reader:
        eval_texts.append(row[0])

print("Number of textual reviews in development.csv =", df.shape[0])
print("Number of textual reviews in evaluation.csv =", len(eval_texts))

mask_p = df['class'] == 'pos'
mask_n = df['class'] == 'neg'

t_pos = df[mask_p].count()[0]
t_neg = df[mask_n].count()[0]

print(f"There are {t_pos} positive reviews in development.csv -> {t_pos/df.shape[0]*100:.2f}% of dataset")
print(f"There are {t_neg} positive reviews in development.csv -> {t_neg/df.shape[0]*100:.2f}% of dataset")

# Plot a bar chart

height = [df.shape[0], t_pos, t_neg]
x = [1, 2, 3]
bar_labels = ['Reviews in development.csv', 'Positive reviews', 'Negative Reviews']

bar_fig, bar_ax = plt.subplots()
bar_ax.bar(x, height, tick_label=bar_labels)
plt.show()

# Plot a pie

pie = plt.pie([t_pos, t_neg], labels=["Positives", "Negatives"], colors=['forestgreen', 'red'], autopct='%1.2f%%', explode=(0, 0.1), shadow=True, startangle=0)
plt.show()

# Statistics from dataset
dev_all_words = count_words(df['text'])
dev_pos_words = count_words(df.loc[mask_p, 'text'])
dev_neg_words = count_words(df.loc[mask_n, 'text'])

tot_words = 0
for str in df['text']:
    tot_words+=len(str)

print("Number of total words in development.csv =", tot_words)
print("Number of different words in development.csv =", dev_all_words)
print("Number of different words in positive reviews =", dev_pos_words)
print("Number of different words in negative reviews =", dev_neg_words)


### 5 - Data Preprocessing ---------------------------------------------

preproc_dev_texts = preprocessing(df['text'])
preproc_eval_texts = preprocessing(eval_texts)

### 6 - First data Analysis --------------------------------------------

print("\n\n1 - Analysis of data after first step of preprocessing\n\n")

all_preproc_dev_words = count_words(preproc_dev_texts)

print("There are", all_preproc_dev_words, "different words in development.csv\n")

all_most_preproc_occur = top_n_words_occur(preproc_dev_texts, 10)

print("Top 10 words in all texts:")
print(all_most_preproc_occur, "\n")

# Plot a bar char of all best words

bar_chart(all_most_preproc_occur)

# Divide in positive and negative reviews

preproc_dev_pos_reviews, preproc_dev_neg_reviews = divide_dev_pos_neg(preproc_dev_texts)

all_most_pos_occur = top_n_words_occur(preproc_dev_pos_reviews, 10)
all_most_neg_occur = top_n_words_occur(preproc_dev_neg_reviews, 10)

# Plot a bar chart of positive words
bar_chart(all_most_pos_occur)

# Plot a bar char of negative words
bar_chart(all_most_neg_occur)

print("Top 15 words in positive reviews:")
print(all_most_pos_occur, '\n')
print("Top 15 words in negative reviews:")
print(all_most_neg_occur, '\n')

### 7 - Second data analysis ----------------------------------

# Tokenization

dev_token_list = tokenization(preproc_dev_texts)
eval_token_list = tokenization(preproc_eval_texts)

print("\n\n2 - Analysis of data after first step of preprocessing\n\n")

# Count different words

diff_words = count_words(dev_token_list)

print("There are", diff_words, "different tokens in development.csv")

all_most_final_occur = top_n_words_occur(dev_token_list, 10)

print("Top 10 words in all texts:")
print(all_most_final_occur, "\n")

bar_chart(all_most_final_occur)

final_dev_pos_reviews, final_dev_neg_reviews = divide_dev_pos_neg(dev_token_list)

final_pos_occur = top_n_words_occur(final_dev_pos_reviews, 10)
final_neg_occur = top_n_words_occur(final_dev_neg_reviews, 10)

print("Top 10 words in positive reviews:")
print(final_pos_occur, '\n')
print("Top 10 words in negative reviews:")
print(final_neg_occur, '\n')

### Create a vector of TfidfVectorizer with many configurations -------------

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=1000)

X = vectorizer.fit_transform(dev_token_list)
X_eval = vectorizer.transform(eval_token_list)

pca = IncrementalPCA(n_components=100)
X_projection = pca.fit_transform(X.toarray())
plt.plot(pca.explained_variance_ratio_, marker='o', linestyle='')
plt.show()

# from the plot we find that n_components is 10 more or less

n = 10
pca = IncrementalPCA(n_components=n)
X_projection = pca.fit_transform(X.toarray())

### Define a function which calculates f1_score_weighted ----------

def train_model(classifier, X, y, nfolds):
    # Cross validation score

    f1_cvs = cross_val_score(classifier, X, y, cv=nfolds, scoring='f1_weighted', n_jobs=-1)
    mean_f1_cvs = f1_cvs.mean()

    return mean_f1_cvs

### GridSearch for best model with best params ------------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

clfs = []
all_params = []

# Add LogisticRegression

clfs.append(DecisionTreeClassifier())
param_grid_DT = {
    'criterion' : ['gini', 'entropy'],
    'spltter' : ['best', 'random'],
    'max_features' : ['auto', 'sqrt', 'log2'],
    'random_state' : [42],
    'class_weight' : [{0: 68, 1: 32}]
}
all_params.append(param_grid_DT)

# Add SGDClassifier
clfs.append(SGDClassifier())
param_grid_SGD = {
    #'alpha' : ['optimal']
    'tol': [1e-10, 1e-7, 1e-5, 1e-3],
    'n_jobs': [-1]
}
all_params.append(param_grid_SGD)

# Add Random Forests

clfs.append(RandomForestClassifier())
param_grid_RF = {
    'n_estimators':[10, 50, 100],
    'n_jobs': [-1]
}
all_params.append(param_grid_RF)

# Add LinearSVC

clfs.append(LinearSVC())
param_grid_SVC = {
    'tol': [1e-11, 1e-10, 1e-8, 1e-6, 1e-4],
    'C': [0.5, 1, 3, 5],
}
all_params.append(param_grid_SVC)

### Algorithm choice --------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict
import seaborn as sns

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, shuffle=True)

for clf in clfs:
    clf.fit(X_train, y_train)
    y_pred = cross_val_predict(clf, X_test, y_test, cv=10)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy of %s is %s"%(clf, acc))
    
    # Print the confusion matrix
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix of %s is %s"%(clf, cm))
    conf_mat_df = pd.DataFrame(cm)
    conf_mat_df.index.name = 'Actual'
    conf_mat_df.columns.name = 'Predicted'
    sns.heatmap(conf_mat_df, annot=True, cmap='GnBu', annot_kws={"size":16}, fmt='g')
    plt.show()

from sklearn.model_selection import GridSearchCV

best_score = 0.0
best_configs = []

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=100000)

X = vectorizer.fit_transform(dev_token_list)
X_eval = vectorizer.transform(eval_token_list)

for i in range(0, len(clfs)):
    gridsearch = GridSearchCV(clfs[i], all_params[i], scoring='f1_weighted', cv=10, n_jobs=-1)
    gridsearch.fit(X_projection, y)
    #print(gridsearch.best_estimator_)
    best_configs.append(gridsearch.best_estimator_)

best_index = 9
best_score = 0.0

for i in range(0, len(best_configs)):

    print("----", "----")
    print("Best config =", best_configs[i])
    clf = best_configs[i]
    score = train_model(clf, X, y, 10)
    print(score)
    if score > best_score:
        best_index = i
        best_score = score

### Run best classifier --------------------------------

best_clf = best_configs[best_index]
best_clf.fit(X, y)
y_eval_pred = best_clf.predict(X_eval)

### Print results in submission_final.csv file ---------

def trad_classes(y):
    classes = []
    for dc in y:
        if dc == 0:
            classes.append("pos")
        elif dc == 1:
            classes.append("neg")
    return classes

y_file = trad_classes(y_eval_pred)

csvData = [['Id', 'Predicted']]
for i in range(0, len(y_file)):
    csvData.append([i,y_file[i]])

with open('submission_final.csv', mode='w', encoding='utf-8', newline='') as csvFile:
    wr = csv.writer(csvFile,delimiter=',')
    wr.writerows(csvData)

csvFile.close()
print(f'printed file in submission_final.csv')
