# Sentiment Analysis on Tripadvisor reviews
Sentiment analysis, analyzing user’s textual reviews, to understand if a comment includes a positive or negative mood (i.e., binary classification model). The dataset has been scraped from the tripadvisor.it Italian web site and contains 41077 textual reviews written in the Italian language.

For all the details about data exploration, data preprocessing, model selection and results see the [project report](https://github.com/alessandrodesole/Sentiment-Analysis-on-Tripadvisor-reviews/blob/main/Report.pdf).

## Models implemented

In `solution.py` has been implemented different classifiers: Decision Tree, Random Forest, Stochastic Gradient Descent (SGD) and Linear Support Vector Machine (SVM). During the training of each model, an hyperparameter search has been performed. To handle the imbalance of the dataset, during training and validation phase a cross-validation approach has been applied. This approach consists of dividing all the samples in groups of subsamples and consequently the prediction function was learned using k-1 folds and the fold left out was used for test. The default value for the number of folds was set to 10.

## Results

Overall the best *accuracy* (**0.9476**) and the best *f1-weighted* score (**0.9681**) has been obtained by applying *Linear SVM* model.
