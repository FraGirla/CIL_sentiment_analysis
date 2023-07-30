import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
from utils import *
import time
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler


os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(torch.cuda.get_device_name(device))
torch.cuda.empty_cache()
ts = int(time.time())

def training(X_train, y_train, X_test, y_test, model_name):
    """
    Train and evaluate a model.

    Args:
        X_train (numpy.ndarray): Training data features.
        y_train (numpy.ndarray): Training data labels.
        X_test (numpy.ndarray): Test data features.
        y_test (numpy.ndarray): Test data labels.
        model_name (str): Name of the model to train.

    Returns:
        tuple: A tuple containing accuracy score and predicted labels.
    """
    print("training model", model_name)
    if model_name=='LogisticRegression':
        model = LogisticRegression(n_jobs=-1, random_state=config.general.seed, solver='saga')
        model.fit(X_train, y_train)
    elif model_name == 'SVC':
        model = LinearSVC(random_state=config.general.seed)
        model.fit(X_train, y_train)
    elif model_name == 'RandomForestClassifier':
        model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=config.general.seed)
        model.fit(X_train, y_train)
    elif model_name == 'XGBClassifier':
        model = XGBClassifier(n_jobs=-1, random_state=config.general.seed)
        model.fit(X_train, y_train)
    elif model_name == 'GradientBoostingClassifier':
        model = GradientBoostingClassifier(random_state=config.general.seed)
        model.fit(X_train, y_train)
    elif model_name == 'Naive Bayes':
        model = GaussianNB()
        model.fit(X_train, y_train)
    else:
        raise ValueError("Model not supported")    
    print("model trained")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy: ", accuracy)
    return accuracy, y_pred


def cross_val(train_df, embedding, model):
    """
    Perform k-fold cross-validation on the dataset using a specified embedding technique and model.

    Args:
        train_df (pandas.DataFrame): The training dataset.
        embedding (str): The embedding technique to use.
        model (str): The model to train.

    Returns:
        None
    """
    accuracies = []

    folds = StratifiedKFold(n_splits=config.general.n_folds)
    for fold_, (train_index, test_index) in enumerate(folds.split(train_df, train_df['label'])):
        print ("Fold {}".format(fold_))
        train_fold = train_df.iloc[train_index]
        test_fold = train_df.iloc[test_index]
        
        if config.general.use_subsampling:
            train_fold = train_fold.iloc[:config.subsampling.train].sample(frac=1, random_state=config.general.seed)
            test_fold = test_fold.iloc[:config.subsampling.test]
        else:
            train_fold = train_fold.sample(frac=1, random_state=config.general.seed)
            test_fold = test_fold

        y_train = train_fold['label']
        y_test = test_fold['label']
        if embedding == 'bow':
            vectorizer = CountVectorizer(max_features=5000)
            X_train = vectorizer.fit_transform(train_fold['text'])
            X_test = vectorizer.transform(test_fold['text'])
        elif embedding == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=5000)
            X_train = vectorizer.fit_transform(train_fold['text'])
            X_test = vectorizer.transform(test_fold['text'])
        elif embedding == 'glove':
            pass

        #use standard scaler
        scaler = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


        accuracy, _ = training(X_train, y_train, X_test, y_test, model)
        accuracies.append(accuracy)

        del train_fold
        del test_fold
        
            
    print("mean acc: ", sum(accuracies) / len(accuracies))
    print("std acc: ", np.std(accuracies))

if __name__ == '__main__':
    #Load the configuration from 'config.yaml'
    config = get_config('config.yaml')
    config = dictionary_to_namespace(config)
    set_seed(config.general.seed)

    #create pandas dataframe for training dataset
    train_df = pd.read_csv(config.general.train_path)

    #drop nan
    train_df.dropna(inplace=True)

    #drop every column except text and label
    train_df = train_df[[config.general.column_text_name,config.general.column_label_name]]

    #rename columns
    train_df.columns = ['text','label']

    for embedding in ['bow','tfidf']:
        for model in ['LogisticRegression', 'SVC', 'XGBClassifier','GradientBoostingClassifier']:
            print(f"Embedding: {embedding}, Model: {model}")
            cross_val(train_df, embedding, model)