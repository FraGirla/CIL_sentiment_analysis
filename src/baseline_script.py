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
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(torch.cuda.get_device_name(device))
torch.cuda.empty_cache()
ts = int(time.time())

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, embeddings=None, label=None):
        embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1] // self.input_size, self.input_size)
        lstm_out, _ = self.lstm(embeddings)
        logits = self.fc(lstm_out[:, -1, :])  ## Taking the last output of LSTM as the prediction
        logits = torch.sigmoid(logits)
        loss = None
        if label is not None:
            loss = nn.functional.binary_cross_entropy(logits, label)
            return logits, loss
        else:
            return logits

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.word2vec = GloVe(name='twitter.27B', dim=100)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids = self.word2vec.get_vecs_by_tokens(self.data[index].split(), lower_case_backup=True).view(-1)
        return input_ids, self.label[index]
    
def collate_fn(batch):
    # Sort sentences by length in descending order
    inputs, targets = zip(*batch)
    # Pad sequences in the batch
    padded_batch = pad_sequence(inputs, batch_first=True, padding_value=0)
    return padded_batch, targets

def train_func_lstm(model: nn.Module, input_size:int, hidden_size: int, train_embeddings: pd.DataFrame, batch_size=32, epochs=3, lr=0.001, save_path=None, load_path=None, done_epochs=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    print(torch.cuda.get_device_name(device))

    X = train_embeddings['embeddings']
    y = train_embeddings['label']

    num_folds = 5

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    accuracy_scores = []
    
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        model = LSTMClassifier(input_size, hidden_size)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        if load_path is not None:
            #load epoch, model, optimizer, loss
            checkpoint = torch.load(f"{load_path}/lstm_{hidden_size}_{fold}_{done_epochs}_{batch_size}_{lr}.pt")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            done_epochs = checkpoint['epoch']
            fold = checkpoint['fold']
            loss = checkpoint['loss']
        else:
            done_epochs = 0
        # Initialize the model, criterion, and optimizer
        model.to(device)
        print(f"Fold {fold + 1}")
        # Split data into training and validation sets for this fold
        train_data = X.iloc[train_index].to_list()
        train_label = y.iloc[train_index].to_list()
        val_data = X.iloc[val_index].to_list()
        val_label = y.iloc[val_index].to_list()

    
        # Create training and validation datasets
        train_dataset = CustomDataset(train_data, torch.tensor(train_label).float())
        val_dataset = CustomDataset(val_data, torch.tensor(val_label).float())
    
        # Create DataLoader for training and validation
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn , shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn =collate_fn, shuffle=True)

        epoch_accuracy_scores = []
        assert done_epochs < epochs
        for epoch in range(done_epochs, epochs):
            # Training loop
            model.train()
            for batch in tqdm.tqdm(train_loader, ncols=100, desc=f"training epoch {epoch + 1}", total=len(train_loader)):
                optimizer.zero_grad()
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = torch.tensor(targets).float().unsqueeze(1)
                targets = targets.to(device)
                outputs, loss = model(inputs, targets)
                del inputs

                loss.backward()
                optimizer.step()
    
            # Validation loop
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in tqdm.tqdm(val_loader, ncols=100, desc=f"validation epoch {epoch + 1}", total=len(val_loader)):
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    targets = torch.tensor(targets).float().unsqueeze(1)
                    targets = targets.to(device)
                    outputs, loss = model(inputs, targets)
                    del inputs

                    val_loss += loss.item()
                    predicted = (outputs >= 0.5).int()
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
    
            val_loss /= len(val_loader)
            accuracy = 100 * correct / total
            epoch_accuracy_scores.append(accuracy)
    
            # Print validation metrics for this fold
            print(f"Fold {fold + 1}, Epoch {epoch + 1}: Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        if save_path is not None:
            #save epoch, model, optimizer, loss, fold
            torch.save({
                'epoch': epochs,
                'fold': fold,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, f"{save_path}/lstm_{hidden_size}_{fold}_{epochs}_{batch_size}_{lr}.pt")
        accuracy_scores.append(epoch_accuracy_scores[-1])

    #print average accuracy and standard deviation
    print(f"mean accuracy: {np.mean(accuracy_scores)}")
    print(f"standard deviation: {np.std(accuracy_scores)}")

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

    input_size = 100
    hidden_size = 100
    lstm = LSTMClassifier(input_size, hidden_size)

    epochs = 12
    batch_size = 32
    lr = 0.0001
    tfl = lambda model, train_embeddings: train_func_lstm(model, input_size, hidden_size, train_embeddings, \
                                                          epochs=epochs, batch_size=batch_size, lr=lr, save_path=f"lstm",  \
                                                            load_path=None)
    train_df.rename(columns={'text': 'embeddings'}, inplace=True)
    tfl(lstm, train_df)