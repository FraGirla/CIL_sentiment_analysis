import torch
from datasets import DatasetDict
from datasets import Dataset
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
import torch.nn as nn

from transformers import DataCollatorWithPadding, get_scheduler, AutoTokenizer
import transformers
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.auto import tqdm
from utils import *


os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(device))
torch.cuda.empty_cache()


def tokenize_function(examples,tokenizer):
    return tokenizer(examples["text"], max_length=config.model.max_len, padding='longest',)

def train_loop(model, optimizer, lr_scheduler, train_dataloader, test_dataloader):

    if config.general.awp:
        awp = AWP(model=model,
        optimizer=optimizer,
        adv_lr=config.adversarial.adv_lr,
        adv_eps=config.adversarial.adv_eps,
        adv_epoch=config.adversarial.adv_epoch)

    for epoch in range(config.general.num_epochs):
        model.train()
        with tqdm(train_dataloader) as train_bar:
            train_bar.set_description(f"Epoch [{epoch+1}/{config.general.num_epochs}]")
            for batch in train_dataloader:
                batch = { k: v.to(device) for k, v in batch.items() }

                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                if config.general.awp:
                    awp.attack_backward(batch,epoch)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                train_bar.update(1)
            train_bar.close()

        threshold = 0.5
        correct_predictions = 0
        total_predictions = 0
        metric_count = 0

        model.eval()
        with tqdm(test_dataloader) as test_bar:
            test_bar.set_description(f"Validation")
            for count, batch in enumerate(test_dataloader):
                batch = { k: v.to(device) for k, v in batch.items() }
                with torch.no_grad():
                    outputs = model(**batch)

                logits = outputs.logits
                metric_count += torch.nn.functional.binary_cross_entropy(logits, batch['labels'])

                predictions = (logits >= threshold).int()
                correct_predictions += (predictions == batch['labels']).sum().item()
                total_predictions += len(batch['labels'])
                test_bar.set_postfix({'loss': str(round((metric_count / (count+1)).item(),7)) , 'acc': str(round(correct_predictions / total_predictions,7))})
                test_bar.update(1)
            test_bar.close()
        if (epoch+1) == config.general.num_epochs:
            final_accuracy = (correct_predictions / total_predictions)


    torch.cuda.empty_cache()

    return final_accuracy


def training(dataset):
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)

    tokenized_datasets = dataset.map(lambda examples: tokenize_function(examples,tokenizer), batched=True)

    tokenized_datasets.set_format('torch', columns=["input_ids", "attention_mask", "label"] )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle = True, batch_size = config.general.batch_size, collate_fn = data_collator)

    test_dataloader = DataLoader(tokenized_datasets['test'], batch_size = config.general.batch_size, collate_fn = data_collator)

    model = CustomModel(checkpoint=config.model.name, num_labels=1, classifier_dropout=config.model.classification_dropout).to(device)

    freeze_layers(model,config.model.require_grad)

    optimizer = optim.AdamW(model.parameters(), lr=config.general.lr)

    lr_scheduler = get_scheduler(
        'linear',
        optimizer = optimizer,
        num_warmup_steps=0,
        num_training_steps = config.general.num_epochs*len(train_dataloader),   
    )

    accuracy = train_loop(model, optimizer, lr_scheduler, train_dataloader, test_dataloader)

    del tokenized_datasets
    del train_dataloader
    del data_collator
    del optimizer
    del lr_scheduler
    return model, accuracy, test_dataloader

def evaluate_ensemble(models, test_dataloaders, weights):
    for model in models:
        model.eval()
    correct_predictions = 0
    total_predictions = 0
    THRESHOLD = 0.5
    with tqdm(test_dataloaders[0]) as test_bar:
        test_bar.set_description(f"Evaluating Ensemble")
        for batch_number in range(len(test_dataloaders[0])):
            batches = []
            for i in range(len(test_dataloaders)):
                batch = next(iter(test_dataloaders[i]))
                batches.append({ k: v.to(device) for k, v in batch.items() })
            final_pred = torch.zeros(batches[0]["labels"].shape[0])
            
            for i, model in enumerate(models):
                with torch.no_grad():
                    outputs = model(**batches[i])
                logits = outputs.logits.cpu()
                predictions = logits
                if config.ensemble.strategy == "avg":
                    predictions = torch.clamp(2*predictions-1,-1,1)
                    final_pred += predictions.squeeze()
                elif config.ensemble.strategy == "vote":
                    predictions = (logits >= THRESHOLD).int()
                    predictions[predictions == 0] = -1
                    final_pred += predictions.squeeze()
                elif config.ensemble.strategy == "weighted":
                    predictions = torch.clamp(2*predictions-1,-1,1)
                    final_pred += predictions.squeeze() * weights[i]

            final_pred[final_pred >= 0] = 1
            final_pred[final_pred < 0] = -1
            labels = batches[i]["labels"].cpu().squeeze()
            labels[labels == 0] = -1
            correct_predictions += (final_pred == labels).sum().item()
            total_predictions += len(batches[i]['labels'])
            test_bar.set_postfix({'acc': str(round(correct_predictions / total_predictions,7))})
            test_bar.update(1)
        test_bar.close()
    return correct_predictions / total_predictions

def cross_val(train_df):
    accuracies = []

    folds = StratifiedKFold(n_splits=config.general.n_folds)
    for fold_, (train_index, test_index) in enumerate(folds.split(train_df, train_df['label'])):
        print ("Fold {}".format(fold_))
        train_fold = train_df.iloc[train_index]
        test_fold = train_df.iloc[test_index]
        dataset = DatasetDict({'train': Dataset.from_pandas(train_fold), 'test': Dataset.from_pandas(test_fold)})
        
        if config.general.use_subsampling:
            dataset["train"] = dataset["train"].shuffle(seed=config.general.seed).select(range(config.subsampling.train))
            dataset["test"] = dataset["test"].shuffle(seed=config.general.seed).select(range(config.subsampling.test))
        else:
            dataset["train"] = dataset["train"].shuffle(seed=config.general.seed)
            dataset["test"] = dataset["test"].shuffle(seed=config.general.seed)
        dataset = dataset.map(lambda x: {"label": [float(x[config.general.column_label_name])]})

        if config.general.ensemble:
            models = []
            weights = []
            test_dataloaders = []
            for model_name in config.ensemble.model_names:
                print("Training model: ", model_name)
                config.model.name = model_name
                model, weight, test_dataloader = training(dataset)
                models.append(model)
                weights.append(weight)
                test_dataloaders.append(test_dataloader)
        
            accuracy = evaluate_ensemble(models, test_dataloaders, weights)
            accuracies.append(accuracy)
            for model in models:
                del model
            del models

        else:
            
            model, accuracy, _ = training(dataset)
            accuracies.append(accuracy)

            del model

        del train_fold
        del test_fold
        del dataset
        
            
    print("mean acc: ", sum(accuracies) / len(accuracies))
    print("std acc: ", np.std(accuracies))

if __name__ == '__main__':
    
    config = get_config('config.yaml')
    config = dictionary_to_namespace(config)

    if config.general.debug:
        transformers.logging.set_verbosity_debug()
    else:
        transformers.logging.set_verbosity_error()   


    #create pandas dataframe for training dataset
    train_df = pd.read_csv(config.general.train_path)

    #drop nan
    train_df.dropna(inplace=True)

    #drop every column except text and label
    train_df = train_df[[config.general.column_text_name,config.general.column_label_name]]
    #rename columns
    train_df.columns = ['text','label']

    if config.general.grid:
        for learning_rate in config.grid.lr:
            for batch_size in config.grid.batch_size:
                for num_epoch in config.grid.num_epochs:
                    config.general.lr = learning_rate
                    config.general.batch_size = batch_size
                    config.general.num_epochs = num_epoch
                    print("lr: {}\nbatch_size: {}\nnum_epochs: {}".format(learning_rate,batch_size,num_epoch))
                    cross_val(train_df)
    else:
        cross_val(train_df)