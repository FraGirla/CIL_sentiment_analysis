import torch
from datasets import DatasetDict
from datasets import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
import torch.nn as nn
from transformers import DataCollatorWithPadding, get_scheduler, AutoTokenizer
import transformers
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.auto import tqdm
from utils import *
import wandb
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(device))
torch.cuda.empty_cache()
ts = int(time.time())


def tokenize_function(examples,tokenizer):
    """
    Tokenizes samples using the provided tokenizer.
    """
    return tokenizer(examples["text"], max_length=config.model.max_len, padding='max_length', truncation=True)

def train_loop(model, optimizer, lr_scheduler, train_dataloader, val_dataloader):
    """
    Training loop for a single model

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use during training.
        lr_scheduler: The learning rate scheduler.
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.

    Returns:
        float: Accuracy obtained by evaluating the model on the validation set during the last epoch.
    """

    if config.model.awp:
        awp = AWP(model=model,
        optimizer=optimizer,
        adv_lr=config.adversarial.adv_lr,
        adv_eps=config.adversarial.adv_eps,
        adv_epoch=config.adversarial.adv_epoch)

    for epoch in range(config.model.num_epochs):
        model.train()
        with tqdm(train_dataloader) as train_bar:
            train_bar.set_description(f"Epoch [{epoch+1}/{config.model.num_epochs}]")

            for batch in train_dataloader:
                batch = { k: v.to(device) for k, v in batch.items() }

                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                if config.model.awp:
                    awp.attack_backward(batch,epoch)
                
                if config.general.wandb:
                    wandb.log({"loss": loss})

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
        with tqdm(val_dataloader) as test_bar:
            test_bar.set_description(f"Validation")

            for count, batch in enumerate(val_dataloader):
                batch = { k: v.to(device) for k, v in batch.items() }
                with torch.no_grad():
                    outputs = model(**batch)

                logits = outputs.logits
                metric_count += torch.nn.functional.binary_cross_entropy(logits, batch['labels'])

                predictions = (logits >= threshold).int()
                correct_predictions += (predictions == batch['labels']).sum().item()
                total_predictions += len(batch['labels'])

                if config.general.wandb:
                    wandb.log({'acc': float(str(round(correct_predictions / total_predictions,7)))})

                test_bar.set_postfix({'loss': str(round((metric_count / (count+1)).item(),7)) , 'acc': str(round(correct_predictions / total_predictions,7))})
                test_bar.update(1)

            test_bar.close()
        if (epoch+1) == config.model.num_epochs:
            final_accuracy = (correct_predictions / total_predictions)


    torch.cuda.empty_cache()

    return final_accuracy


def training(dataset):
    """
    Tokenize dataset and train the model

    Args:
        dataset: The dataset to use for training.

    Returns:
        tuple: A tuple containing the trained model, accuracy on validation set, and validation dataloader (for further ensemble evaluation).
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    tokenized_datasets = dataset.map(lambda examples: tokenize_function(examples,tokenizer), batched=True)
    tokenized_datasets.set_format('torch', columns=["input_ids", "attention_mask", "label"] )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle = True, batch_size = config.model.batch_size, collate_fn = data_collator)
    val_dataloader = DataLoader(tokenized_datasets['test'], batch_size = config.general.test_batch, collate_fn = data_collator)

    model = CustomModel(checkpoint=config.model.name, num_labels=1, classifier_dropout=config.model.classification_dropout).to(device)
    config_layers(model,config.model.require_grad,config.model.lora,config.model.lora_params)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.model.lr)

    lr_scheduler = get_scheduler(
        'linear',
        optimizer = optimizer,
        num_warmup_steps=0,
        num_training_steps = config.model.num_epochs*len(train_dataloader),   
    )

    accuracy = train_loop(model, optimizer, lr_scheduler, train_dataloader, val_dataloader)

    del tokenized_datasets
    del train_dataloader
    del data_collator
    del optimizer
    del lr_scheduler
    return model, accuracy, val_dataloader

def evaluate_ensemble(models, val_dataloaders, weights):
    """
    Evaluate the ensemble of models on the validation data.

    Args:
        models (list): List of models to ensemble.
        test_dataloaders (list): List of validation dataloaders corresponding to each model.
        weights (list): List of weights corresponding to accuracies obtained for each model, useful if ensemble strategy is "weighted"

    Returns:
        float: The accuracy of the ensemble on the validation data.
    """
    for model in models:
        model.eval()

    correct_predictions = 0
    total_predictions = 0
    THRESHOLD = 0.5
    with tqdm(val_dataloaders[0]) as test_bar:
        test_bar.set_description(f"Evaluating Ensemble")
        iterators = [iter(data_loader) for data_loader in val_dataloaders]

        for _ in range(len(val_dataloaders[0])): #every batch in dataloader
            batches = []
            for i in range(len(val_dataloaders)):
                batch = next(iterators[i])
                batches.append({ k: v.to(device) for k, v in batch.items() })
            final_pred = torch.zeros(batches[0]["labels"].shape[0])
            
            for i, model in enumerate(models): #inference of every model
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
            
            if config.general.wandb:
                wandb.log({'acc': float(round(correct_predictions / total_predictions,7))})

            test_bar.set_postfix({'acc': str(round(correct_predictions / total_predictions,7))})
            test_bar.update(1)
        test_bar.close()

    return correct_predictions / total_predictions

def cross_val(train_df):
    """
    Perform k-fold cross-validation on the training dataset.

    Args:
        train_df (pandas.DataFrame): The training dataset.
    """

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
            val_dataloaders = []
            for model_cfg in config.ensemble.models:
                #set model config
                model_cfg = dictionary_to_namespace(model_cfg)
                print("Training model: ", model_cfg.name)
                config.model.name = model_cfg.name
                if not config.general.grid: 
                    config.model.batch_size = model_cfg.batch_size
                    config.model.lr = model_cfg.lr
                    config.model.num_epochs = model_cfg.num_epochs
                config.model.max_len = model_cfg.max_len
                config.model.classification_dropout = model_cfg.classification_dropout
                config.model.require_grad = model_cfg.require_grad
                config.model.awp = model_cfg.awp
                config.model.lora = model_cfg.lora
                config.model.lora_params = model_cfg.lora_params

                if config.general.wandb:
                    wandb.init( 
                        project='CIL_sentiment_analysis', 
                        job_type='train', 
                        name=f'{ts}_fold_{fold_}_{model_cfg.name}',
                        config=config.model)
                    wandb.config.fold = fold_
                    wandb.config.model = model_cfg.name

                #train model
                model, weight, val_dataloader = training(dataset)
                models.append(model)
                weights.append(weight)
                val_dataloaders.append(val_dataloader)
                if config.general.wandb:
                    wandb.finish()

            #evaluate ensemble
            if config.general.wandb:
                wandb.init( 
                    project='CIL_sentiment_analysis', 
                    job_type='eval', 
                    name=f'{ts}_evaluate_ensemble_fold_{fold_}')
                wandb.config.fold = fold_
                wandb.config.ensemble_strategy = config.ensemble.strategy
                model_names = ''
                for i, model_cfg in enumerate(config.ensemble.models):
                    model_cfg = dictionary_to_namespace(model_cfg)
                    model_names += model_cfg.name + ' '
                wandb.config.name = model_names
                accuracy = evaluate_ensemble(models, val_dataloaders, weights)
                wandb.finish()
            else:
                accuracy = evaluate_ensemble(models, val_dataloaders, weights)
            accuracies.append(accuracy)
            for model in models:
                del model
            del models

        else:
            if config.general.wandb:
                wandb.init( 
                        project='CIL_sentiment_analysis', 
                        job_type='train', 
                        name=f'fold_{fold_}',
                        config=model_cfg)
                wandb.config.fold = fold_
                wandb.config.model = config.model.name
            model, accuracy, _ = training(dataset)
            accuracies.append(accuracy)

            del model
        if config.general.wandb:
            wandb.finish()

        del train_fold
        del test_fold
        del dataset
        
            
    print("mean acc: ", sum(accuracies) / len(accuracies))
    print("std acc: ", np.std(accuracies))

if __name__ == '__main__':
    
    config = get_config('config.yaml')
    config = dictionary_to_namespace(config)
    set_seed(config.general.seed)

    if config.general.debug:
        transformers.logging.set_verbosity_debug()
    else:
        transformers.logging.set_verbosity_error()   
        os.environ["WANDB_SILENT"] = "true"


    #create pandas dataframe for training dataset
    train_df = pd.read_csv(config.general.train_path)


    if config.general.grid: #grid search 
        for learning_rate in config.grid.lr:
            for batch_size in config.grid.batch_size:
                for num_epoch in config.grid.num_epochs:
                    config.model.lr = learning_rate
                    config.model.batch_size = batch_size
                    config.model.num_epochs = num_epoch
                    print("lr: {}\nbatch_size: {}\nnum_epochs: {}".format(learning_rate,batch_size,num_epoch))
                    cross_val(train_df)
                    if config.general.wandb:
                        wandb.finish()
    else:
        cross_val(train_df)
        if config.general.wandb:
            wandb.finish()