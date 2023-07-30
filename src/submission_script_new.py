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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(device))
torch.cuda.empty_cache()


def tokenize_function(examples,tokenizer):
    """
    Tokenizes samples using the provided tokenizer.
    """
    return tokenizer(examples["text"], max_length=config.model.max_len, padding='max_length', truncation=True)

def train_loop(model, optimizer, lr_scheduler, train_dataloader):
    """
    Training loop for a single model

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use during training.
        lr_scheduler: The learning rate scheduler.
        train_dataloader: DataLoader for training data.

    Returns:
        None
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
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                train_bar.update(1)
            train_bar.close()
    torch.cuda.empty_cache()

def training(dataset):
    """
    Tokenize dataset and train the model

    Args:
        dataset: The entire dataset: train + test 

    Returns:
        tuple: A tuple containing the trained model and the test dataloader (for generating predictions with ensemble technique).
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    tokenized_datasets = dataset.map(lambda examples: tokenize_function(examples,tokenizer), batched=True)
    tokenized_datasets['train'].set_format('torch', columns=["input_ids", "attention_mask", "label"] )
    tokenized_datasets['test'].set_format('torch', columns=["input_ids", "attention_mask"] )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle = True, batch_size = config.model.batch_size, collate_fn = data_collator)
    test_dataloader = DataLoader(tokenized_datasets['test'], batch_size = config.general.test_batch, collate_fn = data_collator)
    
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

    train_loop(model, optimizer, lr_scheduler, train_dataloader)

    del tokenized_datasets
    del train_dataloader
    del data_collator
    del optimizer
    del lr_scheduler
    return model, test_dataloader

def generate_predictions_ensemble(models, test_dataloaders, weights):
    """
    Generate predictions using ensemble technique.

    Args:
        models (list): List of trained models.
        test_dataloaders (list): List of test dataloaders.
        weights (list): List of weights for the models (for weighted ensemble).

    Returns:
        list: List of final predictions.
    """
    for model in models:
        model.eval()
    THRESHOLD = 0.5
    final_predictions = []
    with tqdm(test_dataloaders[0]) as test_bar:
        test_bar.set_description(f"Evaluating Ensemble")
        iterators = [iter(data_loader) for data_loader in test_dataloaders]
        for _ in range(len(test_dataloaders[0])): #every batch in dataloader
            batches = []
            for i in range(len(test_dataloaders)):
                batch = next(iterators[i])
                batches.append({ k: v.to(device) for k, v in batch.items() })
            final_pred = torch.zeros(batches[0]["input_ids"].shape[0])
            
            for i, model in enumerate(models): #inference of every model
                with torch.no_grad():
                    outputs = model(**batches[i])

                predictions = outputs.logits.cpu()

                if config.ensemble.strategy == "avg":
                    predictions = torch.clamp(2*predictions-1,-1,1)
                    final_pred += predictions.squeeze()
                elif config.ensemble.strategy == "vote":
                    predictions = (predictions >= THRESHOLD).int()
                    predictions[predictions == 0] = -1
                    final_pred += predictions.squeeze()
                elif config.ensemble.strategy == "weighted":
                    predictions = torch.clamp(2*predictions-1,-1,1)
                    final_pred += predictions.squeeze() * weights[i]

            final_pred[final_pred >= 0] = 1
            final_pred[final_pred < 0] = -1
            final_predictions.extend(final_pred.int().numpy().tolist())
            test_bar.update(1)
        test_bar.close()
    return final_predictions

def generate_predictions(model, test_dataloader):
    """
    Generate predictions using a single model.

    Args:
        model (torch.nn.Module): The trained model.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for test data.

    Returns:
        list: List of final predictions.
    """
    model.eval()
    threshold = 0.5
    final_predictions = []
    with tqdm(test_dataloader) as test_bar:
        test_bar.set_description(f"Validation")
        for count, batch in enumerate(test_dataloader):
            batch = { k: v.to(device) for k, v in batch.items() }
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits

            predictions = (logits >= threshold).int()
            final_predictions.extend(predictions.cpu().numpy().tolist())
            test_bar.update(1)
        test_bar.close()

    return final_predictions



if __name__ == '__main__':

    #Load the configuration from 'config.yaml'
    config = get_config('config.yaml')
    config = dictionary_to_namespace(config)
    set_seed(config.general.seed)

    if config.general.debug:
        transformers.logging.set_verbosity_debug()
    else:
        transformers.logging.set_verbosity_error()   


    #create pandas dataframe for training dataset
    train_df = pd.read_csv(config.general.train_path)
    test_df = pd.read_csv(config.general.test_path)

    dataset = DatasetDict({'train': Dataset.from_pandas(train_df), 'test': Dataset.from_pandas(test_df)})

    #shuffle training data to ensure that the training batches are more representative of the dataset
    if config.general.use_subsampling:
        dataset["train"] = dataset["train"].shuffle(seed=config.general.seed).select(range(config.subsampling.train))
        dataset["test"] = dataset["test"].select(range(config.subsampling.test))
    else:
        dataset["train"] = dataset["train"].shuffle(seed=config.general.seed)
    dataset['train'] = dataset['train'].map(lambda x: {"label": [float(x[config.general.column_label_name])]})

    if config.general.ensemble:
        models = []
        weights = config.inference.weights
        test_dataloaders = []
        for model_cfg in config.ensemble.models:
            #configs for every model
            model_cfg = dictionary_to_namespace(model_cfg)
            print("Training model: ", model_cfg.name)

            config.model.name = model_cfg.name
            config.model.batch_size = model_cfg.batch_size
            config.model.lr = model_cfg.lr
            config.model.num_epochs = model_cfg.num_epochs
            config.model.max_len = model_cfg.max_len
            config.model.classification_dropout = model_cfg.classification_dropout
            config.model.require_grad = model_cfg.require_grad
            config.model.awp = model_cfg.awp
            config.model.lora = model_cfg.lora
            config.model.lora_params = model_cfg.lora_params

            model, test_dataloader = training(dataset)
            models.append(model)
            test_dataloaders.append(test_dataloader)
    
        predictions = generate_predictions_ensemble(models, test_dataloaders, weights)
        for model in models:
            del model
        del models
    else:
        model, test_dataloader = training(dataset)
        predictions = generate_predictions(model, test_dataloader)
    
    #save predictions on a csv file
    predictions_pd = pd.DataFrame({"Prediction": np.array(predictions).ravel()})
    predictions_pd['Prediction'] = predictions_pd['Prediction'].replace(0, -1)
    predictions_pd.index = np.arange(1, len(predictions_pd) + 1)
    predictions_pd.index.name = "Id"
    predictions_pd.to_csv("Prediction.csv")
