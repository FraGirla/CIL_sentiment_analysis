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
    torch.cuda.empty_cache()

def training(dataset):
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)

    tokenized_datasets = dataset.map(lambda examples: tokenize_function(examples,tokenizer), batched=True)

    tokenized_datasets['train'].set_format('torch', columns=["input_ids", "attention_mask", "label"] )
    tokenized_datasets['test'].set_format('torch', columns=["input_ids", "attention_mask"] )

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

    train_loop(model, optimizer, lr_scheduler, train_dataloader, test_dataloader)

    del tokenized_datasets
    del train_dataloader
    del data_collator
    del optimizer
    del lr_scheduler
    return model, test_dataloader

def generate_predictions_ensemble(models, test_dataloaders, weights):
    for model in models:
        model.eval()
    THRESHOLD = 0.5
    final_predictions = []
    with tqdm(test_dataloaders[0]) as test_bar:
        test_bar.set_description(f"Evaluating Ensemble")
        for batch_number in range(len(test_dataloaders[0])):
            batches = []
            for i in range(len(test_dataloaders)):
                batch = next(iter(test_dataloaders[i]))
                batches.append({ k: v.to(device) for k, v in batch.items() })
            final_pred = torch.zeros(batches[0]["input_ids"].shape[0])
            
            for i, model in enumerate(models):
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
    
    config = get_config('config.yaml')
    config = dictionary_to_namespace(config)

    if config.general.debug:
        transformers.logging.set_verbosity_debug()
    else:
        transformers.logging.set_verbosity_error()   


    #create pandas dataframe for training dataset
    train_df = pd.read_csv(config.general.train_path)
    test_df = pd.read_csv(config.general.test_path)
    #drop nan
    train_df.dropna(inplace=True)

    #drop every column except text and label
    train_df = train_df[[config.general.column_text_name,config.general.column_label_name]]
    test_df = test_df[[config.general.column_text_name]]
    #rename columns
    train_df.columns = ['text','label']
    test_df.columns = ['text']

    dataset = DatasetDict({'train': Dataset.from_pandas(train_df), 'test': Dataset.from_pandas(test_df)})
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
        for model_name in config.ensemble.model_names:
            print("Training model: ", model_name)
            config.model.name = model_name
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
    
    predictions_pd = pd.DataFrame({"Prediction": np.array(predictions).ravel()})
    predictions_pd['Prediction'] = predictions_pd['Prediction'].replace(0, -1)
    predictions_pd.index = np.arange(1, len(predictions_pd) + 1)
    predictions_pd.index.name = "Id"
    predictions_pd.to_csv("Prediction.csv")
