#!pip install transformers
#!pip install evaluate
#!pip install datasets
#!pip install datasets[audio]

#module load gcc/8.2.0 python_gpu/3.10.4 r/4.0.2 git-lfs/2.3.0 eth_proxy npm/6.14.9 libsndfile/1.0.23

import torch
from datasets import load_dataset
from datasets import DatasetDict
from datasets import Dataset
from transformers import AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from transformers import TrainingArguments, Trainer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
import torch.nn as nn
from transformers import AutoModel,AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import get_scheduler
import torch.optim as optim
from tqdm.auto import tqdm
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(device))
torch.cuda.empty_cache()

class CustomModel(nn.Module):
    def __init__(self, checkpoint, num_labels ):
        super(CustomModel, self).__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(checkpoint, output_hidden_state = True )
        self.model = AutoModel.from_pretrained(checkpoint, config = self.config)
        self.dropout = nn.Dropout(0.15)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels )
        
    def forward(self, input_ids = None, attention_mask=None, labels = None ):
        outputs = self.model(input_ids = input_ids, attention_mask = attention_mask  )
        
        last_hidden_state = outputs[0]
        sequence_outputs = self.dropout(last_hidden_state)
        
        logits = self.classifier(sequence_outputs[:, 0, : ].view(-1, self.config.hidden_size))
        logits = torch.sigmoid(logits)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.binary_cross_entropy(logits, labels)
            return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=last_hidden_state)
        else:
            return TokenClassifierOutput(loss=None, logits=logits, hidden_states=last_hidden_state)

class AWP:
    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=0.00001,
        adv_eps=0.001,
        adv_epoch=2,
        adv_step=1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.adv_epoch = adv_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, batch, epoch):
        if (self.adv_lr == 0) or (epoch+1 < self.adv_epoch):
            return None

        self._save() 
        for _ in range(self.adv_step):
            self._attack_step() 
            outputs = self.model(**batch)
            loss = outputs.loss
            self.optimizer.zero_grad()
            loss.backward()
            
        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

# Kfold object
folds = StratifiedKFold(n_splits=5)

#create pandas dataframe for training dataset
train_df = pd.read_csv('preprocessed/train_full.csv')

print("Dataset loaded")

#drop nan
train_df.dropna(inplace=True)

#remove unused columns
train_df = train_df.drop(columns=['tweet', 'tokenized_tweet', 'tokenized_tweet_no_stopwords', 'text'])

MAX_LEN=128
NUM_EPOCHS = 2
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
ADVERSARIAL_LR = 0.00001
ADVERSARIAL_EPS = 0.001
ADVERSARIAL_EPOCH_STARTS = 2
hugging_face_model = "vinai/bertweet-base"

def tokenize_function(examples):
    return tokenizer(examples["partial_clean_tweet"], max_length=MAX_LEN, padding='longest',)

accuracies = []
accuracies_awp = []
for fold_, (train_index, test_index) in enumerate(folds.split(train_df, train_df['label'])):
    print ("Fold {}".format(fold_))
    train_fold = train_df.iloc[train_index]
    test_fold = train_df.iloc[test_index]

    dataset = DatasetDict({'train': Dataset.from_pandas(train_fold), 'test': Dataset.from_pandas(test_fold)})

    dataset["train"] = dataset["train"].shuffle(seed=42)
    dataset["test"] = dataset["test"].shuffle(seed=42)

    tokenizer = AutoTokenizer.from_pretrained(hugging_face_model)

    dataset = dataset.map(lambda x: {"label": [float(x["label"])]})
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets.set_format('torch', columns=["input_ids", "attention_mask", "label"] )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle = True, batch_size = BATCH_SIZE, collate_fn = data_collator)

    test_dataloader = DataLoader(tokenized_datasets['test'], batch_size = BATCH_SIZE, collate_fn = data_collator)

    model = CustomModel(checkpoint=hugging_face_model, num_labels=1).to(device)
    model_awp = CustomModel(checkpoint=hugging_face_model, num_labels=1).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    optimizer_awp = optim.AdamW(model_awp.parameters(), lr=LEARNING_RATE)

    lr_scheduler = get_scheduler(
        'linear',
        optimizer = optimizer,
        num_warmup_steps=0,
        num_training_steps = NUM_EPOCHS*len(train_dataloader),   
    )
    lr_scheduler_awp = get_scheduler(
        'linear',
        optimizer = optimizer_awp,
        num_warmup_steps=0,
        num_training_steps = NUM_EPOCHS*len(train_dataloader),   
    )

    awp = AWP(model=model_awp,
            optimizer=optimizer_awp,
            adv_lr=ADVERSARIAL_LR,
            adv_eps=ADVERSARIAL_EPS,
            adv_epoch=ADVERSARIAL_EPOCH_STARTS)
    
    print("NO AWP:")
    for epoch in range(NUM_EPOCHS):
        model.train()
        with tqdm(train_dataloader) as train_bar:
            train_bar.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
            for batch in train_dataloader:
                batch = { k: v.to(device) for k, v in batch.items() }
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

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
                test_bar.set_postfix(loss = metric_count / (count+1), acc = correct_predictions/total_predictions)
                test_bar.update(1)
            test_bar.close()
        if (epoch+1) == NUM_EPOCHS:
            print("Validation Accuracy: ", correct_predictions / total_predictions)
            print("Validation BinaryCrossEntropy: ", metric_count / len(test_dataloader))
            accuracies.append(correct_predictions / total_predictions)

    print("WITH AWP:")
    for epoch in range(NUM_EPOCHS):
        model_awp.train()
        with tqdm(train_dataloader) as train_bar:
            train_bar.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
            for batch in train_dataloader:
                batch = { k: v.to(device) for k, v in batch.items() }
                outputs = model_awp(**batch)
                loss = outputs.loss
                loss.backward()
                
                awp.attack_backward(batch,epoch)

                optimizer_awp.step()
                lr_scheduler_awp.step()
                optimizer_awp.zero_grad()
                train_bar.update(1)
            train_bar.close()
        threshold = 0.5
        correct_predictions = 0
        total_predictions = 0
        metric_count = 0

        model_awp.eval()
        with tqdm(test_dataloader) as test_bar:
            test_bar.set_description(f"Validation")
            for count, batch in enumerate(test_dataloader):
                batch = { k: v.to(device) for k, v in batch.items() }
                with torch.no_grad():
                    outputs = model_awp(**batch)

                logits = outputs.logits
                metric_count += torch.nn.functional.binary_cross_entropy(logits, batch['labels'])

                predictions = (logits >= threshold).int()
                correct_predictions += (predictions == batch['labels']).sum().item()
                total_predictions += len(batch['labels'])
                test_bar.set_postfix(loss = metric_count / (count+1), acc = correct_predictions/total_predictions)
                test_bar.update(1)
            test_bar.close()
        if (epoch+1) == NUM_EPOCHS:
            print("Validation Accuracy: ", correct_predictions / total_predictions)
            print("Validation BinaryCrossEntropy: ", metric_count / len(test_dataloader))
            accuracies_awp.append(correct_predictions / total_predictions)

    del train_fold
    del test_fold
    del dataset
    del tokenized_datasets
    del data_collator
    del train_dataloader
    del test_dataloader
    del model
    del optimizer
    del lr_scheduler
    del model_awp
    del optimizer_awp
    del lr_scheduler_awp
    del awp  
    torch.cuda.empty_cache()




print("Mean accuracy no AWP: ", sum(accuracies) / len(accuracies))
print("Std accuracy no AWP: ", np.std(accuracies))

print("Mean accuracy with AWP: ", sum(accuracies_awp) / len(accuracies_awp))
print("Std accuracy with AWP: ", np.std(accuracies_awp))



