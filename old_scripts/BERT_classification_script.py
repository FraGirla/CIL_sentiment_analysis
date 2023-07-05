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
    def __init__(self,
                 model,
                 optimizer,
                 *,
                 adv_param='weight',
                 adv_lr=0.001,
                 adv_eps=0.001,
                 adv_epoch=2):

        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.adv_epoch = adv_epoch
        self.adv_started = False
        self.backup = {}

    def perturb(self, epoch):
        """
        Perturb model parameters for AWP gradient
        Call before loss and loss.backward()
        """
        if (epoch+1) >= self.adv_epoch:
            if not self.adv_started:
                print('AWP: Start perturbing')
                self.adv_started = True

            self._save()  # save model parameters
            self._attack_step()  # perturb weights

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                grad = self.optimizer.state[param]['exp_avg']
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.detach())

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Set lower and upper limit in change
                    limit_eps = self.adv_eps * param.detach().abs()
                    param_min = param.data - limit_eps
                    param_max = param.data + limit_eps

                    # Perturb along gradient
                    # w += (adv_lr * |w| / |grad|) * grad
                    param.data.add_(grad, alpha=(self.adv_lr * (norm_data + e) / (norm_grad + e)))

                    # Apply the limit to the change
                    param.data.clamp_(param_min, param_max)

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.clone().detach()
                else:
                    self.backup[name].copy_(param.data)

    def restore(self):
        """
        Restore model parameter to correct position; AWP do not perturbe weights, it perturb gradients
        Call after loss.backward(), before optimizer.step()
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])

# Kfold object
folds = StratifiedKFold(n_splits=5)

#create pandas dataframe for training dataset
train_df = pd.read_csv('preprocessed/train_full.csv')

#drop nan
train_df.dropna(inplace=True)

print(train_df.info)

#remove unused columns
train_df = train_df.drop(columns=['tweet', 'tokenized_tweet', 'tokenized_tweet_no_stopwords', 'text'])

MAX_LEN=128
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
ADVERSARIAL_LR = 0.00001
ADVERSARIAL_EPS = 0.001
ADVERSARIAL_EPOCH_STARTS = 2
hugging_face_model = "vinai/bertweet-large"

def tokenize_function(examples):
    return tokenizer(examples["partial_clean_tweet"], max_length=MAX_LEN, padding='longest',)

accuracies = []
for fold_, (train_index, test_index) in enumerate(folds.split(train_df, train_df['label'])):
    print ("Fold {}".format(fold_))
    train_fold = train_df.iloc[train_index]
    test_fold = train_df.iloc[test_index]

    dataset = DatasetDict({'train': Dataset.from_pandas(train_fold), 'test': Dataset.from_pandas(test_fold)})

    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(100000))
    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(50000))

    tokenizer = AutoTokenizer.from_pretrained(hugging_face_model)

    dataset = dataset.map(lambda x: {"label": [float(x["label"])]})
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets.set_format('torch', columns=["input_ids", "attention_mask", "label"] )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle = True, batch_size = BATCH_SIZE, collate_fn = data_collator)

    test_dataloader = DataLoader(tokenized_datasets['test'], batch_size = BATCH_SIZE, collate_fn = data_collator)

    model = CustomModel(checkpoint=hugging_face_model, num_labels=1).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    lr_scheduler = get_scheduler(
        'linear',
        optimizer = optimizer,
        num_warmup_steps=0,
        num_training_steps = NUM_EPOCHS*len(train_dataloader),   
    )
    awp = AWP(model=model,
            optimizer=optimizer,
            adv_lr=ADVERSARIAL_LR,
            adv_eps=ADVERSARIAL_EPS,
            adv_epoch=ADVERSARIAL_EPOCH_STARTS)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        with tqdm(train_dataloader) as train_bar:
            train_bar.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
            for batch in train_dataloader:
                batch = { k: v.to(device) for k, v in batch.items() }
                awp.perturb(epoch)
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                
                awp.restore()
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
    
    torch.cuda.empty_cache()


print("Mean accuracy: ", sum(accuracies) / len(accuracies))
print("Std accuracy: ", np.std(accuracies))




