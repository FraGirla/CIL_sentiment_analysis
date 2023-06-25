# %%
#!pip install transformers
#!pip install evaluate
#!pip install datasets
#!pip install datasets[audio]


# %%
#module load gcc/8.2.0 python_gpu/3.10.4 r/4.0.2 git-lfs/2.3.0 eth_proxy npm/6.14.9 libsndfile/1.0.23

# %%
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
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer
import torch.nn as nn
from transformers import AutoModel,AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler
import torch.optim as optim

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# %%
# First make the kfold object
folds = StratifiedKFold(n_splits=5)

#create dataframes for train and test
train_df = pd.read_csv('preprocessed/train_full.csv')
#drop nan
train_df.dropna(inplace=True)
print(train_df.info())
folds_datasets = []
accuracies = []
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
# %%
print(torch.cuda.get_device_name(device))

class CustomModel(nn.Module):
    def __init__(self, checkpoint, num_labels ):
        super(CustomModel, self).__init__()
        self.num_labels = num_labels

        self.model = AutoModel.from_pretrained(checkpoint, config = AutoConfig.from_pretrained(checkpoint, output_hidden_state = True ) )
        self.dropout = nn.Dropout(0.15)
        self.classifier = nn.Linear(768, num_labels )

    def forward(self, input_ids = None, attention_mask=None, labels = None , weight = None):
        outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)

        last_hidden_state = outputs[0]
        sequence_outputs = self.dropout(last_hidden_state)

        logits = self.classifier(sequence_outputs[:, 0, : ].view(-1, 768 ))
        logits = torch.sigmoid(logits)
        criterion = torch.nn.BCELoss(reduction='none')
        if labels is not None and weight is not None:
          loss = criterion(logits,labels).squeeze()
          if weight is not None:
            loss = loss*(weight/weight.sum())
          return TokenClassifierOutput(loss=loss.sum(), logits=logits, hidden_states=last_hidden_state)
        else:
          return TokenClassifierOutput(loss=None, logits=logits, hidden_states=last_hidden_state)

MAX_LEN=128
NUM_MODELS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1
THRESHOLD = 0.5

hugging_face_model = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(hugging_face_model)


def tokenize_function(examples):
    return tokenizer(examples["partial_clean_tweet"], max_length=MAX_LEN, padding='max_length')

#get folds
for fold_, (train_index, test_index) in enumerate(folds.split(train_df, train_df['label'])):
    print ("Fold {}".format(fold_))
    train_fold = train_df.iloc[train_index]
    test_fold = train_df.iloc[test_index]
    #remove unused columns
    train_fold = train_fold.drop(columns=['tweet', 'tokenized_tweet', 'tokenized_tweet_no_stopwords', 'text'])
    test_fold = test_fold.drop(columns=['tweet', 'tokenized_tweet', 'tokenized_tweet_no_stopwords', 'text'])

    dataset = DatasetDict({'train': Dataset.from_pandas(train_fold), 'test': Dataset.from_pandas(test_fold)})
    # %%

    dataset = dataset.map(lambda x: {"label": [float(-1) if x["label"] == 0 else float(1) ]})
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # %%
    tokenized_datasets["train"] = tokenized_datasets["train"].shuffle(seed=42)
    tokenized_datasets["test"] = tokenized_datasets["test"].shuffle(seed=42)

    tokenized_datasets["train"].set_format('torch', columns=["input_ids", "attention_mask", "label", "weight"] )
    tokenized_datasets["test"].set_format('torch', columns=["input_ids", "attention_mask", "label"] )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # %%

    weak_models = [CustomModel(checkpoint=hugging_face_model, num_labels=1).to(device) for _ in range(NUM_MODELS)]

    alpha = []
    for i,model in enumerate(weak_models):
        print("\nTrain model ", i)
        train_dataloader = DataLoader(tokenized_datasets['train'], batch_size = BATCH_SIZE, collate_fn = data_collator)
        num_training_steps = len(train_dataloader)
        optimizer = optim.AdamW(model.parameters(), lr = LEARNING_RATE )
        lr_scheduler = get_scheduler(
            'linear',
            optimizer = optimizer,
            num_warmup_steps=0,
            num_training_steps = num_training_steps,
        )

        for epoch in range(NUM_EPOCHS):
            progress_bar_train = tqdm(range(len(train_dataloader)))
            model.train()
            for batch in train_dataloader:
                batch = { k: v.to(device) for k, v in batch.items() }
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar_train.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
                progress_bar_train.update(1)
        
        print("Evaluate model to update weights")
        model.eval()

        weights_tot = tokenized_datasets['train']['weight'].numpy().squeeze()
        pred_tot = []
        labels_tot = tokenized_datasets['train']['label'].numpy().squeeze()
        progress_bar_val = tqdm(range(len(train_dataloader)))
        for batch in train_dataloader:
            batch = { k: v.to(device) for k, v in batch.items() }
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            predictions = (logits >= THRESHOLD).int()
            predictions[predictions == 0] = -1
            pred_tot.extend(predictions.cpu().numpy().tolist())

            progress_bar_val.update(1)

        pred_tot = np.array(pred_tot).squeeze()
        total_error = weights_tot[pred_tot != labels_tot].sum()
        print("Total error = ",total_error)
        if total_error > 1e-10 and total_error < 1 - 1e-10:
            a = 0.5*np.log((1-total_error)/total_error)
            alpha.append(a)
            weights_tot = weights_tot*np.exp((-1)*a*pred_tot*labels_tot)
            weights_tot = weights_tot/weights_tot.sum()
        


        tokenized_datasets["train"] = tokenized_datasets["train"].remove_columns("weight")
        tokenized_datasets["train"] = tokenized_datasets["train"].add_column("weight", weights_tot)
    print("Alpha values ", alpha)


    test_dataloader = DataLoader(tokenized_datasets['test'], batch_size = BATCH_SIZE, collate_fn = data_collator)
    correct_predictions = 0
    total_predictions = 0

    for model in weak_models:
        model.eval()
    progress_bar_test = tqdm(range(len(test_dataloader)))

    metric_count = 0
    for batch in test_dataloader:
        batch = { k: v.to(device) for k, v in batch.items() }
        final_pred = torch.zeros(batch["labels"].shape[0])
        
        for i, model in enumerate(weak_models):
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            predictions = (logits >= THRESHOLD).int().cpu()
            predictions[predictions == 0] = -1
            final_pred = final_pred + alpha[i]*predictions.squeeze()
            final_pred[final_pred >= 0] = 1
            final_pred[final_pred < 0] = -1

        correct_predictions += (final_pred == batch['labels'].cpu()).sum().item()
        total_predictions += len(batch['labels'])

        progress_bar_test.update(1)
    print("TEST ACC: ", correct_predictions / total_predictions)


    # %%
    del train_fold
    del test_fold
    del dataset
    del tokenized_datasets
    del data_collator
    del train_dataloader
    del test_dataloader
    del weak_models
    del optimizer
    del lr_scheduler
    del progress_bar_train
    del progress_bar_test
    del progress_bar_val
    torch.cuda.empty_cache()


print("mean acc: ", sum(accuracies) / len(accuracies))
print("std acc: ", np.std(accuracies))




