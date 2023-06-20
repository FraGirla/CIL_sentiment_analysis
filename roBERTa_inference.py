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
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#create dataframes for train and test
train_df = pd.read_csv('preprocessed/train_full.csv')
test_df = pd.read_csv('preprocessed/test_full.csv')
#drop nan
train_df.dropna(inplace=True)
print(train_df.info())
#print nan values test
print(test_df.info())

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
# %%
print(torch.cuda.get_device_name(device))

dataset = DatasetDict({'train': Dataset.from_pandas(train_df), 'test': Dataset.from_pandas(test_df)})
# %%
MAX_LEN=512
from transformers import AutoTokenizer

hugging_face_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(hugging_face_model)


def tokenize_function(examples):
    return tokenizer(examples["clean_tweet"], max_length=MAX_LEN, padding='max_length',)


dataset['train'] = dataset.map(lambda x: {"label": [float(x["label"])]})
tokenized_datasets = dataset.map(tokenize_function, batched=True)
# %%
tokenized_datasets["train"] = tokenized_datasets["train"].shuffle(seed=42)
#tokenized_datasets["validation"] = tokenized_datasets["validation"].shuffle(seed=42)
tokenized_datasets["test"] = tokenized_datasets["test"]


from transformers import DataCollatorWithPadding

tokenized_datasets["train"].set_format('torch', columns=["input_ids", "attention_mask", "label"] )
tokenized_datasets["test"].set_format('torch', columns=["input_ids", "attention_mask"] )

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# %%
import torch.nn as nn
from transformers import AutoModel,AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
class CustomModel(nn.Module):
    def __init__(self, checkpoint, num_labels ):
        super(CustomModel, self).__init__()
        self.num_labels = num_labels
        
        self.model = AutoModel.from_pretrained(checkpoint, config = AutoConfig.from_pretrained(checkpoint, output_hidden_state = True ) )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels )
        
    def forward(self, input_ids = None, attention_mask=None, labels = None ):
        outputs = self.model(input_ids = input_ids, attention_mask = attention_mask  )
        
        last_hidden_state = outputs[0]
        sequence_outputs = self.dropout(last_hidden_state)
        
        logits = self.classifier(sequence_outputs[:, 0, : ].view(-1, 768 ))
        logits = torch.sigmoid(logits)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.binary_cross_entropy(logits, labels)
            return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=last_hidden_state)

# %%
from torch.utils.data import DataLoader

BATCH_SIZE = 32 
train_dataloader = DataLoader(
    tokenized_datasets['train'], shuffle = True, batch_size = BATCH_SIZE, collate_fn = data_collator
)

#eval_dataloader = DataLoader(
#    tokenized_datasets['validation'], shuffle = True, batch_size = BATCH_SIZE, collate_fn = data_collator
#)

test_dataloader = DataLoader(
    tokenized_datasets['test'], batch_size = 32, collate_fn = data_collator
)

model = CustomModel(checkpoint=hugging_face_model, num_labels=1).to(device)

# %%
for name, param in model.named_parameters():
    if "encoder.layer.0"  in name:
        param.requires_grad=True
    elif "encoder.layer.1."  in name:
        param.requires_grad=True
    elif "encoder.layer.2"  in name:
        param.requires_grad=True
    elif "encoder.layer.3"  in name:
        param.requires_grad=True
    elif "encoder.layer.4"  in name:
        param.requires_grad=True
    elif "encoder.layer.5"  in name:
        param.requires_grad=True
    elif "encoder.layer.6"  in name:
        param.requires_grad=True
    elif "encoder.layer.7"  in name:
        param.requires_grad=True
    elif "encoder.layer.8"  in name:
        param.requires_grad=True
    elif "encoder.layer.9"  in name:
        param.requires_grad=True
    elif "encoder.layer.10"  in name:
        param.requires_grad=True
    elif "encoder.layer.11"  in name:
        param.requires_grad=True
    elif "embeddings"  in name:
        param.requires_grad=False
    else:
        #print(name)
        pass

# %%
from transformers import get_scheduler

import torch.optim as optim
optimizer = optim.AdamW(model.parameters(), lr=5e-5)


num_epoch = 1

num_training_steps = num_epoch * len(train_dataloader)

lr_scheduler = get_scheduler(
    'linear',
    optimizer = optimizer,
    num_warmup_steps=0,
    num_training_steps = num_training_steps,
    
)


# %%
from tqdm.auto import tqdm

progress_bar_train = tqdm(range(num_training_steps))

for epoch in range(num_epoch):
    model.train()
    for batch in train_dataloader:
        batch = { k: v.to(device) for k, v in batch.items() }
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar_train.update(1)

# %%
torch.save(model.state_dict(), "roBERTa_12_layers_no_emb_light_pre_inference.pt")

# %%
threshold = 0.5
correct_predictions = 0
total_predictions = 0

model.eval()
progress_bar_test = tqdm(range(num_epoch * len(test_dataloader) ))

metric_count = 0
final_predictions = []
for batch in test_dataloader:
    batch = { k: v.to(device) for k, v in batch.items() }
    with torch.no_grad():
        outputs = model(**batch)
        
    logits = outputs.logits
    metric_count += torch.nn.functional.binary_cross_entropy(logits, batch['labels'])
    
    predictions = (logits >= threshold).int()  
    final_predictions.extend(predictions.cpu().numpy().tolist())
    progress_bar_test.update(1)
predictions_pd = pd.DataFrame({"Predictions": np.array(final_predictions)})
predictions_pd['Predictions'] = predictions_pd['Predictions'].replace(0, -1)
predictions_pd.index = np.arange(1, len(predictions_pd) + 1)
predictions_pd.index.name = "Id"
predictions_pd.to_csv("Prediction.csv")







