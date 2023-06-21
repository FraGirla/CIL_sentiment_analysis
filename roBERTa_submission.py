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
test_df = pd.read_csv('preprocessed/test_full.csv')
#print nan values test
print(test_df.info())

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
# %%
print(torch.cuda.get_device_name(device))

dataset = DatasetDict({'test': Dataset.from_pandas(test_df)})
# %%
MAX_LEN=512
from transformers import AutoTokenizer

hugging_face_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(hugging_face_model)


def tokenize_function(examples):
    return tokenizer(examples["clean_tweet"], max_length=MAX_LEN, padding='max_length',)


tokenized_datasets = dataset.map(tokenize_function, batched=True)


from transformers import DataCollatorWithPadding

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
        else:
            return TokenClassifierOutput(loss=None, logits=logits, hidden_states=last_hidden_state)

# %%
from torch.utils.data import DataLoader

test_dataloader = DataLoader(
    tokenized_datasets['test'], batch_size = 32, collate_fn = data_collator
)

model = CustomModel(checkpoint=hugging_face_model, num_labels=1).to(device)
model.load_state_dict(torch.load("roBERTa_12_layers_no_emb_light_pre_inference.pt"))

from tqdm.auto import tqdm
threshold = 0.5
correct_predictions = 0
total_predictions = 0

model.eval()
progress_bar_test = tqdm(range(len(test_dataloader) ))

metric_count = 0
final_predictions = []
for batch in test_dataloader:
    batch = { k: v.to(device) for k, v in batch.items() }
    with torch.no_grad():
        outputs = model(**batch)
        
    logits = outputs.logits
    
    predictions = (logits >= threshold).int()  
    final_predictions.extend(predictions.cpu().numpy().tolist())
    progress_bar_test.update(1)

predictions_pd = pd.DataFrame({"Predictions": np.array(final_predictions).ravel()})
predictions_pd['Predictions'] = predictions_pd['Predictions'].replace(0, -1)
predictions_pd.index = np.arange(1, len(predictions_pd) + 1)
predictions_pd.index.name = "Id"
predictions_pd.to_csv("Prediction.csv")







