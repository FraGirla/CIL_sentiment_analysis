
import torch
import torch.nn as nn
from transformers import AutoModel,AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from types import SimpleNamespace
import yaml 
import loralib as lora
import numpy as np
import random
import os

def get_config(config_path):
    with open(config_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def dictionary_to_namespace(data):
    if type(data) is list:
        return list(map(dictionary_to_namespace, data))
    elif type(data) is dict:
        sns = SimpleNamespace()
        for key, value in data.items():
            setattr(sns, key, dictionary_to_namespace(value))
        return sns
    else:
        return data

def update_to_lora(layer, r=8):
    in_features = layer.in_features
    out_features = layer.out_features

    pretrained_weight = layer.weight
    pretrained_bias = layer.bias
    lora_layer = lora.Linear(in_features, out_features, r=r, lora_alpha=r)
    lora_layer.weight = pretrained_weight
    lora_layer.bias = pretrained_bias

    #lora.mark_only_lora_as_trainable(model) # ricordatevi di fare questo prima di fare il training
    #num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # con questo controllate che il numero di parametri sia giusto
    return lora_layer

def freeze_layers(model, require_grad):
    for name, param in model.named_parameters():
        if 'encoder' in name or 'embeddings' in name:
            param.requires_grad = require_grad
    

class CustomModel(nn.Module):
    def __init__(self, checkpoint, num_labels, classifier_dropout = 0.1 ):
        super(CustomModel, self).__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(checkpoint, output_hidden_state = True )
        self.model = AutoModel.from_pretrained(checkpoint, config = self.config)
        self.dropout = nn.Dropout(classifier_dropout)
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
