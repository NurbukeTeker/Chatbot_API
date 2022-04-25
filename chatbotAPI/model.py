# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:44:52 2022

@author: nurbuketeker
"""

import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.preprocessing import LabelEncoder
from intent_dict import acid_dict, atis_dict, clinc_dict, bank_dict

device = torch.device("cpu")

from transformers import AutoModelForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

labelencoder = LabelEncoder()

class_dictionary={0 :"atis_api",2:"acid_api", 1: "banking_api",3:"clinc_api"}
 
def getModelPrediction(text,pytorch_model):
    test_texts_ = [text]
    
    input_ids = []
    attention_masks = []
    
    for text in test_texts_:
        encoded_dict = tokenizer.encode_plus(
                            text,                     
                            add_special_tokens = True, 
                            max_length = 20,          
                            pad_to_max_length = True,
                            return_attention_mask = True,  
                            return_tensors = 'pt',   
                       )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
            
   
        
    test_labels_ = labelencoder.fit_transform( [1])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(test_labels_.tolist())
    
    batch_size = 32  
    
    prediction_data = TensorDataset(input_ids, attention_masks,labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    
    print('Prediction started on test data')
    pytorch_model.eval()
    predictions , true_labels = [], []
    
    
    for batch in prediction_dataloader:
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_input_mask, b_labels = batch
    
      with torch.no_grad():
          outputs = pytorch_model(b_input_ids, token_type_ids=None, 
                          attention_mask=b_input_mask)
    
      logits = outputs[0]
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to("cpu").numpy()
      
      predictions.append(logits)
      true_labels.append(label_ids)
    
    print('Prediction completed')
    
    prediction_set = []
    
    for i in range(len(true_labels)):
      pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
      prediction_set.append(pred_labels_i)
    
    prediction_scores = [item for sublist in prediction_set for item in sublist]
    return prediction_scores

def getDomainPrediction(text):
    pytorch_model = AutoModelForSequenceClassification.from_pretrained("my_multidomain_model")

    prediction_scores = getModelPrediction(text,pytorch_model)

    print(prediction_scores)
    prediction_class = class_dictionary[prediction_scores[0]]
    
    return prediction_scores[0], prediction_class
   
        
def getACIDModel( text):
    pytorch_model = AutoModelForSequenceClassification.from_pretrained("acid_model")
    dict_name=acid_dict
    prediction_scores = getModelPrediction(text,pytorch_model)
    return dict_name[prediction_scores[0]]
  
    
def getATISModel( text):
    pytorch_model = AutoModelForSequenceClassification.from_pretrained("atis_model")
    dict_name=atis_dict
    prediction_scores = getModelPrediction(text,pytorch_model)
    return dict_name[prediction_scores[0]]


def getBANKModel( text):
    pytorch_model = AutoModelForSequenceClassification.from_pretrained("banking_model")
    dict_name=bank_dict
    prediction_scores = getModelPrediction(text,pytorch_model) 
    return dict_name[prediction_scores[0]]

    
def getCLINCModel( text):
    pytorch_model = AutoModelForSequenceClassification.from_pretrained("clinc_model")
    dict_name=clinc_dict
    prediction_scores = getModelPrediction(text,pytorch_model)
    return dict_name[prediction_scores[0]]

