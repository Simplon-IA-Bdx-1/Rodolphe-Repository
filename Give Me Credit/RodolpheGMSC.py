# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:50:32 2019

@author: Rodolphe
"""
# IMPORT MODULES

import sklearn as sk
import numpy as np
import pandas as pd
from pandas import read_csv

# API BIGML : 

import bigml.api
from bigml.api import BigML

api = BigML ('rodolphevdl','a47cb24c1d3a04762b4b5db6ece5e4c915efa86c',project ='project/5defb2acf80b166cce0012c1')

# LOAD DATA SET into BIG ML 

source_train = api.create_source('train.csv')
source_test = api.create_source('test.csv')

# DATASET CREATION

dataset_train = api.create_dataset(source_train) 
datatest_test = api.create_dataset(source_test)

# TRAIN / VAL SPLIT CREATION

trainset = api.create_dataset(dataset_train, {"name": "Train 80% ", "sample_rate": 0.8})
valset = api.create_dataset(dataset_train, {"name": "Validation 20% ", "sample_rate": 0.8, "out_of_bag": True})

# MODEL ENSEMBLE

ensemble_args = {"objective_field": "SeriousDlqin2yrs"}
ensemble = api.create_ensemble(trainset, ensemble_args)

# EVALUATION ENSEMBLE

evaluation = api.create_evaluation(ensemble, valset)

# BATCH PREDICTION ON VAL  

prediction_args = {"name": "prediction"}
batch_prediction = api.create_batch_prediction(ensemble, valset, {
    "header": True,
    "all_fields": True,
    "probabilities": True})

# BATCH PREDICTION ON TEST
prediction_args = {"name": "prediction"}
batch_prediction_test = api.create_batch_prediction(ensemble, datatest_test, {
    "header": True,
    "all_fields": True,
    "probabilities": True})

# IMPORT CSV BATCH PREDICTION 

api.download_batch_prediction(batch_prediction, filename='Prediction_VAL.csv')
api.download_batch_prediction(batch_prediction, filename='Prediction_TEST.csv')

# DATA PREPARATION FOR KAGGLE [ID / PROBABILITY] - DOWNLOAD PREP FILE FOR SUBMISSION TO KAGGLE

df = read_csv('Prediction_TEST.csv')
df_2 = df[['1 probability']]
df_2.index = np.arange(1,len(df_2)+1)
df_2.index.names = ['Id']

df_2.rename(columns ={'1 probability' : 'Probability'}, inplace = True)
df_2.to_csv('Prediction_TEST.csv')

# SUBMISSION TO KAGGLE
import kaggle

# API KAGGLE

%env KAGGLE_USERNAME=setsin
%env KAGGLE_KEY=26cbe7b5aa9036281f4eb50023c0b353

# SUBMISSION FILE
submission_file = 'Prediction_TEST.csv'
# SEND FILE TO KAGGLE COMPETION
kaggle.api.competition_submit('Prediction_TEST.csv', "Ensemble", "GiveMeSomeCredit")





