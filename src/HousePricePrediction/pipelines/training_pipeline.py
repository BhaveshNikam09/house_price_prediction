from src.HousePricePrediction.components.data_ingestion import dataingestion
from src.HousePricePrediction.components.data_transformation import DataTranformation
from src.HousePricePrediction.components.model_trainer import ModelTrainer

import os       
import sys 
from src.HousePricePrediction.logger import logging
from src.HousePricePrediction.exception import custom_exception

import pandas as pd     

data_ingestion= dataingestion()
train_data_path,test_data_path=data_ingestion.initate_data_ingetion()

data_transformation=DataTranformation()
train_arr,test_arr=data_transformation.initialize_data_tranform(train_data_path,test_data_path)

model=ModelTrainer()
model.initate_model_trainig(train_arr,test_arr)
