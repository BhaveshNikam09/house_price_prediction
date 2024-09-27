import os      
import sys 
from src.HousePricePrediction.exception import custom_exception
from src.HousePricePrediction.logger import logging

import pandas as pd 
import numpy as np      
from sklearn.model_selection import train_test_split   
from dataclasses import dataclass
from pathlib import path

class DataIngestionConfig :
    raw_data_path:str=os.path.join('artifacts','raw.csv')
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    

class dataingestion:
    def __init__(self):
        
        self.ingestionconfig=DataIngestionConfig()
        
    
    def initate_data_ingetion(self):
        logging.info('data ingestion started')
        try:
            data=pd.read_csv('D:/ML/House_Price_Prediction/notebooks/data/Housing.csv')
            logging.info('csv file read successfully as df')

            #making dir to save the data files 
            os.makedirs(os.path.dirname(os.path.join(self.ingestionconfig.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestionconfig.raw_data_path,index=False)
            logging.info('data saved successfully')
            
            logging.info('spliting the data in train and test')
            
            train_data,test_data=train_test_split(data,test_size=0.10,random_state=2)
            #splitinng done 
            
            logging.info('saving the data ')
            train_data.to_csv(self.ingestionconfig.train_data_path,index=False)
            test_data.to_csv(self.ingestionconfig.test_data_path,index=False)
            
            logging.info('data ingestion completed')
            
            return(
                self.ingestionconfig.train_data_path,
                self.ingestionconfig.test_data_path
            )
            
        except Exception as e:
            raise custom_exception(e,sys)