import os
import sys
import pandas as pd 
import numpy as np
 
from dataclasses import dataclass
from src.HousePricePrediction.logger import logging
from src.HousePricePrediction.exception import custom_exception

from sklearn.compose import ColumnTransformer
from sklearn.impute import  SimpleImputer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.HousePricePrediction.utils.utils import save_object

@dataclass
class DataTranformationConfig:
    processor_obj_file=os.path.join('artifacts','preprocessor.pkl')
    

class DataTranformation:
    def __init__(self):
        self.data_trans_config=DataTranformationConfig()
        
    def get_data_tranform(self):
        try:
            logging.info('intiated data tranformation')
            
            #defing the num and cat column
            num_column=['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
            cat_column=['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                        'airconditioning', 'prefarea', 'furnishingstatus']
            
            #defing custom cate
            mainroad=['no','yes']
            guestroom=['no','yes']
            basement=['no','yes']
            hotwaterheating=['no','yes']
            airconditioning=['no','yes']
            prefarea=['no','yes']
            furnishingstatus=['unfurnished','semi-furnished','furnished'] 
            
            #defining pipeline
            logging.info('pipeline initiated')
            
            num_pipeline=Pipeline(
                
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scalinfg',StandardScaler())
                ]
            )
            
            cat_pipeline=Pipeline(
    
                steps=[
                        ('imputer',SimpleImputer(strategy='most_frequent')),
                        ('encoding',OrdinalEncoder(categories=[mainroad,basement,guestroom,airconditioning,
                                               hotwaterheating,prefarea,furnishingstatus])),
                        ('scaler',StandardScaler())
                    ]
                )
            
            
            preprocessor = ColumnTransformer(
                     transformers=[
                        ('num', num_pipeline, num_column),
                        ('cat', cat_pipeline, cat_column)
                         ]
                    )

            return preprocessor
        
        except Exception as e:
            raise custom_exception(e,sys)
        

    def initialize_data_tranform(self,train_path,test_path):
        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)
            
            logging.info('test and train data access')
            
            preprocessor_obj=self.get_data_tranform()
            
            target_column= 'price'
            
            input_feature_train_data=train_data.drop(columns=target_column,axis=1)
            target_feature_train_data=train_data[target_column]
            
            input_featue_test_data=test_data.drop(columns=target_column,axis=1)
            target_feature_test_data=test_data[target_column]
            
            input_train_data_arr=preprocessor_obj.fit_transform(input_feature_train_data)
            input_test_data_arr=preprocessor_obj.transform(input_featue_test_data)
            logging.info("Applying preprocessing object on training and testing datasets.")
            
            train_arr = np.concatenate([input_train_data_arr, np.array(target_feature_train_data).reshape(-1, 1)], axis=1)
            test_arr = np.concatenate([input_test_data_arr, np.array(target_feature_test_data).reshape(-1, 1)], axis=1)
            
            save_object(
                file_path=self.data_trans_config.processor_obj_file,
                obj=preprocessor_obj
            )
            logging.info("preprocessing pickle file saved")
            
            return(
                train_arr,
                test_arr
            )
        
        except Exception as e:
            raise custom_exception(e,sys)
    