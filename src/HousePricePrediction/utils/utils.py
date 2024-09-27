import os
import sys 
import pandas as pd    
import numpy as np      

from src.HousePricePrediction.logger import logging
from src.HousePricePrediction.exceotion import custom_exception

from sklearn.metrics import r2_score,mean_absolute_error,mean_sqaured_error

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as fileobj:
            pickle.dump(obj, file_path)        
    
    except Exception as e :
        raise custom_exception(e,sys)
    

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report=[]
        for i in range(len(models)):
            model = list(models.values())[i]
            #model train
            model.fit(X_train,y_train)
            
            #prediction
            y_pred=model.predict(X_test)
            
            #test_model_score
            test_model_score=r2_score(y_test,y_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return reprot   
    
    except Exception as e:
        logging.info("exception occured in the evaluate models")
        raise custom_exception(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj) 
        
    except Exception as e:
        raise custom_exception(e,sys)   