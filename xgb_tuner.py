#import needed packages
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt


class XGB_tune():
    
    #--- Inputs Explained
    #--- df             => dataframe that must have at least one predictor column and a target variable column
    #--- target_name    => name of the target column, entered as text (case sensitive)
    #--- tree_range     => List of two integers that designate the bounds for n_estimators in the XGBoost algorithm
    #--- lr_range       => List of two numbers that designate the bounds of the learning rate in the XGBoost algorithm
    #--- depth_range    => List of two integers that designate the bounds of the tree depth in the XGBoost algorithm
    #--- test_split     => Size of test data set as either pct or record count
    
    def __init__(self,df,target_name,tree_range,lr_range,depth_range,test_split):
        
        self.df = df
        self.target_name = target_name
        self.tree_range = tree_range
        self.lr_range = lr_range
        self.depth_range = depth_range
        
        #create test and train data based on user inputed test_split
        self.y = df[target_name] 
        self.X = df.drop([target_name],axis=1)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=test_split,random_state=29)
        
        
        
    def random_search(n_iters):
        
        self.rand_perf_df = pd.DataFrame()
        
        for i in range(0,n_iters):
        
            #create random starting space
            rand_trees = np.random.randint(low=tree_range[0],high=tree_range[1])
            rand_lr = np.random.uniform(low=lr_range[0],high=lr_range[1])
            rand_depth = np.random.randint(low=depth_range[0],high=depth_range[1])
        
            #instantiate XGBoost and train model
            temp_model = XGBRegressor(n_estimators= rand_trees,learning_rate=rand_lr,max_depth=rand_depth)
            temp_model.fit(self.X_train,self.y_train)
            
            #make model predictions
            train_pred = temp_model.predict(self.X_train)
            test_pred = temp_model.predict(self.X_test)
            
            
            #caulcate rmse's
            train_rmse = sqrt(mean_squared_error(self.y_train, train_pred))
            test_rmse = sqrt(mean_squared_error(self.y_test, test_pred)) 
            
            #put into rand_perf_df dataframe
            self.rand_perf_df.append({'n_estimators':rand_trees},{'learning_rate':rand_lr},{'max_depth':rand_depth},{'train_rmse':train_rmse},{'test_rmse':test_rmse})
        
        self.rand_per_df = self.rand_per_df.sort_values(by='test_rmse')
        return self.rand_per_df
        

test_df = pd.read_csv('sample_data/test_data.csv')     
        