#import needed packages
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt


class paramTune():
    
    #--- Inputs Explained
    #--- df             => dataframe that must have at least one predictor column and a target variable column
    #--- target_name    => name of the target column, entered as text (case sensitive)
    #--- tree_range     => List of two integers that designate the bounds for n_estimators in the XGBoost algorithm
    #--- lr_range       => List of two numbers that designate the bounds of the learning rate in the XGBoost algorithm
    #--- depth_range    => List of two integers that designate the bounds of the tree depth in the XGBoost algorithm
    #--- test_split     => Size of test data set as either pct or record count
    
    def __init__(self,df,target_name,test_split):
        
        self.df = df
        self.target_name = target_name
        self.test_split = test_split
        
        #create test and train data based on user inputed test_split
        self.y = df[target_name] 
        self.X = df.drop([target_name],axis=1)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=test_split,random_state=29)
      
    #method to train and score a model based on algorithm name and algorithm parameters
    def trainAndScore(self,algo_name,**kwargs):
        
        #instantiate and train model
        temp_model = algo_name(**kwargs)
        temp_model.fit(self.X_train,self.y_train)
        
        #make model predictions
        train_pred = temp_model.predict(self.X_train)
        test_pred = temp_model.predict(self.X_test)
        
        return train_pred, test_pred
    
    #method to calculate model performance based on metric function and predicted and actual values
    def modelPerf(self,metric_func,y,yhat):   
        return metric_func(self,y,yhat)
    
    #RMSE method    
    def rmse(self,y,yhat):
        return sqrt(mean_squared_error(y, yhat))
    
    #Method to append to performance DataFrame
    def appendPerfDF(self,temp_df_all,perf_dict_name,sort_name='test_perf'):
        
        if hasattr(self,perf_dict_name) == False:
            setattr(self,perf_dict_name,temp_df_all)
        else:
            temp_df_attr =  getattr(self,perf_dict_name)
            temp_df_attr = temp_df_attr.append(temp_df_all,ignore_index=True)
            setattr(self,perf_dict_name,temp_df_attr)
        
        #sort dataframe by metric of interest
        temp_df_sort = getattr(self,perf_dict_name)
        temp_df_sort = temp_df_sort.sort_values(by=sort_name)
        setattr(self,perf_dict_name,temp_df_sort)
        
        
    #probably will cut this up into multiple methods, since training/scoring, calculating metrics
    #and saving results in a dataframe are going to be the same for each search type
    #Methods to create:
    #1. Train and Score Model
    #2. Calculate performance metrics
    #3. Create or append to performance DF
    def random_search(self,n_iters,algo_name,perf_dict_name,perf_Func=rmse,**kwargs,):
        
        
        for h in range(0,n_iters):
            
        
            #I need to make a provision to make sure that duplicate searches are not done
            
            #loop through all tuning parameters provided and assign a random value to each
            for i in kwargs:
                
                temp_kwargs = {}
                
                temp_lower = kwargs[i][0]
                temp_upper = kwargs[i][1]
                
                if kwargs[i][2] == 'int':
                    temp_rand = np.random.randint(low=temp_lower,high=temp_upper)
                
                else:
                    temp_rand = np.random.uniform(low=temp_lower,high=temp_upper)
                
                
                
                temp_kwargs[i] = temp_rand
                
            #train and score model
            train_pred, test_pred = self.trainAndScore(algo_name,**temp_kwargs)

            #calculate model performance
            train_rmse = self.modelPerf(perf_Func,self.y_train,train_pred)
            test_rmse = self.modelPerf(perf_Func,self.y_test, test_pred)
            
            #add performance to dictionary that has tuning parameters
            temp_kwargs['train_perf'] = train_rmse
            temp_kwargs['test_perf'] = test_rmse
            
            #put results and parameters into dataframe
            temp_df = pd.DataFrame.from_records(temp_kwargs,index=[0])
            
            
            #append to temp_df_all if it exists, if it doesn't, create it    
            try:
                temp_df_all = temp_df_all.append(temp_df, ignore_index=True)
            except:
                temp_df_all = pd.DataFrame(temp_df)

        #add performance to of tuning to df attribute or append to it
        #   depending on if an attribute dataframe exists
        self.appendPerfDF(temp_df_all,perf_dict_name)
            
        return
    
    
test_df = pd.read_csv('sample_data/test_data.csv')   

test_tune = paramTune(test_df, 'recordCount', 0.3)

test_tune.random_search(5,XGBRegressor,'first_test',n_estimators=[10,20,'int'])
test_tune.random_search(5,XGBRegressor,'first_test',n_estimators=[10,20,'int'])

print(test_tune.first_test)





