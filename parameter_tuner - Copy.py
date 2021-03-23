#import needed packages
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import math
import itertools
import warnings


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
    def random_search(self,n_iters,algo_name,perf_dict_name,perf_Func=rmse,round_num=4,**kwargs,):
        

        #keep track of randomly selected parameters to not waste time repeating
        rand_history = []
        
        temp_df_all = pd.DataFrame()
        
        iters = 0
        
        #for h in range(0,n_iters):
        while len(temp_df_all) < n_iters and iters < 3*n_iters:   
            
            #increment iters
            iters += 1
            
            #loop through all tuning parameters provided and assign a random value to each
            temp_kwargs = {}
            
            #create a list to hold randomly selected data
            temp_rand_history = []
            
            for i in kwargs:

                temp_lower = kwargs[i][0]
                temp_upper = kwargs[i][1]
                
                if kwargs[i][2] == 'int':
                    temp_rand = np.random.randint(low=temp_lower,high=temp_upper)

                else:
                    
                    temp_rand = np.random.uniform(low=temp_lower,high=temp_upper)
                    
                    #round continuous
                    temp_rand = round(temp_rand,round_num)
                    
                temp_rand_history.append(temp_rand)
                    
                #I need to add a test here that if the combinations of the tuning parameters has been
                # done already, recalculate random numbers and add one iteration back (if possible)

                
                temp_kwargs[i] = temp_rand
             

            #check if these parameters have been tested yet            
            if temp_rand_history not in rand_history:
                    
                #add current test to list of parameters that have been tested
                rand_history.append(temp_rand_history)
                print(rand_history)
                
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
                #try:
                #    temp_df_all = temp_df_all.append(temp_df, ignore_index=True)
                #except:
                #    temp_df_all = pd.DataFrame(temp_df)
                
                #append performance results to temp_df_all
                temp_df_all = temp_df_all.append(temp_df, ignore_index=True)
    
        #add performance to of tuning to df attribute or append to it
        #   depending on if an attribute dataframe exists
        self.appendPerfDF(temp_df_all,perf_dict_name)
        
        if len(temp_df_all) < n_iters:
            warnings.warn('randomSearch did not test as many parameters as requested because not enough unique parameter combinations were found.')
            
        return
    
    
    def gridSearch(self,max_iters,algo_name,perf_dict_name,perf_Func=rmse,**kwargs):
        
        
        #calculate number of splits per parameter given a maximum number of iterations explored       
        param_cuts = (math.floor(math.exp((math.log(max_iters)/len(kwargs))))-1)
        
         
        #put an error if param_cuts is equal to 0 that says something about
        #adding more max interations
                                
        #cut data into grids
        
        all_params_list = []
        
        for i in kwargs:
            
            temp_list = []
            
            temp_lower = kwargs[i][0]
            temp_upper = kwargs[i][1]
            
            temp_range = temp_upper - temp_lower
            
            temp_step = temp_range/param_cuts
            
            temp_list.append(temp_lower)
            
            temp_value = temp_lower
            
            #create a list of values that will be tested for each parameter
            for j in range(0,param_cuts):
                
                temp_value = temp_value + temp_step
                
                
                #if integer parameter, round down
                if kwargs[i][2] == 'int':
                                    
                    temp_value_rounded = round(temp_value)
                    
                    temp_list.append(temp_value_rounded)
                else:
                    #round to 10 decimal places, maybe not appropriate for all algorithms
                    temp_value_rounded = round(temp_value,10)
                    temp_list.append(temp_value_rounded)
            
            
            all_params_list.append(temp_list)
            
        
        #Put tuning parameters into a temporary kwargs and pass through the aglorithm
        all_combos = list(itertools.product(*all_params_list))
        
        #if len(all_combos) < 100
        
        for i in all_combos:
            temp_kwargs = kwargs.copy()
            for j,k in zip(range(0,len(i)),kwargs.keys()):
            
                temp_kwargs[k] = i[j]
            
            #train and score model
            train_pred, test_pred = self.trainAndScore(algo_name,**temp_kwargs)
            
            #calculate model performance
            train_perf = self.modelPerf(perf_Func,self.y_train,train_pred)
            test_perf = self.modelPerf(perf_Func,self.y_test, test_pred)
            
            #add performance to dictionary that has tuning parameters
            temp_kwargs['train_perf'] = train_perf
            temp_kwargs['test_perf'] = test_perf
            
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
            
            #print out how many soultions were explored
        print('%s Solutions Explored' % (len(all_combos)))
        
        return
    
    #Create functions for neighborhood based heuristics
    #later I want the user to have to option to make specific step sizes for each
    #parameter, but for now they will just be able to make a % change for each one
    def createNbrhood(self,current_solution,step_size=0.1,round_num=4,**kwargs):
        
        move_list = []
        current_solution_list = []
        #loop through all parameters to create a new nbrhood
        for i in current_solution:
            
            #get max and min for the current parameter
            temp_min = kwargs[i][0]
            temp_max = kwargs[i][1]
            value_range = temp_max - temp_min
            
            temp_current = current_solution[i]
            current_solution_list.append(temp_current)
            
            
            temp_current_lower = temp_current-(value_range*(1+step_size))
            temp_current_upper = temp_current+(value_range*(1+step_size))
            
            #round integers 
            if kwargs[i][2] == 'int':
                
                temp_current_lower = round(temp_current_lower)
                temp_current_upper = round(temp_current_upper)
                
                if temp_current_lower == temp_current:
                    temp_current_lower = temp_current -1
                
                if temp_current_upper == temp_current:
                    temp_current_upper = temp_current + 1
                    
            else:
                temp_current_lower = round(temp_current_lower,round_num)
                temp_current_upper = round(temp_current_upper,round_num)
                
                    
            
            #check if these values are outside of the bounds
            if temp_current_lower < temp_min:
                temp_current_lower = temp_min
                
            if temp_current_upper > temp_max:
                temp_current_upper = temp_max
                
            move_list.append([temp_current_lower,temp_current,temp_current_upper])
            
        nbrhood = list(itertools.product(*move_list))
        
        #remove the current solution from the nbr list
        nbrhood.remove(tuple(current_solution_list))
        
        return nbrhood
        
        
                    
            
                
                
            
            
    def startSol(self,round_num,**kwargs):
        
        temp_kwargs = {}
        
        
        for i in kwargs:
            

            temp_lower = kwargs[i][0]
            temp_upper = kwargs[i][1]
                
            if kwargs[i][2] == 'int':
                temp_rand = np.random.randint(low=temp_lower,high=temp_upper)

            else:
                    
                temp_rand = np.random.uniform(low=temp_lower,high=temp_upper)
                    
                #round continuous
                temp_rand = round(temp_rand,round_num)
                

            temp_kwargs[i] = temp_rand
            
            
        return temp_kwargs
        
            
    def hillClimb(self,algo_name,perf_dict_name,restarts,perf_Func=rmse,step_size=0.10,round_num=4,**kwargs):
        
        #create a random starting solution function
        startSolution = self.startSol(round_num=round_num,**kwargs)
        globalBestNbr = startSolution
        
        print(globalBestNbr)
        
        train_pred, test_pred = self.trainAndScore(algo_name,**globalBestNbr)
        
        train_rmse = self.modelPerf(perf_Func,self.y_train,train_pred)
        test_rmse = self.modelPerf(perf_Func,self.y_test, test_pred)        
        
        
        
        globalBestPerf = test_rmse
        
        obs = 0
    
        #loop through all restarts
        for j in range(0,restarts):
            
            if j == 1:
                currentSolution = startSolution
            else:
                currentSolution = self.startSol(round_num=round_num,**kwargs)
            
            done = 0
            
            while done == 0:
                
                bestNbr = currentSolution
        
                train_pred, test_pred = self.trainAndScore(algo_name,**bestNbr)
                bestPerf = self.modelPerf(perf_Func,self.y_test, test_pred) 
                
                temp_kwargs = kwargs.copy()
                
                for i in self.createNbrhood(currentSolution,step_size,**kwargs):
                    
                    obs += 1
                    
                    for j, k in zip(range(0,len(i)),kwargs.keys()):
                        
                        temp_kwargs[k] = i[j]
                        
                        
                    currentTrainPred, currentTestPred = self.trainAndScore(algo_name,**temp_kwargs)
                    currentTrainrmse = self.modelPerf(perf_Func,self.y_train,currentTrainPred)
                    currentTestrmse = self.modelPerf(perf_Func,self.y_test, currentTestPred) 
                    
                    #print(currentTestrmse)
                    
                    temp_df = temp_kwargs.copy()
                    
                    #add performance to dictionary that has tuning parameters
                    temp_df['train_perf'] = currentTrainrmse
                    temp_df['test_perf'] = currentTestrmse
                        
                    
                
                    #put results and parameters into dataframe
                    temp_df = pd.DataFrame.from_records(temp_df,index=[0])
                        
                    #append to temp_df_all if it exists, if it doesn't, create it    
                    try:
                        temp_df_all = temp_df_all.append(temp_df, ignore_index=True)
                    except:
                        temp_df_all = pd.DataFrame(temp_df)
                        
                        
                    if currentTestrmse < bestPerf:
                        bestNbr = temp_kwargs
                        print(temp_kwargs)
                        bestPerf = currentTestrmse
                        print(bestPerf)
                        break
                
                
                        
                if currentSolution == bestNbr:
                    done = 1
                    print('restart')
                else:
                    currentSolution = bestNbr

            
            if bestPerf < globalBestPerf:
                globalBestPerf = bestPerf
                globalBestNbr = bestNbr
                print('new best value found (%s)' % (globalBestPerf))
        
        print('Best RMSE Found')
        print(globalBestPerf)
        
        print('Number of solutions explored = %s'%(obs))
                
        return [globalBestPerf,globalBestNbr,obs]
                
            
                
                
        
        
    
    
#test the code
test_df = pd.read_csv('sample_data/test_data.csv')   

test_tune = paramTune(test_df, 'recordCount', 0.3)

#test_tune.random_search(5,XGBRegressor,'first_test',n_estimators=[10,20,'int'],
#                        learning_rate=[0.25,0.1,'cont'],
#                        max_depth=[2,5,'int'])

#test_tune.random_search(20,XGBRegressor,'first_test',n_estimators=[10,11,'int'],
#                        max_depth=[2,3,'int'])
#test_tune.random_search(5,XGBRegressor,'first_test',n_estimators=[10,20,'int'],learning_rate=[0.25,0.1,'cont'])


#test_tune.gridSearch(350,XGBRegressor,'temp_dict_name',n_estimators=[10,20,'int'],
#                        learning_rate=[0.1,0.25,'cont'],
#                        max_depth=[2,10,'int'])

return_values = test_tune.hillClimb(XGBRegressor,'temp_dict',2,n_estimators=[10,20,'int'],
                        learning_rate=[0.1,0.25,'cont'],
                        max_depth=[2,10,'int'],step_size=0.5)

#print(test_tune.startSol(round_num=4,n_estimators=[10,20,'int'],learning_rate=[0.25,0.1,'cont'],max_depth=[2,5,'int']))

print(return_values)

print('done')



