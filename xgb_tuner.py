#import needed packages
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBRegressor


class XGB_tune():
    
    def __init__(self,df,tree_range,lr_range,depth_range):
        
        self.df = df
        self.tree_range = tree_range
        self.lr_range = lr_range
        self.depth_range = depth_range
        
    def random_search(n_iters):
        
        
        
        