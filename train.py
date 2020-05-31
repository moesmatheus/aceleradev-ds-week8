import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, median_absolute_error
from catboost import CatBoostRegressor, Pool
# from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV
# from sklearn.svm import SVR
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.base import clone

# from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
# from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

import warnings
warnings.filterwarnings("ignore")

from preprocess import preProcess


def create_models(features, target, models_path = 'models.pkl',
     pipe_path = 'pipe.pkl', plot = False, verbose = True):
    
    models = {
        'Linear Model': LinearRegression(), 
        'Catboost': CatBoostRegressor(
            iterations = 500,
            loss_function='RMSE',
            random_state = 1,
            learning_rate=0.01,
            depth=4,
            l2_leaf_reg=0.1,
        
            ),
        'Random Forrest': RandomForestRegressor(n_estimators= 1000, max_depth = 8, random_state = 1),
        
    }
    
    kwargs = {i:{} for i in models}
    kwargs['Catboost'] = {'verbose': 0}
    
    models_fit = {}
    
    metrics = {}
    
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.25)

    try:
    
        with open(pipe_path, 'rb') as f:
            
            pipe, pipe_target = pickle.load(f)

    except:

        raise ValueError('Pipeline file for preprocessing missing')
        
    y_test = pipe_target.inverse_transform(y_test)
    
    for i in tqdm(models):
        
        print(f'>>>> {i}')
        
        models_fit[i] = models[i].fit(x_train, y_train, **kwargs[i])
        
        predictions = make_predictions_sigle(
            model = models_fit[i], features = x_test, transform_back_target = True, pipe_path = pipe_path)
        
        metrics[i] = eval_predictions(y_test, predictions, plot = plot, verbose = verbose)
    
        
    with open(models_path, 'wb') as f:
        
        pickle.dump(obj = [models_fit, metrics], file = f)
        
    return [models_fit, metrics]

# Make predictions for single model
def make_predictions_sigle(features, model, pipe_path = 'pipe.pkl', transform_back_target = True):
    
    forecast = model.predict(features)
    
    with open(pipe_path, 'rb') as f:
        
        pipe, pipe_target = pickle.load(f)
        
    if transform_back_target:
        
        forecast = pipe_target.inverse_transform(forecast)
    
    return forecast

def eval_predictions(target, predictions, verbose = True, plot = True):
    
    out = {}
    
    out['MSE'] = mean_squared_error(target, predictions)
    
    out['RMSE'] = out['MSE'] ** 0.5
    
    out['MAE'] = mean_squared_error(target, predictions)
    
    out['Max Error'] = max_error(target, predictions)
    
    out['Median Absolute Error'] = median_absolute_error(target, predictions)
    
    if verbose:
        
        print(f'MSE: {round(out["MSE"],3)}')
        print(f'RMSE: {round(out["RMSE"],3)}')
        print(f'Max Error: {round(out["Max Error"],3)}')
        print(f'Median Absolute Error: {round(out["Max Error"],3)}')
        
    if plot:
        
        ax = sns.regplot(x = predictions.reshape(-1), y = target.reshape(-1), )
        
        ax.set(xlabel='Predictions', ylabel='Target Value')
        
        plt.show()
    
    
    return out

# Wrap train for simplification
def train_wraper(file_path = 'train.csv', return_arg = False, plot = False,
     verbose = True, pipe_path = 'pipe.pkl'):
    
    features, target = preProcess(path = file_path)
    
    models, metrics = create_models(features, target, plot = plot, verbose = verbose, pipe_path = pipe_path)
    
    if return_arg:
    
        return models, metrics



if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog = 'predictions', description='Make Predictions on the file')
    parser.add_argument('file_path', help = 'file for training')
    parser.add_argument('-plot', help = 'Plot predictions on train set', action='store_true')
    parser.add_argument('-v', dest='verbose', action='store_true', help = 'Show results while training')
    parser.add_argument('-p', dest = 'pipe_path', action = 'store_true', default = 'pipe.pkl', 
        help='File with pipeline transformations created in preprocess')
    args = parser.parse_args()

    train_wraper(**vars(args))