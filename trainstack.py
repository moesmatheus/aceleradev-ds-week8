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
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVR
from sklearn.preprocessing import RobustScaler
from tpot.builtins import StackingEstimator
from preprocess import preProcess

import warnings
warnings.filterwarnings("ignore")


def create_models_stacking(features, target, models_path = 'models-stack.pkl', pipe_path = 'pipe.pkl', 
                           plot = False, verbose = True, folds = 5, stack_train_split = 0.25):
    
    # Ridge
    kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

    e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
    e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
    alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
    alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
    
    models = {
        'tpot': make_pipeline(
                StackingEstimator(estimator=LinearSVR(C=1.0, dual=True, epsilon=0.01, loss="squared_epsilon_insensitive", tol=0.0001)),
                GradientBoostingRegressor(alpha=0.85, learning_rate=0.1, loss="ls", max_depth=2, max_features=0.7000000000000001, 
                                          min_samples_leaf=1, min_samples_split=17, n_estimators=100, subsample=0.35000000000000003)
            ),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4,
                                max_features='sqrt', min_samples_leaf=15, min_samples_split=10,
                                loss='huber', random_state =42),
        'Ridge': make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds)),
        'lasso': make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, 
                    alphas=alphas2,random_state=42, cv=kfolds)),
        'elasticnet': make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, 
                         alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio)),
        'svr': make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,)),
        'Linear Model': LinearRegression(), 
        'Catboost': CatBoostRegressor(
            iterations = 500,
            loss_function='RMSE',
            random_state = 1,
            learning_rate=0.01,
            depth=4,
            l2_leaf_reg=0.1,
        
            ),
        'Random Forrest': RandomForestRegressor(n_estimators= 300, max_depth = 6, random_state = 1),
        
    }
    
    model_stack = CatBoostRegressor(
            iterations = 500,
            loss_function='RMSE',
            random_state = 1,
            learning_rate=0.01,
            depth=4,
            l2_leaf_reg=0.1,
        
            )
    
    kwargs = {i:{} for i in models}
    kwargs['Catboost'] = {'verbose': 0}
    
    models_fit = {}
    
    metrics = {}
    
    outfold = {}
    
    kfold = KFold(n_splits = folds, shuffle=True, random_state=156)
    
    with open(pipe_path, 'rb') as f:
        
        pipe, pipe_target = pickle.load(f)
        
    y_orig = pipe_target.inverse_transform(target).reshape(-1)
    
    predictions = {}
    
    for i in tqdm(models):
        
        print(f'>>>> {i}')
        
        models_fit[i] = []
        
        outfold[i] = []
        
        predictions[i] = y_orig.copy()
        
        for train, test in kfold.split(features, target):
        
            models_fit[i].append(models[i].fit(features[train], target[train], **kwargs[i]))

            predictions[i][test] = make_predictions_sigle(
                model = models_fit[i][-1], features = features[test],
                transform_back_target = True, pipe_path = pipe_path
            ).reshape(-1)
        
        metrics[i] = eval_predictions(y_orig, predictions[i], plot = plot, verbose = verbose)
        
    if stack_train_split is not None:
        
        x_train, x_test, y_train, y_test = train_test_split(
            pd.DataFrame(predictions), y_orig, test_size = stack_train_split)
        
        #model_stack = model_stack.fit(x_train, y_train, verbose = 0, )
        model_stack = model_stack.fit(x_train, y_train, use_best_model=True, eval_set=(x_test, y_test), verbose=False)
        
        p = make_predictions_sigle(model = model_stack, features = x_test, transform_back_target = False)
        
        metrics['stack'] = eval_predictions(y_test, p, plot = plot, verbose = verbose)
        
    with open(models_path, 'wb') as f:
        
        pickle.dump(obj = [models_fit, metrics, model_stack], file = f)
        
    return [models_fit, metrics, model_stack]

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
    
    models, metrics, model_stack = create_models_stacking(features, target, plot = plot, verbose = verbose, pipe_path = pipe_path)
    
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