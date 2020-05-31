import numpy as np
import pandas as pd
import pickle
from preprocess import preProcess
import argparse

# Make predictions for single model
def make_predictions_sigle(features, model, pipe_path = 'pipe.pkl', transform_back_target = True):
    
    forecast = model.predict(features)
    
    with open(pipe_path, 'rb') as f:
        
        pipe, pipe_target = pickle.load(f)
        
    if transform_back_target:
        
        forecast = pipe_target.inverse_transform(forecast)
    
    return forecast


def predictions_ensemble(features, models, metrics = None, ensamble_method = None, pipe_path = 'pipe.pkl'):
    predictions = {}
    
    for i in models:
        
        predictions[i] = make_predictions_sigle(features, models[i], pipe_path = 'pipe.pkl', transform_back_target = True)
        
    if ensamble_method is not None:
        
        predictions = ensamble_method(predictions)
        
    return predictions


def ensamble_method(predictions):
    
    for i in predictions:
        
        if len(predictions[i].shape) > 1:
            
            predictions[i] = predictions[i].reshape(-1)
    
    return pd.DataFrame(predictions).mean(axis = 1)


def make_predictions_wraper(file_path = 'test.csv', df_ = None, return_predictions = False, save_name = None):
    
    # Select the ones who missed the test
    if df_ is None:
        df = pd.read_csv(file_path, index_col='NU_INSCRICAO')
    else:
        df = df_.copy()

    zeros = df.isnull().query('NU_NOTA_LC')[[]]
    zeros['NU_NOTA_MT'] = 0
    
    # Get ID of columns to be predicted
    predictions = df[['NU_NOTA_LC']].dropna()[[]]#.index
    
    # Load data
    if df_ is None:
        features, pipe = preProcess(path = file_path, train = False)
    else:
        features, pipe = preProcess(df_ = df_, train = False)
    
    # Load models
    try:

        with open('models.pkl', 'rb') as f:
            models, metrics = pickle.load(f)
    except:

        raise ValueError('Model File Missing')
    
    # Make predictions
    predictions['NU_NOTA_MT'] = predictions_ensemble(features, models, ensamble_method = ensamble_method).values
    
    predictions_full = pd.concat([zeros, predictions])
    
    if save_name is not None:
        
        predictions_full.to_csv(save_name)
        
    if return_predictions:
    
        return predictions_full

        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog = 'predictions', description='Make Predictions on the file')
    parser.add_argument('file_path', help = 'file to predict')
    parser.add_argument('-s', dest = 'save_name', help = 'Path to save file', action='store', default = None)
    parser.add_argument('-r', dest = 'return_predictions', action='store_true', help = 'Return Predictions')
    args = parser.parse_args()

    if args.save_name is not None:
        print('Predictions Saved to: ', args.save_name)

    make_predictions_wraper(**vars(args))