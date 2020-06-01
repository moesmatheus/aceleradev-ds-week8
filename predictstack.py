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


def make_predictions_stacking(features, models_path = 'models-stack.pkl'):
    
    with open(models_path, 'rb') as f:
        
        models_fit, metrics, model_stack = pickle.load(file = f)
        
    predictions_base = {}
    
    for i in models_fit:
        
        predictions_base[i] = np.mean(
            [make_predictions_sigle(
                model=j, transform_back_target=True, features = features
                ) for j in models_fit[i]]
            , axis = 0).reshape(-1)
        
    return model_stack.predict(pd.DataFrame(predictions_base))


def make_predictions_wraper_stacking(file_path = 'test.csv', df_ = None, return_predictions = False, save_name = None,
                        models_path = 'models-stack.pkl'):
    
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
    
    
    # Make predictions
    predictions['NU_NOTA_MT'] = make_predictions_stacking(features, models_path = models_path)
    
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

    make_predictions_wraper_stacking(**vars(args))