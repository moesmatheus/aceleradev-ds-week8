import pandas as pd
import argparse
import numpy as np
#import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
import pickle
from category_encoders.cat_boost import CatBoostEncoder
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, median_absolute_error


def preProcess(path = 'train.csv', df_ = None, train = True, save = False,
               save_path = None, pipe_path = 'pipe.pkl'):
    
    # Read Data
    if df_ is None:
        df = pd.read_csv(path, index_col=0)
        if path is None:
            raise ValueError('Must Define path or DataFrame')
    else:
        df = df_.copy()
         
    # Drop when NU_NOTA_LC is null if train
    df.dropna(subset = ['NU_NOTA_LC'], inplace = True)
    
    if train:
        df.dropna(subset = ['NU_NOTA_CN', 'NU_NOTA_CH'], inplace = True)
        
        
    # Create target data if train
    if train:
        try:
            target = df['NU_NOTA_MT']
        except:         
            raise ValueError('Column NU_NOTA_MT missing from data')    
    else:  
        target = None
            
    
    # Columns to select
    cols_select = [
            'SG_UF_RESIDENCIA', 'NU_IDADE', 'TP_SEXO', 'TP_COR_RACA',
           'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TP_ENSINO',
           'TP_DEPENDENCIA_ADM_ESC', 'CO_PROVA_CH', #'CO_PROVA_LC', 
           'CO_PROVA_MT',
           'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'TP_LINGUA',
           'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3',
           'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO', 'Q001', 'Q002',
           'Q006', 'Q024', 'Q025', 'Q026', 'Q027', 'Q047'
                  ]
    
    # Select Columsn
    try:      
        df = df[cols_select]
    except:    
        raise ValueError('Column missing from data')
        
     
    # Columns that have floats but are categorical
    # cols_cat = [
    #     'TP_COR_RACA', 'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO', 'TP_ENSINO', 'TP_DEPENDENCIA_ADM_ESC',
    #     'CO_PROVA_CH', 'CO_PROVA_MT',#'CO_PROVA_LC',
    #      'TP_STATUS_REDACAO'
    #            ]
    
    # Convert columns
    
    
        
    float_cols = [
        'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_COMP1','NU_NOTA_COMP2',
        'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5','NU_NOTA_REDACAO',
        'NU_IDADE', 'TP_ANO_CONCLUIU'
    ]   

    df[float_cols] = df[float_cols].astype('float64')
        
    # Create Pipeline floats
    pipe_float = Pipeline([
        ('inputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler()),
    ])
    
    # Create Pipeline Categorical Features
    cat_cols = [
        'SG_UF_RESIDENCIA', 'TP_SEXO', 'CO_PROVA_CH',#'CO_PROVA_LC',
        'CO_PROVA_MT', 'Q001', 'Q002', 'Q006', 'Q024', 'Q025',
       'Q026', 'Q027', 'Q047',
        
        'TP_COR_RACA', 'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO', 'TP_ENSINO', 'TP_DEPENDENCIA_ADM_ESC',
         'TP_STATUS_REDACAO',
         'TP_LINGUA'
    ]

    df[cat_cols] = df[cat_cols].astype('object')
    
    pipe_cat = Pipeline([
        ('label encoder', CatBoostEncoder())
    ])
    
    # Create full pipeline
    pipe = ColumnTransformer(transformers=[
        ('pipe_float', pipe_float, float_cols),
        ('pipe_cat', pipe_cat, cat_cols)
    ]#, remainder = 'passthrough'
    )
    
    pipe_target = Pipeline(
        [('scaler', StandardScaler())]
    )
    
    # print(df.head())
    
    # Fit pipelines
    if train:
        
        pipe.fit(df, target)
        
        pipe_target.fit(target.values.reshape(-1,1))
        
        with open(pipe_path, 'wb') as f:
            
            pickle.dump([pipe, pipe_target], f)
            
    else:
        
        with open(pipe_path, 'rb') as f:
            
            pipe, pipe_target = pickle.load(f)
    
    
    # Transform variables
    df = pipe.transform(df)
    
    if train:
    
        target = pipe_target.transform(target.values.reshape(-1,1))
     
    # Save file to pickle
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump([df, target], f)
        
    return [df, target]


if __name__=='__main__':

    parser = argparse.ArgumentParser(prog = 'preproces', description='Pre process dataset')
    parser.add_argument('path', help = 'file to preprocess', default = 'train.csv')
    parser.add_argument('-s', dest = 'save_path', help = 'Path to save file', action='store', default = None)
    parser.add_argument('-test', dest= 'train', action='store_false')
    args = parser.parse_args()
    preProcess(**vars(args))

