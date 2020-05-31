import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import predict
import preprocess
#import seaborn as sns
#plt.style.use('seaborn')

@st.cache
def load_data():

    df = pd.read_csv('train.csv', index_col='NU_INSCRICAO')

    # Columns to select
    cols_select = [
            'SG_UF_RESIDENCIA', 'NU_IDADE', 'TP_SEXO', 'TP_COR_RACA',
           'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TP_ENSINO',
           'TP_DEPENDENCIA_ADM_ESC', 'CO_PROVA_CH', #'CO_PROVA_LC', 
           'CO_PROVA_MT',
           'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'TP_LINGUA',
           'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3',
           'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO', 'Q001', 'Q002',
           'Q006', 'Q024', 'Q025', 'Q026', 'Q027', 'Q047', 'NU_NOTA_MT'
                  ]
    
    df = df[cols_select]

    cat_cols = [
        'SG_UF_RESIDENCIA', 'TP_SEXO', 'CO_PROVA_CH',#'CO_PROVA_LC',
        'CO_PROVA_MT', 'Q001', 'Q002', 'Q006', 'Q024', 'Q025',
       'Q026', 'Q027', 'Q047',
        
        'TP_COR_RACA', 'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO', 'TP_ENSINO', 'TP_DEPENDENCIA_ADM_ESC',
         'TP_STATUS_REDACAO',
         'TP_LINGUA'
    ]

    float_cols = [
        'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_COMP1','NU_NOTA_COMP2',
        'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5','NU_NOTA_REDACAO',
        'NU_IDADE', 'TP_ANO_CONCLUIU'
    ]  

    df[cat_cols] = df[cat_cols].astype('object')
    df[float_cols] = df[float_cols].astype('float64')

    return df

def main():

    st.title('ENEM Prediction Model')

    data = load_data()

    st.subheader('Data:')

    st.dataframe(data.head())

    target = data['NU_NOTA_MT'].dropna()



    # Sidebar
    st.sidebar.header('Select Variables')

    d = []

    for i in data.columns[:-1]:
        if data[i].dtype == 'O':
            unique = data[i].dropna().unique()
            d.append(
                st.sidebar.selectbox(
                    f'Select {i}',
                    unique.astype('object'), 
                    unique.tolist().index(data[i].mode()[0])
                    )
                )
        elif data[i].dtype == 'int64':
            d.append(
                st.sidebar.slider(
                    f'Select {i}', 
                    int(data[i].min()), 
                    int(data[i].max()), 
                    int(data[i].dropna().median())
                    )
                )
        elif data[i].dtype == 'float64':
            d.append(
                st.sidebar.slider(
                    f'Select {i}', 
                    float(data[i].min()), 
                    float(data[i].max()), 
                    float(data[i].dropna().median())
                    )
                )
        else:
            st.sidebar.text(i)

    df_input = pd.DataFrame(np.array(d).reshape(1,-1), columns = data.columns[:-1])

    cat_cols = [
        'SG_UF_RESIDENCIA', 'TP_SEXO', 'CO_PROVA_CH',#'CO_PROVA_LC',
        'CO_PROVA_MT', 'Q001', 'Q002', 'Q006', 'Q024', 'Q025',
       'Q026', 'Q027', 'Q047',
        
        'TP_COR_RACA', 'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO', 'TP_ENSINO', 'TP_DEPENDENCIA_ADM_ESC',
         'TP_STATUS_REDACAO'
    ]

    float_cols = [
        'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_COMP1','NU_NOTA_COMP2',
        'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5','NU_NOTA_REDACAO',
        'NU_IDADE', 'TP_ANO_CONCLUIU'
    ]  

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

    int_cols = ['TP_COR_RACA', 'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO', 'TP_LINGUA']
    
    df_input = df_input[cols_select]

    df_input[cat_cols] = df_input[cat_cols].astype('object')
    df_input[float_cols] = df_input[float_cols].astype('float64')
    df_input[int_cols] = df_input[int_cols].astype('int64')

    f = ['TP_ENSINO', 'TP_DEPENDENCIA_ADM_ESC', 'TP_STATUS_REDACAO']
    df_input[f] = df_input[f].astype('float64')

    #st.dataframe(df_input)

    # st.text(len(d))

    prediction = predict.make_predictions_wraper(df_ = df_input, return_predictions = True).values[0][0]
    # st.text(preprocess.preProcess(df_ = df_input, train = False)[0].shape)
    # st.text(prediction)
    # st.text(
    #     preprocess.preProcess(df_ = df_input, train = False)[0]
    # )

    # df_input.to_csv('st.csv')
    # df_input.to_pickle('st.pkl')

    #Plot data
    st.subheader('Prediction')
    plt.hist(target, bins = 70, alpha = 0.6)
    plt.axvline(prediction, color='k', linestyle='dashed', linewidth=1, label = 'Prediction', alpha = .5)
    min_ylim, max_ylim = plt.ylim()
    # plt.text(target.mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(target.mean()), alpha = .5)
    plt.legend()
    st.pyplot()

    
    

    


if __name__ == '__main__':

    main()