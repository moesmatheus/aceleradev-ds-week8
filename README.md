# ENEM Model

## Run ML model
### Pre process data
    python preprocess.py train.csv
### Train data
    python train.py -v train.csv
    # or using a stacking model
    python trainstack.py -v train.csv
### Predict data
    python predict.py -s 'answer.csv' test.csv
    # or using a stacking model
    python predictstack.py -s 'answer.csv' test.csv

## Run Streamlit
    streamlit run stream.py

## The project
The current project is divided in some different parts
- **exploratory-analysis.ipynb:** a notebook with an exploratory analysis of the dataset
- **preprocess.py:** a script to preprocess some dataset in order to use in a Machine Learning Model
- **train.py:** a script to train the machine learning models ant evaluate their metrics
- **predict.py:** a script to make predictions with the fitted models
- **stream.py:** a script to simulate results using the created models in a web [platform](https://enem-codenation.herokuapp.com/) 