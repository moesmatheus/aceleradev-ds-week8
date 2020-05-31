# ENEM Model

## Run ML model
### Pre process data
    python preprocess.py train.csv
### Train data
    python train.py -v train.csv
### Predict data
    python predict.py -s 'answer.csv' test.csv

## Run Streamlit
    streamlit run stream.py