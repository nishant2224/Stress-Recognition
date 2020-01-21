from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV 
from tpot import TPOTClassifier
%matplotlib inline
import matplotlib.pyplot as plt
from scipy import signal 
import pickle


import sklearn.metrics
from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support 

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)

dataframe_hrv = pd.read_csv("Stress_classifier_with_AutoML_and_wearable_devices/dataset/dataframe_hrv.csv")
dataframe_hrv = dataframe_hrv.reset_index(drop=True)
display(dataframe_hrv.describe())
print(dataframe_hrv.columns)


def fix_stress_labels(df='',label_column='stress'):
    df['stress'] = np.where(df['stress']>=0.5, 1, 0)
    display(df["stress"].unique())
    return df
dataframe_hrv = fix_stress_labels(df=dataframe_hrv)

def missing_values(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df[~np.isfinite(df)] = np.nan
    df=df.fillna(df.mean())
    return df

selected_x_columns = ['HR','interval in seconds','AVNN', 'RMSSD', 'pNN50', 'TP', 'ULF', 'VLF', 'LF', 'HF','LF_HF']

X = dataframe_hrv[selected_x_columns]
y = dataframe_hrv['stress']

display(X.columns)
display(X.describe())
display(X.shape)

dataframe_hrv = missing_values(dataframe_hrv)

def do_tpot(generations=5, population_size=10,X='',y=''):

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.80, test_size=0.20)

    tpot = TPOTClassifier(generations=generations, population_size=population_size, verbosity=2,cv=3)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_pipeline.py')
    return tpot

tpot_classifer = do_tpot(generations=10, population_size=20,X=X,y=y)

def plotFitBitReading(dfnewHRV='', predictor = "none",selected_x_columns=''):
    dfnewHRV = missing_values(dfnewHRV)
    dfnewPol = dfnewHRV[selected_x_columns].fillna(0)

    print(dfnewPol.columns)
    print(dfnewPol.shape)
    pred = predictor.predict_proba(dfnewPol)
    
    dfpred = pd.DataFrame(pred)

    dfpred.columns = [["FALSE","TRUE"]]
    dfpred['stress'] = np.where(dfpred["TRUE"] > 0.5, 1, np.nan)

    
    dfnewHRV["stress"] = dfpred["stress"]
    dfnewHRV.loc[dfnewHRV["steps"] > 0, 'stress'] = np.nan
    #mark is to mark the RR peaks as stress
    dfnewHRV.loc[dfnewHRV["stress"] == 1, 'stress'] = dfnewHRV['interval in seconds'] 
    dfnewHRV.loc[dfnewHRV["steps"] > 0, 'moving'] = dfnewHRV['interval in seconds'] 
    dfnewHRV["minutes"] = (dfnewHRV['newtime']/60)/1000
    
    from itertools import cycle, islice
    my_colors = list(islice(cycle(['b', 'r', 'y', 'k']), None, len(dfnewHRV)))
    plot = dfnewHRV.plot(x="minutes", y=['interval in seconds',"stress", "moving"],color=my_colors)
    
    fig = plot.get_figure()

import glob

for filename in glob.iglob('dataset/**/*.csv', recursive=True):
    if 'dfnew' in filename:
        print(filename)
input_df = pd.read_csv('dataset/Vikings/Female_24_years_old/dfnewHRV.csv')
plotFitBitReading(input_df,tpot_classifer,selected_x_columns)