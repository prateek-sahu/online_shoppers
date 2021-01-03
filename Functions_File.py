import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


print("Functions_File is successfully imported.")



def total_missing_values(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = round(((df.isnull().sum()/df.isnull().count())*100).sort_values(ascending = False),2)
    missing_data = pd.concat([total , percent] , axis = 1 , keys = ['Total' , 'Percentage'])
    return missing_data

def column_type(df):
    numerical_feats = df.dtypes[df.dtypes != "object"].index
    print("Number of Numerical features: ", len(numerical_feats))

    categorical_feats = df.dtypes[df.dtypes == "object"].index
    print("Number of Categorical features: ", len(categorical_feats))
    
    return numerical_feats ,categorical_feats

def scale(df ,numerical_feats ):
    scaler_num = MinMaxScaler()
    df[numerical_feats] = scaler_num.fit_transform(df[numerical_feats])
       
    return df  

#Imputation Logic :: Categorical columns -> Most Frequent value , Numerical variables -> Median

def impute_nan_cat(df , categorical_feats ):
    for col in categorical_feats:
        df[col].fillna(df[col].value_counts().index[0], inplace=True)
    return df

def impute_nan_num(df , numerical_feats):
    for i in numerical_feats:
        df[i].fillna(df[i].median(), inplace=True)
    return df


def distribution_categorical_features(data, feature_list):
    for var in feature_list:
        print( '\033[1m' +  var + '\033[0m' +'\n')
        print(data[var].value_counts(normalize=True).mul(100).round(2).astype(str)+ '%' )
        print("============================================================================================")
        

def correlation_matrix(data , feature_list):
    sns.set(style="white")
    f, ax = plt.subplots(figsize=(16,8))
    sns.heatmap(data[feature_list].corr(),annot=True )
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()

def univariate_plots(data , feature_list):
    
    for var in feature_list:
        sns.set(style='whitegrid',  font_scale=1.1, rc={"figure.figsize": [12, 5]})
        sns.distplot(
            data[var], norm_hist=False,  bins=50, hist_kws={"alpha": 1}
        ).set(xlabel=var, ylabel='Count')
        plt.show();