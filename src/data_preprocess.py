import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
'''  标准差归一化（Z-score）
numvars = numvars.apply(lambda x: (x - x.mean()) / x.std())
'''
def identify_columns(df, threshold=25):
    # Identify categorical object columns, categorical numerical columns, and non-categorical columns
    cat_object_list = [i for i in df.columns if df[i].dtype == 'object']
    cat_num_list = [i for i in df.columns if df[i].dtype in ['int64', 'float64'] and df[i].nunique() < threshold]
    catvars_list = cat_object_list + cat_num_list
    # cat_list = [i for i in df.columns if df[i].nunique() < threshold]
    non_cat_list = [i for i in df.columns if i not in cat_object_list and i not in cat_num_list]
    # Identify object columns and numerical columns in non-categorical columns
    mix_serise_col = df[non_cat_list]
    non_cat_obj = [i for i in mix_serise_col.columns if mix_serise_col[i].dtype == 'object']
    non_cat_num = [i for i in mix_serise_col.columns if mix_serise_col[i].dtype in ['int64', 'float64']]
    numvars_list = non_cat_num
    results = {
        'catvars_list': catvars_list,
        'numvars_list': numvars_list}

    return results
def label_encoding(df):
    le_vars = []
    for col in df.columns:
        if df[col].dtype == 'object':
            if len(df[col].unique()) == 2:
                le_vars.append(col)
                le = LabelEncoder()
                le.fit(df[col])
                df[col] = le.transform(df[col])
            # else:
            #     df[col] = pd.get_dummies(df[col])
    # print(df[le_vars])
    return df[le_vars]
def missing_values_table(df):
    # Check if input is a dataframe or a series
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    # Get columns with missing values
    na_columns = df.columns[df.isnull().any()].tolist()
    # Count missing values and calculate ratio
    n_miss = df[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (n_miss / df.shape[0] * 100).sort_values(ascending=False)
    # Create DataFrame with missing values and ratio
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['Missing Values', 'Percentage'])
    print(missing_df)
    return missing_df
def missing_preprocess_data(df):
    # Identify columns with more than 60% missing values
    missing_cols = df.columns[df.isnull().mean() > 0.6]
    # Drop columns with more than 60% missing values
    num_cols_dropped = len(missing_cols)
    df.drop(columns=missing_cols, inplace=True)
    print(f'Dropped {num_cols_dropped} columns due to missing value threshold')
    results = identify_columns(df)
    numvars_list = results['numvars_list']
    catvars_list = results['catvars_list']
    # Fill remaining missing values using median imputation
    # imputer = SimpleImputer(strategy='median')
    # df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    df[numvars_list] = df[numvars_list].fillna(df[numvars_list].mean())
    for col in catvars_list:
        df[col] = df[col].fillna(df[col].value_counts().index[0])
    return df

def Ant_data():
    data = pd.read_csv('./data/外流数据.csv')
    data.rename(columns={'label': 'target'}, inplace=True)
    print(data['target'].value_counts())
    all_features = list(data.columns)
    all_features.pop(0)
    all_features.pop(0)
    all_features.pop(0)
    all_features.pop(0)
    all_features.pop(0)
    all_features.pop(0)
    data_fea = data.loc[:,all_features]
    data_fea = missing_preprocess_data(data_fea)
    numeric_features = list(data_fea.select_dtypes(exclude=['object']).columns)
    data_fea.loc[:,numeric_features] = data_fea.loc[:,numeric_features].apply(lambda x: (x - x.min()) / (x.max()-x.min()))
    # data_fea.loc[:, numeric_features] = data_fea.loc[:, numeric_features].apply(lambda x: (x - x.mean()) / x.std())
    data_labels = data.loc[:,'target']
    X_train, X_test, Y_train, Y_test = train_test_split(data_fea, data_labels, test_size=0.2, random_state=11)
    # print(X_train.columns)
    return data_fea, data_labels,X_train, X_test, Y_train, Y_test
# Ant_data()

def Taiwan():
    data = pd.read_csv('./data/Taiwan.csv', header=1)
    data.rename(columns={'default payment next month': 'target'}, inplace=True)
    print(data['target'].value_counts())
    all_features = list(data.columns)
    all_features.pop(0)
    all_features.pop(-1)
    data_fea = data.loc[:, all_features]
    # data_fea = missing_preprocess_data(data_fea)
    results = identify_columns(data_fea)
    catvars = results['catvars_list']
    numvars = results['numvars_list']
    numeric_features = numvars
    data_fea.loc[:, numeric_features] = data_fea.loc[:, numeric_features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    # data_fea.loc[:, numeric_features] = data_fea.loc[:, numeric_features].apply(lambda x: (x - x.mean()) / x.std())
    data_fea = data_fea.astype('str')
    dummyvars = pd.get_dummies(data_fea[catvars])
    data_fea = pd.concat([data_fea[numeric_features], dummyvars], axis=1)
    data_fea = data_fea.astype('float')
    data_labels = data.loc[:, 'target']
    # 随机打乱列顺序
    # data_fea = data_fea.sample(frac=1, axis=1,random_state=111)
    X_train, X_test, Y_train, Y_test = train_test_split(data_fea, data_labels, test_size=0.2, random_state=11)
    # print(X_train.columns)
    return data_fea, data_labels, X_train, X_test, Y_train, Y_test
# Taiwan()
def Give_me_some_credit():
    data = pd.read_csv('./data/cs-training.csv')
    data.rename(columns={'SeriousDlqin2yrs': 'target'}, inplace=True)
    print(data['target'].value_counts())
    all_features = list(data.columns)
    all_features.pop(0)
    all_features.pop(0)
    data_fea = data.loc[:,all_features]
    data_fea = missing_preprocess_data(data_fea)
    numeric_features = list(data_fea.select_dtypes(exclude=['object']).columns)
    data_fea.loc[:,numeric_features] = data_fea.loc[:,numeric_features].apply(lambda x: (x - x.min()) / (x.max()-x.min()))
    # data_fea.loc[:, numeric_features] = data_fea.loc[:, numeric_features].apply(lambda x: (x - x.mean()) / x.std())
    data_labels = data.loc[:,'target']
    X_train, X_test, Y_train, Y_test = train_test_split(data_fea, data_labels, test_size=0.2, random_state=11)
    # print(X_train.columns)
    return data_fea, data_labels,X_train, X_test, Y_train, Y_test

def Loan_Data():
    data = pd.read_csv('./data/Loan Data.csv', delimiter=';')
    data.rename(columns={'BAD':'target'}, inplace=True)
    print(data['target'].value_counts())
    all_features = list(data.columns)
    all_features.pop(-1)
    data_fea = data.loc[:, all_features]
    catvars = ['AES','RES']
    numvars = [k for k in all_features if k not in catvars]
    numeric_features = numvars
    data_fea.loc[:,numeric_features] = data_fea.loc[:,numeric_features].apply(lambda x: (x - x.min()) / (x.max()-x.min()))
    # data_fea.loc[:, numeric_features] = data_fea.loc[:, numeric_features].apply(lambda x: (x - x.mean()) / x.std())
    dummyvars = pd.get_dummies(data_fea[catvars], dtype=float)
    data_fea = pd.concat([data_fea[numeric_features], dummyvars], axis=1)
    data_labels = data.loc[:,'target'].astype('int')
    X_train, X_test, Y_train, Y_test = train_test_split(data_fea, data_labels, test_size=0.2, random_state=11)
    # print(X_train.columns)
    return data_fea, data_labels,X_train, X_test, Y_train, Y_test
# Loan_Data()

def German():
    names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount',
             'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors',
             'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing',
             'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'target']
    data = pd.read_csv('./data/german.csv', names=names, delimiter=',')
    data.target.replace([1, 2], [0, 1], inplace=True)
    print(data['target'].value_counts())
    all_features = list(data.columns)
    all_features.pop(-1)
    data_fea = data.loc[:,all_features]
    # results = identify_columns(data_fea)
    # catvars = results['catvars_list']
    # numvars = results['numvars_list']
    numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'age',
               'existingcredits', 'peopleliable']
    catvars = ['existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',
               'statussex', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job',
               'telephone', 'foreignworker']
    numeric_features = numvars
    data_fea.loc[:,numeric_features] = data_fea.loc[:,numeric_features].apply(lambda x: (x - x.min()) / (x.max()-x.min()))
    # train_fea[numeric_features] = train_fea[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    dummyvars = pd.get_dummies(data_fea[catvars])
    data_fea = pd.concat([data_fea[numeric_features], dummyvars], axis=1)
    data_labels = data.loc[:, 'target']
    X_train, X_test, Y_train, Y_test = train_test_split(data_fea, data_labels, test_size=0.2, random_state=11)
    # print(X_train.columns)
    return data_fea, data_labels,X_train, X_test, Y_train, Y_test
# German()

def HMEQ():
    data = pd.read_csv('./data/hmeq.csv', delimiter=',')
    data.rename(columns={'BAD': 'target'}, inplace=True)
    print(data['target'].value_counts())
    all_features = list(data.columns)
    all_features.pop(0)
    data_fea = data.loc[:,all_features]
    data_fea = missing_preprocess_data(data_fea)
    levars = label_encoding(data_fea)
    results = identify_columns(data_fea)
    catvars = results['catvars_list']
    numvars = results['numvars_list']
    numeric_features = numvars
    data_fea.loc[:, numeric_features] = data_fea.loc[:, numeric_features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    # data_fea.loc[:, numeric_features] = data_fea.loc[:, numeric_features].apply(lambda x: (x - x.mean()) / x.std())
    dummyvars = pd.get_dummies(data_fea[catvars], columns=catvars, dtype=float)
    data_fea = pd.concat([data_fea[numeric_features], levars, dummyvars], axis=1)
    data_labels = data.loc[:, 'target']
    X_train, X_test, Y_train, Y_test = train_test_split(data_fea, data_labels, test_size=0.2, random_state=11)
    # print(X_train.columns)
    return data_fea, data_labels,X_train, X_test, Y_train, Y_test
# HMEQ()
def PAKDD():
    data = pd.read_excel('./data/process_PAKDD2.xlsx')
    # data = pd.read_excel('E:\Software\PycharmProjects\pythonProject\IVLR-ACS\data\process_PAKDD2.xlsx')
    data.rename(columns={'TARGET': 'target'}, inplace=True)
    print(data['target'].value_counts())
    all_features = list(data.columns)
    all_features.pop(-1)
    data_fea = data.loc[:, all_features]
    numvars = all_features[:14]
    catvars = all_features[14:]
    numeric_features = numvars
    data_fea.loc[:, numeric_features] = data_fea.loc[:, numeric_features].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()))
    data_fea = pd.concat([data_fea[numeric_features], data_fea[catvars]], axis=1)
    data_labels = data.loc[:, 'target']
    X_train, X_test, Y_train, Y_test = train_test_split(data_fea, data_labels, test_size=0.2, random_state=11)
    # print(X_train.columns)
    return data_fea, data_labels,X_train, X_test, Y_train, Y_test
# PAKDD()
def HomeCredit():
    data = pd.read_csv('./data/application_train.csv', delimiter=',')
    # data = pd.read_csv('E:\Software\PycharmProjects\pythonProject\IVLR-ACS\data\\application_train.csv',delimiter=',')
    data.rename(columns={'TARGET': 'target'}, inplace=True)
    print(data['target'].value_counts())
    all_features = list(data.columns)
    all_features.pop(0)
    all_features.pop(0)
    data_fea = data.loc[:, all_features]
    data_fea = missing_preprocess_data(data_fea)
    levars = label_encoding(data_fea)
    results = identify_columns(data_fea)
    catvars_list = results['catvars_list']
    numvars_list = results['numvars_list']
    dummyvars = pd.get_dummies(data[catvars_list], columns=catvars_list)
    numvars = data[numvars_list]
    numvars = numvars.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    data_fea = pd.concat([numvars, levars, dummyvars], axis=1)
    data_fea = missing_preprocess_data(data_fea)
    data_labels = data.loc[:, 'target']
    X_train, X_test, Y_train, Y_test = train_test_split(data_fea, data_labels, test_size=0.2, random_state=11)
    # print(X_train.columns)
    return data_fea, data_labels,X_train, X_test, Y_train, Y_test
# HomeCredit()

def Lendingclub():
    data = pd.read_csv('./data/lending club2005_2012.csv', delimiter=',')
    # data = pd.read_csv('E:\Software\PycharmProjects\pythonProject\IVLR-ACS\data\lending club2005_2012.csv', delimiter=',')
    data.rename(columns={'lable': 'target'}, inplace=True)
    print(data['target'].value_counts())
    all_features = list(data.columns)
    all_features.pop(-1)
    data_fea = data.loc[:, all_features]
    numvars = all_features[:30]
    catvars = all_features[30:]
    numeric_features = numvars
    data_fea.loc[:, numeric_features] = data_fea.loc[:, numeric_features].apply(lambda x: (x - x.min()) / (x.max()-x.min()))
    data_fea = pd.concat([data_fea[numeric_features], data_fea[catvars]], axis=1)
    data_fea = missing_preprocess_data(data_fea)
    data_labels = data.loc[:, 'target']
    X_train, X_test, Y_train, Y_test = train_test_split(data_fea, data_labels, test_size=0.2, random_state=11)
    # print(X_train.columns)
    return data_fea, data_labels,X_train, X_test, Y_train, Y_test