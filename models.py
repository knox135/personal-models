# import dependencies

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report

# split data into training and testing data
def split(X,y):
    return train_test_split(X,y)

# fill na's with unknown
def fill_unknown(x):
    x = x.fillna('unknown')
    return x
def fill_education(X):
    X['education'] = X['education'].fillna('primary')
    return X
def fill_job(X):
    X['job'] = X['job'].fillna('unknown')
    return X
# fill na's with primary
def fill_primary(x):
    x = x.fillna('primary')
    return x

# fill missing data
def fill_missing(x):
    x = fill_education(x)
    x = fill_job(x)
    return x

# change data with label encoder
def label(x, xt):
    le = LabelEncoder()
    x = le.fit_transform(x)
    xt = le.transform(xt)
    print(f'train:{x}\ntest:{xt}')
    return x, xt

# Create a OneHotEncoder
def onehot(y):
    encode_y = OneHotEncoder(drop='first', max_categories=5, handle_unknown='infrequent_if_exist', sparse_output=False)
    encode_y.fit(y)
    return encode_y

# binary OrdinalEncoder
def binary(x):
    oe = OrdinalEncoder(categories=[['no', 'yes']], handle_unknown='use_encoded_value', unknown_value=-1)
    oe.fit(x.values.reshape(-1,1))
    return oe

# scale data with standardscaler
def scaled(x,xt):
    scaled = StandardScaler()
    X_train_scaled = scaled.fit_transform(x)
    X_test = scaled.fit(xt)
    X_train_scaled, X_test
    return X_train_scaled,X_test

# run processed data through various models to determine which is best
def many_models(x,y,xt,yt):
    
    # scale and fit data
    scaled = StandardScaler()
    x_scale = scaled.fit_transform(x)
    xt_scale = scaled.transform(xt)
    
    # Random Forest Classifier model
    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(x,y)
    rf_predict = rf.predict(x)
    rf_test = rf.predict(xt)
    print(f'Random Forest \nbalanced test score: {balanced_accuracy_score(yt,rf_test)}')
    print(f'classification report: \n {classification_report(yt,rf_test)}')
    
    # Gradient Boosting Classifier model
    gr = GradientBoostingClassifier()
    gr.fit(x,y)
    gr_predict = gr.predict(x)
    gr_test = gr.predict(xt)
    print(f'Gradient Boost \nbalanced test score: {balanced_accuracy_score(yt,gr_test)}')
    print(f'\n classification report: \n {classification_report(yt,gr_test)}')
    
    # Logistic Regression model
    lr = LogisticRegression(max_iter=120)
    lr.fit(x_scale,y)
    lr_predict = lr.predict(x_scale)
    lr_test = lr.predict(xt_scale)
    print(f'Logistic Regression \nbalanced test score: {balanced_accuracy_score(yt,lr_test)}')
    print(f'classification report: \n {classification_report(yt,lr_test)}')
    
    # Poly Support Vector Classifier model
    svc = SVC(kernel='poly')
    svc.fit(x_scale,y)
    svc_predict = svc.predict(x_scale)
    svc_test = svc.predict(xt_scale)
    print(f'Poly Support Vector \nbalanced test score: {balanced_accuracy_score(yt,svc_test)}')
    print(f'classification report: \n {classification_report(yt,svc_test)}')

    
    # ADA Boost model(slows the function down quite a lot, comment out or delete if need to speed up for rest)
    ada_low = AdaBoostClassifier(n_estimators=20)
    ada_low.fit(x_scale,y)
    ada_low_predict = ada_low.predict(x_scale)
    ada_low_test = ada_low.predict(xt_scale)
    print(f'ADA low estimators \nbalanced test score: {balanced_accuracy_score(yt,ada_low_test)}')
    print(f'classification report: \n {classification_report(yt,ada_low_test)}')
    
    ada = AdaBoostClassifier(n_estimators=2000)    
    ada.fit(x_scale,y)
    ada_predict = ada.predict(x_scale)
    ada_test = ada.predict(xt_scale)
    print(f'ADA \nbalanced test score: {balanced_accuracy_score(yt,ada_test)}')
    print(f'classification report: \n {classification_report(yt,ada_test)}')
    
    # Linear Support Vector Classifier model
    svc_sigmoid = SVC(kernel='sigmoid')
    svc_sigmoid.fit(x_scale,y)
    svc_sigmoid_predict = svc_sigmoid.predict(x_scale)
    svc_sigmoid_test = svc_sigmoid.predict(xt_scale)
    
    print(f'SVC Sigmoid \nbalanced test score: {balanced_accuracy_score(yt,svc_sigmoid_test)}')
    print(f'classification report: \n {classification_report(yt,svc_sigmoid_test)}')
    
    # comparison dataframe of the models
    comparison = pd.DataFrame(
        [
            ['Logistic Regression', lr.score(x_scale,y),lr.score(xt_scale,yt),balanced_accuracy_score(y,lr_predict),balanced_accuracy_score(yt,lr_test)],    
            ['SVC poly',svc.score(x_scale,y), svc.score(xt_scale,yt),balanced_accuracy_score(y,svc_predict),balanced_accuracy_score(yt,svc_test)],
            ['SVC sigmoid',svc_sigmoid.score(x_scale,y),svc_sigmoid.score(xt_scale,yt),balanced_accuracy_score(y,svc_sigmoid_predict),balanced_accuracy_score(yt,svc_sigmoid_test)],
            ['Random Forest',rf.score(x,y),rf.score(xt,yt),balanced_accuracy_score(y,rf_predict),balanced_accuracy_score(yt,rf_test)],
            ['Gradient Boosting',gr.score(x,y),gr.score(xt,yt),balanced_accuracy_score(y,gr_predict),balanced_accuracy_score(yt,gr_test)],
            ['ADA Low Estimators',ada_low.score(x_scale,y),ada_low.score(xt_scale,yt),balanced_accuracy_score(y,ada_low_predict),balanced_accuracy_score(yt,ada_low_test)],
            ['ADA boost', ada.score(x_scale,y),ada.score(xt_scale,yt),balanced_accuracy_score(y,ada_predict),balanced_accuracy_score(yt,ada_test)],
        ],
        columns=['Model Name','Trained Score', 'Test Score','Balanced Trained Score','Balanced Test Score']
    ).set_index('Model Name')
    
    # make new column calculating difference between training and test scores
    comparison['Balanced Difference'] = (comparison['Balanced Trained Score'] - comparison['Balanced Test Score'])
    
    # sort by difference column
    comparison = comparison.sort_values(by='Balanced Test Score', ascending=False)
    #comparison = comparison[['Trained Score','Test Score','Difference','Balanced Test Score']]
    comparison # type: ignore
    
    # plot the comparison dataframe on a bar graph for visualizations
    comparison.plot(kind='bar').tick_params(axis='x',labelrotation=45)
    
    # check best depth for random forest
    models = {'train_score': [], 'test_score': [], 'max_depth': []}
    
    # loop through range to see where deviation happens
    for depth in range(1,15):
        models['max_depth'].append(depth)
        model = RandomForestClassifier(n_estimators=500, max_depth=depth)
        model.fit(x, y)
        y_test_pred = model.predict(xt)
        y_train_pred = model.predict(x)
        models['train_score'].append(balanced_accuracy_score(y, y_train_pred))
        models['test_score'].append(balanced_accuracy_score(yt, y_test_pred))
    models_df = pd.DataFrame(models)
    
    # show graph of for loop
    models_df.plot(title='Random forest max depth', x='max_depth')
    return comparison,models_df



def build_default_encoder(X_filled):
    default_encoder = OrdinalEncoder(categories=[['no', 'yes']], handle_unknown='use_encoded_value', unknown_value=-1)

    # Train the encoder
    default_encoder.fit(X_filled['default'].values.reshape(-1, 1))
    return {'column': 'default',
            'multi_col_output': False,
            'encoder': default_encoder}

def build_housing_encoder(X_filled):
    housing_encoder = OrdinalEncoder(categories=[['no', 'yes']], handle_unknown='use_encoded_value', unknown_value=-1)

    # Train the encoder
    housing_encoder.fit(X_filled['housing'].values.reshape(-1, 1))
    return {'column': 'housing',
            'multi_col_output': False,
            'encoder': housing_encoder}

def build_loan_encoder(X_filled):
    loan_encoder = OrdinalEncoder(categories=[['no', 'yes']], handle_unknown='use_encoded_value', unknown_value=-1)

    # Train the encoder
    loan_encoder.fit(X_filled['loan'].values.reshape(-1, 1))
    return {'column': 'loan',
            'multi_col_output': False,
            'encoder': loan_encoder}
