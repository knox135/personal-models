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




# run processed data through various models to determine which is best
# 'format is: model = models.many_models(X_train, y_train, X_test, y_test)'

# run processed data through various models to determine which is best
def many_models(x,y,xt,yt):
    
    # scale and fit data
    scaled = StandardScaler()
    x_scale = scaled.fit_transform(x)
    xt_scale = scaled.transform(xt)
    
    # Random Forest Classifier model
    rf = RandomForestClassifier()
    rf.fit(x,y)
    rf_predict = rf.predict(x)
    rf_test = rf.predict(xt)
    
    # print scores and classification reports
    print(f'\nRandom Forest \nTest Accuracy: {rf.score(xt,yt)}\nbalanced test score: {balanced_accuracy_score(yt,rf_test)}')
    print(f'classification report: \n {classification_report(yt,rf_test)}')
    
    # Gradient Boosting Classifier model
    gr = GradientBoostingClassifier()
    gr.fit(x,y)
    gr_predict = gr.predict(x)
    gr_test = gr.predict(xt)
    
    # print scores and classification reports
    print(f'\nGradient Boost \nTest Accuracy: {gr.score(xt,yt)}\nbalanced test score: {balanced_accuracy_score(yt,gr_test)}')
    print(f'\n classification report: \n {classification_report(yt,gr_test)}')
    
    # Logistic Regression model
    lr = LogisticRegression(max_iter=240)
    lr.fit(x_scale,y)
    lr_predict = lr.predict(x_scale)
    lr_test = lr.predict(xt_scale)
    
    # print scores and classification reports    
    print(f'\nLogistic Regression \nTest Accuracy: {lr.score(xt_scale,yt)}\nbalanced test score: {balanced_accuracy_score(yt,lr_test)}')
    print(f'classification report: \n {classification_report(yt,lr_test)}')
    
    # Poly Support Vector Classifier model
    svc = SVC(kernel='poly')
    svc.fit(x_scale,y)
    svc_predict = svc.predict(x_scale)
    svc_test = svc.predict(xt_scale)
    # print scores and classification reports
    
    print(f'\nPoly Support Vector \nTest Accuracy: {svc.score(xt_scale,yt)}\nBalanced test score: {balanced_accuracy_score(yt,svc_test)}')
    print(f'classification report: \n {classification_report(yt,svc_test)}')

    
    # ADA Boost model(slows the function down quite a lot, comment out or delete if need to speed up for rest)
    ada_low = AdaBoostClassifier(n_estimators=20)
    ada_low.fit(x_scale,y)
    ada_low_predict = ada_low.predict(x_scale)
    ada_low_test = ada_low.predict(xt_scale)
    
    # print scores and classification reports
    print(f'\nADA low estimators \nTest Accuracy: {ada_low.score(xt_scale,yt)}\nbalanced test score: {balanced_accuracy_score(yt,ada_low_test)}')
    print(f'classification report: \n {classification_report(yt,ada_low_test)}')
    
    ada = AdaBoostClassifier(n_estimators=1500)    
    ada.fit(x_scale,y)
    ada_predict = ada.predict(x_scale)
    ada_test = ada.predict(xt_scale)
    
    # print scores and classification reports
    print(f'\nADA \nTest Accuracy: {ada.score(xt_scale,yt)}\nbalanced test score: {balanced_accuracy_score(yt,ada_test)}')
    print(f'classification report: \n {classification_report(yt,ada_test)}')
    
    # Linear Support Vector Classifier model
    svc_sigmoid = SVC(kernel='sigmoid')
    svc_sigmoid.fit(x_scale,y)
    svc_sigmoid_predict = svc_sigmoid.predict(x_scale)
    svc_sigmoid_test = svc_sigmoid.predict(xt_scale)
    
    # print scores and classification reports
    print(f'\nSVC Sigmoid \nTest Accuracy: {svc_sigmoid.score(xt_scale,yt)}\nbalanced test score: {balanced_accuracy_score(yt,svc_sigmoid_test)}')
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
    comparison = comparison[['Trained Score','Test Score','Balanced Test Score','Balanced Difference']]
    print(comparison.head(7))
    
    # plot the comparison dataframe on a bar graph for visualizations
    comparison.plot(kind='bar').tick_params(axis='x',rotation=45)
    
    
    
    
    
    # Create checks for Random Forest Best Parameters
    
    
    
    
    
    # check best depth for random forest
    models = {'train_score': [], 'test_score': [], 'max_depth': []}
    
    # loop through range to see where deviation happens
    for depth in range(1,20):
        
        # save max depth to models
        models['max_depth'].append(depth)
        
        # create model for max depth 
        model = RandomForestClassifier(max_depth=depth)
        
        # fit the random forest model
        model.fit(x, y)
        
        # make max depth predictions
        y_test_pred = model.predict(xt)
        y_train_pred = model.predict(x)
        
        # save max depth scores to models
        models['train_score'].append(balanced_accuracy_score(y, y_train_pred))
        models['test_score'].append(balanced_accuracy_score(yt, y_test_pred))
        
    # save models data to dataframe and sort by test scores    
    models_df = pd.DataFrame(models)#.sort_values(by='test_score',ascending=False)#.reset_index('max_depth')
    
    # show graph of for loop
    models_df.plot(title='Random forest max depth', x='max_depth')
    
    #set max_depth as index
    models_df = models_df.set_index('max_depth').sort_values(by='test_score',ascending=False)
    print(f'\nRandom Forest Parameter Tuning:\n \nbest depth: \n{models_df}')
    
    # save top 5 to a list
    best_rf = models_df.head().index.tolist()
    print(f'max depths: \n{best_rf}\n')
    
    
    
    
    
    
    # test for leafs
    leaf = range(2,12)
    leaves = {'train_score':[],'test_score':[],'min_leaf': []}
    
    # loop through leaf values
    for i in leaf:
        
        # create Random Forest Classifier instance with depth
        model = RandomForestClassifier(min_samples_split=i)
        
        # fit model
        model.fit(x,y)
        
        # predict
        X_test_pred = model.predict(xt)
        y_train_pred = model.predict(x)
        
        #store scores and best amounts in leaves dictionary
        leaves['train_score'].append(balanced_accuracy_score(y,y_train_pred))
        leaves['test_score'].append(balanced_accuracy_score(yt,X_test_pred))
        leaves['min_leaf'].append(i)
        
    # store dictionary in a dataframe
    leaves_df = pd.DataFrame(leaves).sort_values(by='test_score',ascending=False)
    
    # set leaves_df index to 'min_leaf'
    leaves_df = leaves_df.set_index('min_leaf')
    print(f'best leaf: \n{leaves_df.head()}')
    
    # plot best leaf on graph
    leaves_df.plot(title='Random Forest Best Leaf')
    
    # save to list
    best_split = leaves_df.head().index.tolist()
    print(f'best leaf \n{best_split}\n')
    
    
    
    
    
    
    
    # test for best n_estimators
    # create n_estimators range search
    est = range(25,250,2)
    
    # create dictionary to store results
    est_models = {'train_score':[],'test_score':[],'n_estimators':[]}
    
    # loop through range to find best n_estimators
    for i in est:
        
        # create random forest model
        est_model = RandomForestClassifier(n_estimators=i)
        
        # fit model 
        est_model.fit(x,y)
        
        # make predictions for n_estimators
        X_test_pred = est_model.predict(xt)
        y_train_pred = est_model.predict(x)
        
        # store results in dictionairy
        est_models['train_score'].append(balanced_accuracy_score(y,y_train_pred))
        est_models['test_score'].append(balanced_accuracy_score(yt,X_test_pred))
        est_models['n_estimators'].append(i)
        
    # save to dataframe
    estimators_df = pd.DataFrame(est_models).sort_values(by='test_score',ascending=False)
    
    # set index to n_estimators
    estimators_df = estimators_df.set_index('n_estimators')
    print(f'\nbest n_estimators: \n{estimators_df.head()}')
    
    # plot n_estimators on graph
    estimators_df.plot(title='best n_estimators')
    
    # save to list
    best_est = estimators_df.head().index.tolist()
    print(f'best n_estimators \n{best_est}\n')
    
    
    
    
    
    
    
    
    # save best parameters to dictionary to use to find best combination
    best_params = {
        'n_estimators': best_est,
        'max_depth': best_rf,
        'min_samples_leaf': best_split
    }
    
    # create dictionary to store results
    best_results = {'n_estimators':[],'max_depth':[],'min_samples_leaf':[],'train_score':[],'test_score':[]}
    
    # loop through combinations(refactor in future for efficiency)
    for e in best_params['n_estimators']:
        for m in best_params['max_depth']:
            for l in best_params['min_samples_leaf']:
                
                # create random forest model with best parameters
                rf_model = RandomForestClassifier(n_estimators=e,max_depth=m,min_samples_leaf=l)
                
                # fit data
                rf_model.fit(x,y)
                
                # predictions
                test_pred = rf_model.predict(xt)
                train_pred = rf_model.predict(x)
                
                # append balanced scores
                best_results['train_score'].append(balanced_accuracy_score(y,train_pred))
                best_results['test_score'].append(balanced_accuracy_score(yt,test_pred))
                best_results['n_estimators'].append(e)
                best_results['max_depth'].append(m)
                best_results['min_samples_leaf'].append(l)
                    
    # save best results to dataframe
    best_results_df = pd.DataFrame(best_results).sort_values(by='test_score',ascending=False)
    best_results_df = best_results_df.set_index('test_score')
    print(f'\nbest results:\n{best_results_df.head()}')
    
    # save best parameters to print
    best_n = best_results_df['n_estimators'].iloc[0]
    best_d = best_results_df['max_depth'].iloc[0]
    best_l = best_results_df['min_samples_leaf'].iloc[0]
    print(f'best n_estimators: {best_n} \nbest max_depth: {best_d} \nbest leaves: {best_l}')
    
    # create random forest instance with best parameters
    tuned_rf = RandomForestClassifier(n_estimators=best_results_df['n_estimators'].iloc[0], max_depth=best_results_df['max_depth'].iloc[0], min_samples_leaf=best_results_df['min_samples_leaf'].iloc[0],random_state=13)    
    
    # train the tuned model
    tuned_rf.fit(x,y)
    
    # predict with the tuned model
    best_train = tuned_rf.predict(x)
    best_test = tuned_rf.predict(xt)
    
    # view tuned Random Forest Model scores and classification report
    print(f'best Random Forest Tuned Parameters scores \n')
    print(f'\nBest Random Forest Tuned Parameters Scores \nTest Accuracy: {tuned_rf.score(xt,yt)}\nbalanced test score: {balanced_accuracy_score(yt,best_test)}')
    print(f'classification report: \n {classification_report(yt,best_test)}')
    
    # add tuned Random Forest Model to the comparison DataFrame and Plot
    best_df = pd.DataFrame(
        [
            ['Tuned Random Forest',tuned_rf.score(x,y),tuned_rf.score(xt,yt),balanced_accuracy_score(y,best_train),balanced_accuracy_score(yt,best_test)]
        ],
            columns=['Model Name','Trained Score', 'Test Score','Balanced Trained Score','Balanced Test Score']
    ).set_index('Model Name')
    best_df['Balanced Difference'] = (best_df['Balanced Trained Score'] - best_df['Balanced Test Score'])
    best_df = best_df[['Trained Score','Test Score','Balanced Test Score','Balanced Difference']]
    comparison_df = pd.concat([comparison,best_df]).sort_values(by='Balanced Test Score',ascending=False)
    print(f'updated comparison DataFrame \n{comparison_df.head(8)}')
    
    # plot updated comparison model against initial models
    comparison_df.plot(kind='bar').tick_params(axis='x',rotation=45)
    
    
    return comparison,models_df,leaves_df,estimators_df,best_results_df,comparison_df
