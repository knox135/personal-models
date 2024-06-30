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




""" Format is: 'model = models.many_models(X_train, y_train, X_test, y_test)'
 if you get zero division errors and/or y_pred has values not in y_true,
 check and make sure your prediction values are expected.
 EX: 'arr([3,4,5,6,7,8,9])'
 while expected is from 0, therefore predictions will have 0,1,2 values that do not 
 exist in the check values.
"""


# run processed data through various models to determine which is best
def many_models(x,y,xt,yt):
    
    # Create instance of StandardScalar to fit and transform data
    scaled = StandardScaler()
    
    # Fit and Transform training data
    x_scale = scaled.fit_transform(x)
    
    # Transform test data
    xt_scale = scaled.transform(xt)
    
    # Create initial Random Forest Classifier model
    rf = RandomForestClassifier()
    
    # Fit data to Random forest model
    rf.fit(x,y)
    
    # Make predictions on Random Forest Model
    rf_predict = rf.predict(x)
    rf_test = rf.predict(xt)
    
    # Print scores and classification reports
    print(f'\nRandom Forest \nTest Accuracy: {rf.score(xt,yt)}\nbalanced test score: {balanced_accuracy_score(yt,rf_test)}')
    print(f'classification report: \n {classification_report(yt,rf_test)}')
    
    # Create Gradient Boosting Classifier model
    gr = GradientBoostingClassifier()
    
    # Fit data to Gradient Boosting Classifier model
    gr.fit(x,y)
    
    # Make predictions on Gradient Boosting Classifier model
    gr_predict = gr.predict(x)
    gr_test = gr.predict(xt)
    
    # Print scores and classification reports
    print(f'\nGradient Boost \nTest Accuracy: {gr.score(xt,yt)}\nbalanced test score: {balanced_accuracy_score(yt,gr_test)}')
    print(f'\n classification report: \n {classification_report(yt,gr_test)}')
    
    # Create Logistic Regression model
    lr = LogisticRegression(max_iter=240)
    
    # Fit data to Logistic Regression model
    lr.fit(x_scale,y)
    
    # Make predictions on Logistic Regression model
    lr_predict = lr.predict(x_scale)
    lr_test = lr.predict(xt_scale)
    
    # Print scores and classification reports    
    print(f'\nLogistic Regression \nTest Accuracy: {lr.score(xt_scale,yt)}\nbalanced test score: {balanced_accuracy_score(yt,lr_test)}')
    print(f'classification report: \n {classification_report(yt,lr_test)}')
    
    # Create Poly Support Vector Classifier model
    svc = SVC(kernel='poly')
    
    # Fit data to Poly Support Vector Classifier model
    svc.fit(x_scale,y)
    
    # Make predictions on Poly Support Vector Classifier model
    svc_predict = svc.predict(x_scale)
    svc_test = svc.predict(xt_scale)
    
    # Print scores and classification reports
    print(f'\nPoly Support Vector \nTest Accuracy: {svc.score(xt_scale,yt)}\nBalanced test score: {balanced_accuracy_score(yt,svc_test)}')
    print(f'classification report: \n {classification_report(yt,svc_test)}')

    
    # Adaptive Boosting models(slows the function down quite a lot, comment out or delete if need to speed up for rest)
    # Create Adaptive Boosting models
    ada_low = AdaBoostClassifier(n_estimators=20)
    # Fit data to low estimators Adaptive Boosting model 
    ada_low.fit(x_scale,y)
    
    # Make predictions on low estimators Adaptive Boosting model
    ada_low_predict = ada_low.predict(x_scale)
    ada_low_test = ada_low.predict(xt_scale)
    
    # print scores and classification reports
    print(f'\nADA low estimators \nTest Accuracy: {ada_low.score(xt_scale,yt)}\nbalanced test score: {balanced_accuracy_score(yt,ada_low_test)}')
    print(f'classification report: \n {classification_report(yt,ada_low_test)}')
    
    # Create Adaptive Boosting model (1500 estimators)
    ada = AdaBoostClassifier(n_estimators=1500)    
    
    # Fit data to Adaptive Boosting model (1500 estimators)
    ada.fit(x_scale,y)
    
    # Make predictions on Adaptive Boosting model (1500 estimators)
    ada_predict = ada.predict(x_scale)
    ada_test = ada.predict(xt_scale)
    
    # print scores and classification reports
    print(f'\nADA \nTest Accuracy: {ada.score(xt_scale,yt)}\nbalanced test score: {balanced_accuracy_score(yt,ada_test)}')
    print(f'classification report: \n {classification_report(yt,ada_test)}')
    
    # Create Sigmoid Support Vector Classifier model
    svc_sigmoid = SVC(kernel='sigmoid')
    
    # Fit data to Sigmoid Support Vector Classifier model
    svc_sigmoid.fit(x_scale,y)
    
    # Make predictions on Sigmoid Support Vector Classifier model
    svc_sigmoid_predict = svc_sigmoid.predict(x_scale)
    svc_sigmoid_test = svc_sigmoid.predict(xt_scale)
    
    # print scores and classification reports
    print(f'\nSVC Sigmoid \nTest Accuracy: {svc_sigmoid.score(xt_scale,yt)}\nbalanced test score: {balanced_accuracy_score(yt,svc_sigmoid_test)}')
    print(f'classification report: \n {classification_report(yt,svc_sigmoid_test)}')
    
    # Create comparison dataframe of the models with Model Name as Index
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
    
    # Make new column calculating difference between training and test scores
    comparison['Balanced Difference'] = (comparison['Balanced Trained Score'] - comparison['Balanced Test Score'])
    
    # Sort by Balanced Test Score column and print results
    comparison = comparison.sort_values(by='Balanced Test Score', ascending=False)
    # Drop Balanced Trained Score for easier readability and print output
        # comparison = comparison[['Trained Score','Test Score','Balanced Test Score','Balanced Difference']]
    
    comparison = comparison.drop('Balanced Trained Score',axis=1)
    print(comparison.head(7))
    
    # Plot the comparison dataframe on a bar graph for visualizations
    comparison.plot(kind='bar').tick_params(axis='x',rotation=45)
    
    
    
    
    
    # Create checks for Random Forest Best Parameters
    
    # Check best depth for random forest
    models = {'train_score': [], 'test_score': [], 'max_depth': []}
    
    # Loop through range to see where deviation happens
    for depth in range(1,20):
        
        # Save max depth to models
        models['max_depth'].append(depth)
        
        # Create model for max depth 
        model = RandomForestClassifier(max_depth=depth)
        
        # Fit the random forest model
        model.fit(x, y)
        
        # Make max depth predictions
        y_test_pred = model.predict(xt)
        y_train_pred = model.predict(x)
        
        # Save max depth scores to models
        models['train_score'].append(balanced_accuracy_score(y, y_train_pred))
        models['test_score'].append(balanced_accuracy_score(yt, y_test_pred))
        
    # Save models data to dataframe and sort by test scores    
    models_df = pd.DataFrame(models)#.sort_values(by='test_score',ascending=False)#.reset_index('max_depth')
    
    # Show graph of for loop
    models_df.plot(title='Random forest max depth', x='max_depth')
    
    # Set max_depth as index
    models_df_tolist = models_df.set_index('max_depth').sort_values(by='test_score',ascending=False)
    models_df = models_df.sort_values(by='test_score',ascending=False)
    print(f'\nRandom Forest Parameter Tuning:\n \nbest depth balanced test score: \n{models_df.head()}')
    
    # Save top 5 to a list
    best_rf = models_df_tolist.head().index.tolist()
    print(f'max depths: \n{best_rf}\n')
    
    
    
    
    
    
    # Test for leafs
    leaf = range(2,30)
    leaves = {'train_score':[],'test_score':[],'min_leaf': []}
    
    # Loop through leaf values
    for i in leaf:
        
        # Create Random Forest Classifier instance with depth
        model = RandomForestClassifier(min_samples_split=i)
        
        # Fit model
        model.fit(x,y)
        
        # Predict
        X_test_pred = model.predict(xt)
        y_train_pred = model.predict(x)
        
        # Store scores and best amounts in leaves dictionary
        leaves['train_score'].append(balanced_accuracy_score(y,y_train_pred))
        leaves['test_score'].append(balanced_accuracy_score(yt,X_test_pred))
        leaves['min_leaf'].append(i)
        
    # Store dictionary in a dataframe
    leaves_df = pd.DataFrame(leaves)
    
     # Plot best leaf on graph
    leaves_df.plot(title='Random Forest Best Leaf',x='min_leaf')
    
    # Set leaves_df index to 'min_leaf'
    leaves_df_tolist = leaves_df.set_index('min_leaf').sort_values(by='test_score',ascending=False)
    leaves_df = leaves_df.sort_values(by='test_score',ascending=False)
    print(f'best leaf balanced test score: \n{leaves_df.head()}')
    
    # Save to list
    best_split = leaves_df_tolist.head().index.tolist()
    print(f'best leaf \n{best_split}\n')
    
    
    
    
    
    
    
    # Test for best n_estimators
    # Create n_estimators range search
    est = range(10,100,1)
    
    # Create dictionary to store results
    est_models = {'train_score':[],'test_score':[],'n_estimators':[]}
    
    # Loop through range to find best n_estimators
    for i in est:
        
        # Create random forest model
        est_model = RandomForestClassifier(n_estimators=i)
        
        # Fit model 
        est_model.fit(x,y)
        
        # Make predictions for n_estimators
        X_test_pred = est_model.predict(xt)
        y_train_pred = est_model.predict(x)
        
        # Store results in dictionairy
        est_models['train_score'].append(balanced_accuracy_score(y,y_train_pred))
        est_models['test_score'].append(balanced_accuracy_score(yt,X_test_pred))
        est_models['n_estimators'].append(i)
        
    # Save to dataframe
    estimators_df = pd.DataFrame(est_models)
    
    # Graph best n_estimators
    estimators_df.plot(title='best n_estimators',x='n_estimators')
    
    # Set index to n_estimators
    estimators_df_tolist = estimators_df.set_index('n_estimators').sort_values(by='test_score',ascending=False)
    estimators_df = estimators_df.sort_values(by='test_score',ascending=False)
    print(f'\nbest n_estimators balanced test score: \n{estimators_df.head()}')
    
    # Save to list
    best_est = estimators_df_tolist.head().index.tolist()
    print(f'best n_estimators \n{best_est}\n')
    
    
    
    
   
    
    # Save best parameters to dictionary to use to find best combination
    best_params = {
        'n_estimators': best_est,
        'max_depth': best_rf,
        'min_samples_leaf': best_split
    }

    # Create dictionary to store results
    best_results = {'n_estimators':[],'max_depth':[],'min_samples_leaf':[],'train_score':[],'test_score':[]}
    
    # Loop through combinations(refactor in future for efficiency)
    for e in best_params['n_estimators']:
        for m in best_params['max_depth']:
            for l in best_params['min_samples_leaf']:
                
                # Create random forest model with best parameters
                rf_model = RandomForestClassifier(n_estimators=e,max_depth=m,min_samples_leaf=l)
                
                # Fit data
                rf_model.fit(x,y)
                
                # Predictions
                test_pred = rf_model.predict(xt)
                train_pred = rf_model.predict(x)
                
                # Append balanced scores
                best_results['train_score'].append(balanced_accuracy_score(y,train_pred))
                best_results['test_score'].append(balanced_accuracy_score(yt,test_pred))
                best_results['n_estimators'].append(e)
                best_results['max_depth'].append(m)
                best_results['min_samples_leaf'].append(l)
                    
    # Save best results to dataframe
    best_results_df = pd.DataFrame(best_results).sort_values(by='test_score',ascending=False)
    best_results_df = best_results_df.set_index('test_score')
    print(f'\nbest results:\n{best_results_df.head()}')
    
    # Save best parameters to print
    best_n = best_results_df['n_estimators'].iloc[0]
    best_d = best_results_df['max_depth'].iloc[0]
    best_l = best_results_df['min_samples_leaf'].iloc[0]
    print(f'best n_estimators: {best_n} \nbest max_depth: {best_d} \nbest leaves: {best_l}')
    
    # Create random forest instance with best parameters
    tuned_rf = RandomForestClassifier(n_estimators=best_results_df['n_estimators'].iloc[0], max_depth=best_results_df['max_depth'].iloc[0], min_samples_leaf=best_results_df['min_samples_leaf'].iloc[0],random_state=13)    
    
    # Train the tuned model
    tuned_rf.fit(x,y)
    
    # Predict with the tuned model
    best_train = tuned_rf.predict(x)
    best_test = tuned_rf.predict(xt)
    
    # View tuned Random Forest Model scores and classification report
    print(f'best Random Forest Tuned Parameters scores \n')
    print(f'\nBest Random Forest Tuned Parameters Scores \nTest Accuracy: {tuned_rf.score(xt,yt)}\nbalanced test score: {balanced_accuracy_score(yt,best_test)}')
    print(f'classification report: \n {classification_report(yt,best_test)}')
    
    # Add tuned Random Forest Model to the comparison DataFrame and Plot
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
    
    # Plot updated comparison model against initial models
    comparison_df.plot(kind='bar').tick_params(axis='x',rotation=45)
    
    
    return comparison,models_df,leaves_df,estimators_df,best_results_df,comparison_df










     #test_tuned_rf = RandomForestClassifier(n_estimators=estimators_df['n_estimators'].iloc[0],max_depth=models_df['max_depth'].iloc[0],min_samples_leaf=leaves_df['min_leaf'].iloc[0])
    #    # train the tuned model
    #test_tuned_rf.fit(x,y)
    #
    ## predict with the tuned model
    #test_tuned_train = test_tuned_rf.predict(x)
    #tuned_test = test_tuned_rf.predict(xt)
    #
    ## view tuned Random Forest Model scores and classification report
    #print(f'\nTest best Random Forest Tuned Parameters scores \n')
    #print(f'\nTest Best Random Forest Tuned Parameters Scores \nTest Accuracy: {test_tuned_rf.score(xt,yt)}\nbalanced test score: {balanced_accuracy_score(yt,tuned_test)}')
    #print(f'classification report: \n {classification_report(yt,tuned_test)}')
    #
    # # add tuned Random Forest Model to the comparison DataFrame and Plot
    #test_df = pd.DataFrame(
    #    [
    #        ['Test Random Forest',test_tuned_rf.score(x,y),test_tuned_rf.score(xt,yt),balanced_accuracy_score(y,test_tuned_train),balanced_accuracy_score(yt,tuned_test)]
    #    ],
    #        columns=['Model Name','Trained Score', 'Test Score','Balanced Trained Score','Balanced Test Score']
    #).set_index('Model Name')
    #test_df['Balanced Difference'] = (test_df['Balanced Trained Score'] - test_df['Balanced Test Score'])
    #test_df = test_df[['Trained Score','Test Score','Balanced Test Score','Balanced Difference']]
    #comparison_df = pd.concat([comparison,test_df]).sort_values(by='Balanced Test Score',ascending=False)
    #print(f'updated comparison DataFrame \n{comparison_df.head(8)}')
    
    
    
    
    # test for best n_estimators
    # create n_estimators range search
