import numpy as np
import pandas as pd
import random

import scipy.stats.stats as stats
import pandas.core.algorithms as algos
#from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from . import PipelineInputs as pl

from . import custom_classes 
from .custom_classes import  MonotonicBinning

random.seed(144)

import os
from mlflow import log_metric, log_param, log_artifacts

def feature_selector_model(data, target, max_features = None, numerical_feat_process = None, numerical_feat_missing = -9999, category_feat_process = 'woe', category_feat_missing= 'missing', method = 'RFE'):
    from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV, train_test_split
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, RobustScaler, LabelEncoder
    import category_encoders as ce
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    import numpy as np
    import pandas as pd
    import random
    import math
    from sklearn.feature_selection import RFE, SelectFromModel, SequentialFeatureSelector
    from sklearn.linear_model import Lasso, LogisticRegression
    import sklearn.metrics as skme
    random.seed(144)

    result_info = {}
    result_info['method'] = method

    n_features = data.shape[1]-1

    if max_features == None:
        max_features = max(int(math.log2(n_features)),5)

    result_info['max_features'] = max_features

    if category_feat_process == 'woe':
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=category_feat_missing)),
            ('cat', ce.woe.WOEEncoder(handle_unknown='value', random_state = 144))])

    else:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=category_feat_missing)),
            ('cat', OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -9999))])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())])

    numeric_features = list(data.select_dtypes(include=['int16','int32','int64','float16','float32','float64']).columns)
    if target in numeric_features:
        numeric_features.remove(target)
    categorical_features = list(data.select_dtypes(include=['object']).columns)
    if target in categorical_features:
        categorical_features.remove(target)
    
    result_info['numeric_features'] = numeric_features
    result_info['categorical_features'] = categorical_features

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
            ])

    X_train = data.drop([target],axis=1)
    Y_train = data[target].copy()

    if method == 'RFE':
        logit_model = LogisticRegression()
        varPipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feat', RFE(estimator = logit_model, n_features_to_select = max_features))])

    elif method == 'forward':
        logit_model = LogisticRegression()
        varPipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feat', SequentialFeatureSelector(estimator = logit_model, n_features_to_select = max_features, direction = 'forward'))])

    elif method == 'backward':
        logit_model = LogisticRegression()
        varPipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feat', SequentialFeatureSelector(estimator = logit_model, n_features_to_select = max_features, direction = 'backward'))])

    else:
        logit_model = LogisticRegression(C = 0.1, penalty = 'l1', solver = 'liblinear')
        varPipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feat', SelectFromModel(estimator = logit_model, max_features = max_features))])
    
    #data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
    #print("X : {}".format(X_train.isin([np.nan, np.inf, -np.inf]).any()))
    #print("Y : {}".format(Y_train.isin([np.nan, np.inf, -np.inf]).any()))
    #print(X_train.dtypes)
    varPipeline.fit(X_train, Y_train)
    result_info['variable_pipeline'] = varPipeline

    # categorical_feature_list = list(varPipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['cat'].get_feature_names())
    # all_features_list = list(numeric_features)
    # all_features_list.extend(categorical_feature_list)

    all_features_list = numeric_features
    all_features_list.extend(categorical_features)

    result_info['all_features'] = all_features_list

    result_info['selected_features'] = [x for x, y in zip(result_info['all_features'], list(varPipeline.named_steps['feat'].get_support())) if y == True]

    return result_info

def create_bin_model(data, target, category_feat_missing='missing', variableset = None, val_set = None, penalty = 'l1', tune_hyperparams = True, param_grid = None, get_summary_report = True):
    from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV
    # from xverse.transformer import WOE
    from sklearn.preprocessing import RobustScaler
    import category_encoders as ce
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    import numpy as np
    import pandas as pd
    import random
    import math
    from sklearn.linear_model import Lasso, LogisticRegression
    import sklearn.metrics as skme
    random.seed(144)

    result_model = {}
    result_model['target'] = 'yTarget'

    data.reset_index(drop = True, inplace = True)

    if variableset is not None:
        variableset.append(target)
        data = data[variableset]
        if val_set is not None:
            val_set = val_set[variableset]

    if tune_hyperparams == False:
        if val_set is None:
            param_grid = None
    else:
        param_grid = { 
        'classifier__C' : np.logspace(-3, 1, 5),
        'classifier__penalty' : ['l1', 'l2', 'elasticnet', 'none'],
        'classifier__solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
        'classifier__max_iter' : [1000, 1500, 2500, 3500, 5000],
        }

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=category_feat_missing)),
        ('cat_woe', ce.woe.WOEEncoder(handle_unknown='value', random_state = 144))])

    # numeric_transformer = Pipeline(steps=[
    #     #('bins', MonotonicBinning()),
    #     ('num_woe', WOE(treat_missing = 'separate'))])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())])

    numeric_features = list(data.select_dtypes(include=['int16','int32','int64','float16','float32','float64']).columns)
    if target in numeric_features:
        numeric_features.remove(target)
    categorical_features = list(data.select_dtypes(include=['object','category']).columns)
    if target in categorical_features:
        categorical_features.remove(target)
    
    result_model['numeric_features'] = numeric_features
    result_model['categorical_features'] = categorical_features

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
            ])

    X_train = data.drop([target],axis=1)
    Y_train = data[target].copy()

    numBinTransformer = MonotonicBinning(force_bins=5)
    numBinTransformer.fit(X_train[numeric_features],Y_train)

    X_train[numeric_features] = numBinTransformer.transform(X_train[numeric_features])

    result_model['monotonic_binning_transformer'] = numBinTransformer

    up_numeric_features = list(X_train.select_dtypes(include=['int16','int32','int64','float16','float32','float64']).columns)
    up_categorical_features = list(X_train.select_dtypes(include=['object','category']).columns)

    if not up_numeric_features:
        preprocessor = ColumnTransformer(
            transformers=[
                ('categorical', categorical_transformer, up_categorical_features)
                ])

    if penalty == 'l2':
        logit_model = LogisticRegression(C = 0.1, penalty = 'l2', random_state = 144)
        if tune_hyperparams:
            param_grid = { 
                'classifier__C' : np.logspace(-3, 1, 5),
                'classifier__max_iter' : [1000, 1500, 2500, 3500, 5000],
                'classifier__solver' : ['lbfgs','newton-cg','sag'],
                }
    elif penalty == 'elastinet':
        logit_model = LogisticRegression(C = 0.1, penalty = 'elastinet', solver = 'saga', random_state = 144)
        if tune_hyperparams:
            param_grid = { 
                'classifier__C' : np.logspace(-3, 1, 5),
                'classifier__max_iter' : [1000, 1500, 2500, 3500, 5000],
                }
    elif penalty == 'l1':
        logit_model = LogisticRegression(C = 0.1, penalty = 'l1', solver = 'liblinear', random_state = 144, max_iter = 2000)
        if tune_hyperparams:
            param_grid = { 
                'classifier__C' : np.logspace(-3, 1, 5),
                'classifier__max_iter' : [1000, 1500, 2500, 3500, 5000],
                'classifier__solver' : ['liblinear','saga'],
                }
    else:
        logit_model = LogisticRegression(C = 0.1, random_state = 144)
        
    classifierPipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', logit_model)])

    classifierPipeline.fit(X_train, Y_train)

    if tune_hyperparams:
        print('Performing Hyperparamter tuning')
        try:
            tunedclassifierPipeline = RandomizedSearchCV(classifierPipeline, param_distributions = param_grid, n_iter = 20, scoring = 'roc_auc', n_jobs = -1, cv = 5, random_state = 144)
            tunedclassifierPipeline.fit(X_train, Y_train)
            print(tunedclassifierPipeline.best_params_)    
            print(tunedclassifierPipeline.best_score_)
            classifierPipeline = tunedclassifierPipeline.best_estimator_
            result_model['param_grid'] = param_grid
        except:
            pass
    
    result_model['model_bins'] = X_train
    #print("X train type  "+type(X_train))
    result_model['model_pipeline'] = classifierPipeline
    model_features = np.append(['Intercept'], classifierPipeline.named_steps['preprocessor'].transformers_[0][2])
    result_model['model_features'] = model_features
    model_coefficients = np.append(classifierPipeline.named_steps["classifier"].intercept_[0], classifierPipeline.named_steps["classifier"].coef_[0])
    result_model['model_coefficients'] = model_coefficients

    if get_summary_report:
        y_fitted = classifierPipeline.predict(X_train)
        confusion_matrix = skme.confusion_matrix(Y_train, y_fitted)
        classification_report = skme.classification_report(Y_train, y_fitted)
        roc_auc_score = skme.roc_auc_score(Y_train, y_fitted)
        model_parameters = classifierPipeline.named_steps["classifier"].get_params()
        md_coeff = pd.DataFrame()
        md_coeff['model_features'] = model_features
        md_coeff['model_coefficients'] = model_coefficients
        import json
        with open(pl.DefaultInfo.default_report_path+"/Model_Training_Summary.html", "w", encoding = 'utf-8') as file:
            file.write(str("<h2>Logistic Model Summary</h2>"))
            file.write(str('<br/>'))        
            file.write(str('<h3>Logistic Model Parameters</h3>'))            
            file.write('<h3>'+str('Model Coefficients')+'</h3>')
            file.write('<h3>'+str(model_parameters)+'</h3>')
            pd.set_option("display.max_rows", None, "display.max_columns", None)
            file.write(convert_df_to_html(md_coeff))
            pd.reset_option("display.max_rows")
            pd.reset_option("display.max_columns")
            file.write(str('<br/>')) 
            file.write(str('<h3>Model Report on Train Data</h3>'))
            file.write(str('<h3>Train Data Confusion Marix</h3>'))            
            file.write('<h3>'+str(confusion_matrix)+'</h3>')
            file.write(str('<h3>Train Data ROC AUC Score</h3>'))
            file.write('<h3>'+str(roc_auc_score)+'</h3>')
            file.write(str('<h3>Train Data Classification Report<h3>'))
            file.write(str(classification_report))
            file.close()

    return result_model
'''
def make_predictions(test_data, result_model_pipeline):
    from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV
    from xverse.transformer import WOE
    import category_encoders as ce
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    import numpy as np
    import pandas as pd
    import random
    import math
    from sklearn.linear_model import Lasso, LogisticRegression
    import sklearn.metrics as skme
    random.seed(144)

    predictions = {}
    target = result_model_pipeline['target']
    y_test = test_data[target].copy()
    predictions['ytest'] = y_test

    numeric_features = result_model_pipeline['numeric_features'].copy()
    categorical_features = result_model_pipeline['categorical_features'].copy()
    features = numeric_features.copy()
    features.extend(categorical_features)

    dtest = test_data[features].copy()

    numBinTransformer = result_model_pipeline['monotonic_binning_transformer']
    dtest[numeric_features] = numBinTransformer.transform(dtest[numeric_features])

    classifierPipeline = result_model_pipeline['model_pipeline']

    y_pred = classifierPipeline.predict(dtest)
    predictions['ypred'] = y_pred
    predictions['confusion_matrix'] = skme.confusion_matrix(y_test, y_pred)
    predictions['classification_report'] = skme.classification_report(y_test, y_pred)
    predictions['roc_auc_score'] = skme.roc_auc_score(y_test, y_pred)
    predictions['binned_variables'] = dtest.copy()
    print(skme.confusion_matrix(y_test, y_pred))
    print(skme.classification_report(y_test, y_pred))
    print(skme.roc_auc_score(y_test, y_pred))

    y_pred_probability = classifierPipeline.predict_proba(dtest)

    predictions['y_pred_probability'] = y_pred_probability

    return predictions

def test():
    trainfilepath = r'Project_1_Train_Data.csv'
    testfilepath = r'Project_1_OOS_Data.csv'

    dtrain = pd.read_csv(trainfilepath)
    dtrain.drop(['Unnamed: 0','loan_status'], axis = 1, inplace = True)

    dtest = pd.read_csv(testfilepath)
    dtest.drop(['Unnamed: 0','loan_status'], axis = 1, inplace = True)

    dtrain['yTarget'] = np.where(dtrain['yTarget'].isin(['Level0']),1,0)
    dtest['yTarget'] = np.where(dtest['yTarget'].isin(['Level0']),1,0)

    result_variables = feature_selector_model(data = dtrain.copy(), target = 'yTarget', numerical_feat_process = None, 
                                              numerical_feat_missing = -9999, category_feat_process = 'woe', 
                                              category_feat_missing= 'missing', method = 'Lasso')

    result_model_pipelines = create_bin_model(data = dtrain.copy(), target = 'yTarget', 
        variableset = result_variables['selected_features'].copy(), 
                                              val_set = None, penalty = 'l1', param_grid = None)

    test_predictions = make_predictions(test_data = dtest.copy(), result_model_pipeline = result_model_pipelines)
'''
def create_categorical_mapping(category_maps, category_list):
    category_map_list = list(category_maps.keys())
    np_categories = list(set(category_list)-set(category_map_list))
    if len(np_categories) > 0:
        for category in np_categories:
            category_maps.update({category : category})
    return category_maps
'''
def RefineBSModel(data, target, variableset = None, category_feat_missing = 'missing', 
 numeric_trasformations = None, numeric_bining = True, custom_binning = None, 
  categorical_grouping = None, penalty = 'l1', tune_hyperparams = False, param_grid = None, 
   get_summary_report = True):
    from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV
    # from xverse.transformer import WOE
    from sklearn.preprocessing import RobustScaler
    import category_encoders as ce
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    import numpy as np
    import pandas as pd
    import random
    import math
    from sklearn.linear_model import Lasso, LogisticRegression
    import sklearn.metrics as skme
    random.seed(144)

    result_model = {}
    result_model['target'] = 'yTarget'

    data.reset_index(drop = True, inplace = True)
    if variableset is not None:
        variableset.append(target)
        data = data[variableset]

    if tune_hyperparams == False:
        param_grid = None
    else:
        param_grid = { 
        'classifier__C' : np.logspace(-3, 1, 5),
        'classifier__penalty' : ['l1', 'l2', 'elasticnet', 'none'],
        'classifier__solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
        'classifier__max_iter' : [1000, 1500, 2500, 3500, 5000],
        }

    if categorical_grouping is not None:
        result_model['categorical_grouping'] = True
        #result_model['categorical_grouping'] = categorical_grouping
        result_model['group_maps'] = {}
        for col, group_maps in categorical_grouping.items():
            collevels = list(data[col].unique())
            cat_group_maps = create_categorical_mapping(category_maps = group_maps, category_list = collevels)
            result_model['group_maps'].update({col : cat_group_maps})
            data[col] = data[col].map(cat_group_maps)
    else:
        result_model['categorical_grouping'] = False

    if numeric_trasformations is not None:
        print('Performing Numeric Transformations')
        result_model['numeric_trasformations'] = numeric_trasformations
        for col, transf in numeric_trasformations.items():
            if transf == 'log':
                data[col] = data[col].apply(lambda x: np.log(x))
            elif transf == 'sqrt':
                data[col] = data[col].apply(lambda x: math.sqrt(x))
            else:
                data[col] = data[col].apply(lambda x: (1/x) if x!=0 else 0)
    else:
        result_model['numeric_trasformations'] = None
        numeric_bining = True

    numeric_features = list(data.select_dtypes(include=['int16','int32','int64','float16','float32','float64']).columns)
    if target in numeric_features:
        numeric_features.remove(target)
    categorical_features = list(data.select_dtypes(include=['object','category']).columns)
    if target in categorical_features:
        categorical_features.remove(target)
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=category_feat_missing)),
        ('cat_woe', ce.woe.WOEEncoder(handle_unknown='value', random_state = 144))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
            ])
    
    result_model['numeric_features'] = numeric_features
    result_model['categorical_features'] = categorical_features

    X_train = data.drop([target],axis=1)
    Y_train = data[target].copy()

    result_model['numeric_bining'] = numeric_bining

    if numeric_bining:
        if custom_binning is None:
            numBinTransformer = MonotonicBinning(force_bins=5)
            numBinTransformer.fit(X_train[numeric_features],Y_train)
            X_train[numeric_features] = numBinTransformer.transform(X_train[numeric_features])
            result_model['monotonic_binning_transformer'] = numBinTransformer
        else:
            numBinTransformer = MonotonicBinning(custom_binning = custom_binning)
            X_train[numeric_features] = numBinTransformer.transform(X_train[numeric_features])
            result_model['monotonic_binning_transformer'] = numBinTransformer

    up_numeric_features = list(X_train.select_dtypes(include=['int16','int32','int64','float16','float32','float64']).columns)
    up_categorical_features = list(X_train.select_dtypes(include=['object','category']).columns)

    if not up_numeric_features:
        preprocessor = ColumnTransformer(
            transformers=[
                ('categorical', categorical_transformer, up_categorical_features)
                ])

    if penalty == 'l2':
        logit_model = LogisticRegression(C = 0.1, penalty = 'l2', random_state = 144)
        if tune_hyperparams:
            param_grid = { 
                'classifier__C' : np.logspace(-3, 1, 5),
                'classifier__max_iter' : [1000, 1500, 2500, 3500, 5000],
                'classifier__solver' : ['lbfgs','newton-cg','sag'],
                }
    elif penalty == 'elastinet':
        logit_model = LogisticRegression(C = 0.1, penalty = 'elastinet', solver = 'saga', random_state = 144)
        if tune_hyperparams:
            param_grid = { 
                'classifier__C' : np.logspace(-3, 1, 5),
                'classifier__max_iter' : [1000, 1500, 2500, 3500, 5000],
                }
    elif penalty == 'l1':
        logit_model = LogisticRegression(C = 0.1, penalty = 'l1', solver = 'liblinear', random_state = 144, max_iter = 5000)
        if tune_hyperparams:
            param_grid = { 
                'classifier__C' : np.logspace(-3, 1, 5),
                'classifier__max_iter' : [1000, 1500, 2500, 3500, 5000],
                'classifier__solver' : ['liblinear','saga'],
                }
    else:
        logit_model = LogisticRegression(C = 0.1, random_state = 144)
        
    classifierPipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', logit_model)])

    classifierPipeline.fit(X_train, Y_train)

    if tune_hyperparams:
        print('Performing Hyperparamter tuning')
        try:
            tunedclassifierPipeline = RandomizedSearchCV(classifierPipeline, param_distributions = param_grid, n_iter = 20, scoring = 'roc_auc', n_jobs = -1, cv = 5, random_state = 144)
            tunedclassifierPipeline.fit(X_train, Y_train)
            print(tunedclassifierPipeline.best_params_)    
            print(tunedclassifierPipeline.best_score_)
            classifierPipeline = tunedclassifierPipeline.best_estimator_
            result_model['param_grid'] = param_grid
        except:
            pass

    result_model['model_pipeline'] = classifierPipeline
    if numeric_bining:
        model_features = np.append(['Intercept'], classifierPipeline.named_steps['preprocessor'].transformers_[0][2])
    else:
        model_features = list(np.append(['Intercept'], classifierPipeline.named_steps['preprocessor'].transformers_[0][2]))
        model_features.extend(list(classifierPipeline.named_steps['preprocessor'].transformers_[1][2]))
    result_model['model_features'] = model_features
    model_coefficients = np.append(classifierPipeline.named_steps["classifier"].intercept_[0], classifierPipeline.named_steps["classifier"].coef_[0])
    result_model['model_coefficients'] = model_coefficients

    if get_summary_report:
        y_fitted = classifierPipeline.predict(X_train)
        confusion_matrix = skme.confusion_matrix(Y_train, y_fitted)
        classification_report = skme.classification_report(Y_train, y_fitted,output_dict=True)
        roc_auc_score = skme.roc_auc_score(Y_train, y_fitted)
        model_parameters = classifierPipeline.named_steps["classifier"].get_params()
        #print(type(model_parameters))
        md_coeff = pd.DataFrame()
        md_coeff['model_features'] = model_features
        md_coeff['model_coefficients'] = model_coefficients
        import json
        with open(pl.DefaultInfo.default_report_path+"/FineTuning_Summary.html", "w", encoding = 'utf-8') as file:
            file.write(str("<h2>Logistic Model Summary</h2>"))            
            file.write(str('<h3>Logistic Model Parameters</h3>'))    
            file.write(convert_df_to_html(model_parameters))
            file.write(str('<br/>'))
            file.write(str('<h3>Model Coefficients</h3>'))
            pd.set_option("display.max_rows", None, "display.max_columns", None)
            file.write(convert_df_to_html(md_coeff,hide_index=True))
            pd.reset_option("display.max_rows")
            pd.reset_option("display.max_columns")
            file.write(str('<h3>Model Report on Train Data</h3>'))
            file.write(str('<h3>Train Data Confusion Marix</h3>'))
            file.write(str(confusion_matrix))
            file.write(str('<h3>Train Data ROC AUC Score</h3>'))
            file.write("<h3 style=\'bold\'>"+str(roc_auc_score)+"</h3>")
            file.write(str('<br/>'))
            file.write(str('<h3>Train Data Classification Report</h3>'))
            file.write(str('<br/>'))
            file.write(str(classification_report))
            file.close()

    return result_model

def RefineBSPredictions(test_data, result_model_pipeline):
    import category_encoders as ce
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    import numpy as np
    import pandas as pd
    import random
    import math
    from sklearn.linear_model import Lasso, LogisticRegression
    import sklearn.metrics as skme
    random.seed(144)

    test_data.reset_index(drop = True, inplace = True)

    predictions = {}
    target = result_model_pipeline['target']
    y_test = test_data[target].copy()
    predictions['ytest'] = y_test

    numeric_features = result_model_pipeline['numeric_features'].copy()
    categorical_features = result_model_pipeline['categorical_features'].copy()
    features = numeric_features.copy()
    features.extend(categorical_features)

    dtest = test_data[features].copy()

    categorical_grouping = result_model_pipeline['categorical_grouping']
    if categorical_grouping:
        for col, cat_group_maps in result_model_pipeline['group_maps'].items():
            dtest[col] = dtest[col].map(cat_group_maps)
    
    numeric_trasformations = result_model_pipeline['numeric_trasformations']
    if numeric_trasformations is not None:
        for col, transf in numeric_trasformations.items():
            if transf == 'log':
                dtest[col] = dtest[col].apply(lambda x: np.log(x))
            elif transf == 'sqrt':
                dtest[col] = dtest[col].apply(lambda x: math.sqrt(x))
            else:
                dtest[col] = dtest[col].apply(lambda x: (1/x) if x!=0 else 0)
    
    numeric_binning = result_model_pipeline['numeric_bining']
    if numeric_binning:
        numBinTransformer = result_model_pipeline['monotonic_binning_transformer']
        dtest[numeric_features] = numBinTransformer.transform(dtest[numeric_features])

    classifierPipeline = result_model_pipeline['model_pipeline']

    y_pred = classifierPipeline.predict(dtest)
    predictions['ypred'] = y_pred

    print(skme.confusion_matrix(y_test, y_pred))
    print(skme.classification_report(y_test, y_pred))
    print(skme.roc_auc_score(y_test, y_pred))

    y_pred_probability = classifierPipeline.predict_proba(dtest)
    predictions['y_pred_probability'] = y_pred_probability
    predictions['classification_report'] = skme.classification_report(y_test, y_pred)

    return predictions
'''

'''
def create_bin_model(data, target, category_feat_missing='missing', 
variableset = None, val_set = None, penalty = 'l1', tune_hyperparams = True, 
param_grid = None, get_summary_report = True):
    from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV
    # from xverse.transformer import WOE
    from sklearn.preprocessing import RobustScaler
    import category_encoders as ce
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    import numpy as np
    import pandas as pd
    import random
    import math
    from sklearn.linear_model import Lasso, LogisticRegression
    import sklearn.metrics as skme
    random.seed(144)

    result_model = {}
    result_model['target'] = 'yTarget'

    data.reset_index(drop = True, inplace = True)

    if variableset is not None:
        variableset.append(target)
        data = data[variableset]
        if val_set is not None:
            val_set = val_set[variableset]

    if tune_hyperparams == False:
        if val_set is None:
            param_grid = None
    else:
        param_grid = { 
        'classifier__C' : np.logspace(-3, 1, 5),
        'classifier__penalty' : ['l1', 'l2', 'elasticnet', 'none'],
        'classifier__solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
        'classifier__max_iter' : [100, 1000, 2500, 5000]
        }

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=category_feat_missing)),
        ('cat_woe', ce.woe.WOEEncoder(handle_unknown='value', random_state = 144))])

    # numeric_transformer = Pipeline(steps=[
    #     #('bins', MonotonicBinning()),
    #     ('num_woe', WOE(treat_missing = 'separate'))])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())])

    numeric_features = list(data.select_dtypes(include=['int16','int32','int64','float16','float32','float64']).columns)
    if target in numeric_features:
        numeric_features.remove(target)
    categorical_features = list(data.select_dtypes(include=['object','category']).columns)
    if target in categorical_features:
        categorical_features.remove(target)
    
    result_model['numeric_features'] = numeric_features
    result_model['categorical_features'] = categorical_features

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
            ])

    X_train = data.drop([target],axis=1)
    Y_train = data[target].copy()

    numBinTransformer = MonotonicBinning(force_bins=5)
    numBinTransformer.fit(X_train[numeric_features],Y_train)

    X_train[numeric_features] = numBinTransformer.transform(X_train[numeric_features])

    result_model['monotonic_binning_transformer'] = numBinTransformer

    up_numeric_features = list(X_train.select_dtypes(include=['int16','int32','int64','float16','float32','float64']).columns)
    up_categorical_features = list(X_train.select_dtypes(include=['object','category']).columns)

    if not up_numeric_features:
        preprocessor = ColumnTransformer(
            transformers=[
                ('categorical', categorical_transformer, up_categorical_features)
                ])

    if penalty == 'l2':
        logit_model = LogisticRegression(C = 0.1, class_weight='balanced', penalty = 'l2', random_state = 144)
        if tune_hyperparams:
            param_grid = { 
                'classifier__C' : np.logspace(-3, 1, 5),
                'classifier__max_iter' : [100, 1000, 2500, 5000]
                }
    elif penalty == 'elastinet':
        logit_model = LogisticRegression(C = 0.1, class_weight='balanced', penalty = 'elastinet', solver = 'saga', random_state = 144)
        if tune_hyperparams:
            param_grid = { 
                'classifier__C' : np.logspace(-3, 1, 5),
                'classifier__max_iter' : [100, 1000, 2500, 5000]
                }
    elif penalty == 'l1':
        logit_model = LogisticRegression(C = 0.1, penalty = 'l1', solver = 'liblinear', class_weight='balanced', random_state = 144)
        if tune_hyperparams:
            param_grid = { 
                'classifier__C' : np.logspace(-3, 1, 5),
                'classifier__max_iter' : [100, 1000, 2500, 5000]
                }
    else:
        logit_model = LogisticRegression(C = 0.1, class_weight='balanced', random_state = 144)
        
    classifierPipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', logit_model)])

    classifierPipeline.fit(X_train, Y_train)

    if tune_hyperparams:
        print('Performing Hyperparamter tuning')
        try:
            tunedclassifierPipeline = RandomizedSearchCV(classifierPipeline, param_distributions = param_grid, n_iter= 10, scoring = 'roc_auc', n_jobs = -1, cv = 5, random_state = 144)
            tunedclassifierPipeline.fit(X_train, Y_train)
            print(tunedclassifierPipeline.best_params_)    
            print(tunedclassifierPipeline.best_score_)
            classifierPipeline = tunedclassifierPipeline.best_estimator_
            result_model['param_grid'] = param_grid
        except:
            pass

    result_model['model_pipeline'] = classifierPipeline
    model_features = np.append(['Intercept'], classifierPipeline.named_steps['preprocessor'].transformers_[0][2])
    result_model['model_features'] = model_features
    model_coefficients = np.append(classifierPipeline.named_steps["classifier"].intercept_[0], classifierPipeline.named_steps["classifier"].coef_[0])
    result_model['model_coefficients'] = model_coefficients

    if get_summary_report:
        y_fitted = classifierPipeline.predict(X_train)
        confusion_matrix = skme.confusion_matrix(Y_train, y_fitted)
        classification_report = skme.classification_report(Y_train, y_fitted)
        roc_auc_score = skme.roc_auc_score(Y_train, y_fitted)
        model_parameters = classifierPipeline.named_steps["classifier"].get_params()
        md_coeff = pd.DataFrame()
        md_coeff['model_features'] = model_features
        md_coeff['model_coefficients'] = model_coefficients
        import json
        with open("Project_1_Training_Summary.html", "w", encoding = 'utf-8') as file:
            file.write(str("Logistic Model Summary"))
            file.write(str('\n'))
            file.write(str('\n'))
            file.write(str('Logistic Model Parameters'))
            file.write(str('\n'))
            file.write(str(model_parameters))
            file.write(str('\n'))
            file.write(str('\n'))
            file.write(str('Model Coefficients'))
            pd.set_option("display.max_rows", None, "display.max_columns", None)
            file.write(str(md_coeff))
            pd.reset_option("display.max_rows")
            pd.reset_option("display.max_columns")
            file.write(str('\n'))
            file.write(str('\n'))
            file.write(str('Model Report on Train Data'))
            file.write(str('\n'))
            file.write(str('Train Data Confusion Marix'))
            file.write(str('\n'))
            file.write(str(confusion_matrix))
            file.write(str('\n'))
            file.write(str('Train Data ROC AUC Score'))
            file.write(str('\n'))
            file.write(str(roc_auc_score))
            file.write(str('\n'))
            file.write(str('Train Data Classification Report'))
            file.write(str('\n'))
            file.write(str(classification_report))
            file.close()

    return result_model
'''
html_template = "<style type=\"text/css\">table,tr{border:1px solid black;border-spacing:0;text-align:center}tr th{background-color:#47a0ff;border-bottom:1px solid black;border-right:1px solid black;color:darkblue}tr td{border-right:1px solid black}</style><div>{table}</div>"

def convert_df_to_html(dataframe, hide_index=False):                       
        html_output = html_template        
        table_contents = ""
        temp_df = dataframe
        if isinstance(dataframe, dict):
            temp_df = pd.DataFrame(dataframe, index=[0])
        final_df = temp_df
        if(hide_index):        
            table_contents = final_df.style.hide_index().render()
        else:
            table_contents = final_df.style.render()
        # write html to file
        html_output = html_template.replace('{table}', table_contents)
        return html_output
    