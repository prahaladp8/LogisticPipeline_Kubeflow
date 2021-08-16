import numpy as np
import pandas as pd

from .custom_classes import LogitWrapper, MonotonicBinning

def create_bin_model(data, target, category_feat_missing='missing', variableset = None, val_set = None,
         get_Logit_summary = True, alpha = None, penalty = 'l1', tune_hyperparams = False, 
          param_grid = None, get_summary_report = True, risk_band_count=5):
    from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV
    # from xverse.transformer import WOE
    from sklearn.preprocessing import RobustScaler, OneHotEncoder
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

    if get_Logit_summary:
        tune_hyperparams = False
        result_model['Model_tracking'] = 'Stats API Logit'
        if alpha is None:
            alpha = 0.8
    else:
        result_model['Model_tracking'] = 'SKLearn LogisticRegression'

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

    # categorical_transformer = Pipeline(steps=[
    #     ('imputer', SimpleImputer(strategy='constant', fill_value=category_feat_missing)),
    #     ('cat_woe', ce.woe.WOEEncoder(handle_unknown='value', random_state = 144))])

    # numeric_transformer = Pipeline(steps=[
    #     #('bins', MonotonicBinning()),
    #     ('num_woe', WOE(treat_missing = 'separate'))])
    
    categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())])

    numeric_features = list(data.select_dtypes(include=['int16','int32','int64','float16','float32','float64']).columns)
    if target in numeric_features:
        numeric_features.remove(target)
    categorical_features = list(data.select_dtypes(include=['object','category']).columns)
    if target in categorical_features:
        categorical_features.remove(target)
    
    result_model['final_feature_list'] = numeric_features + categorical_features
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

    replace_str_features = list(X_train.select_dtypes(include=['object']).columns)
    X_train[replace_str_features] = X_train[replace_str_features].fillna('missing')  

    replace_categorical_features = list(X_train.select_dtypes(include=['category']).columns)

    for cols in replace_categorical_features:
        if type(X_train[cols].cat.categories.dtype) is pd.core.dtypes.dtypes.IntervalDtype:
            X_train[cols] = X_train[cols].cat.add_categories([pd.Interval(-999, -999)])
            X_train[cols] = X_train[cols].cat.as_ordered()
            X_train[cols] = X_train[cols].fillna(pd.Interval(-999, -999))

        if type(X_train[cols].cat.categories.dtype) is object:
            X_train[cols] = X_train[cols].cat.add_categories(['missing'])
            X_train[cols] = X_train[cols].fillna('missing')
            
        if type(X_train[cols].cat.categories.dtype) is str:
            X_train[cols] = X_train[cols].cat.add_categories(['missing'])

                
        print("{} {}".format(cols, X_train[cols].cat.categories.dtype))
    
    print('New Empty/Nan Bins')
    for cols in replace_categorical_features:
        print(X_train[cols].value_counts())


    result_model['monotonic_binning_transformer'] = numBinTransformer

    up_numeric_features = list(X_train.select_dtypes(include=['int16','int32','int64','float16','float32','float64']).columns)
    up_categorical_features = list(X_train.select_dtypes(include=['object','category']).columns)

    if not up_numeric_features:
        preprocessor = ColumnTransformer(
            transformers=[
                ('categorical', categorical_transformer, up_categorical_features)
                ])

    if get_Logit_summary:
        logit_model = LogitWrapper(alpha = alpha)
        tune_hyperparams = False
    
    else:
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
    #print(preprocessor)

    if tune_hyperparams:
        print('Performing Hyperparamter tuning')
        try:
            tunedclassifierPipeline = RandomizedSearchCV(classifierPipeline, param_distributions = param_grid, n_iter = 20, scoring = 'roc_auc', n_jobs = -1, cv = 5, random_state = 144)
            tunedclassifierPipeline.fit(X_train, Y_train)
            print(tunedclassifierPipeline.best_params_)    
            print(tunedclassifierPipeline.best_score_)
            classifierPipeline = tunedclassifierPipeline.best_estimator_
            result_model['param_grid'] = param_grid
            #add code specific to the hyperparam tuning code here.
        except:
            pass
    
    result_model['model_bins'] = X_train
    #print("X train type  "+type(X_train))
    result_model['model_pipeline'] = classifierPipeline
    #model_features = np.append(['Intercept'], classifierPipeline.named_steps['preprocessor'].transformers_[0][2])
    
    #onehot encoding changes
    featlist = classifierPipeline.named_steps['preprocessor'].transformers_[0][2]
    classifierPipeline.named_steps['preprocessor'].transformers_[0][1]['onehot'].get_feature_names(featlist) 
    model_features = np.append(['Intercept'], classifierPipeline.named_steps['preprocessor'].transformers_[0][1]['onehot'].get_feature_names(featlist))
    
    result_model['model_features'] = model_features
    if get_Logit_summary:
        model_coefficients = classifierPipeline.named_steps["classifier"].results_.params
    else:
        model_coefficients = np.append(classifierPipeline.named_steps["classifier"].intercept_[0], classifierPipeline.named_steps["classifier"].coef_[0])
    result_model['model_coefficients'] = model_coefficients

    #Code for creation of Risk Bands
    train_predictions = classifierPipeline.predict_proba(X_train)
    
    
    result_model['pd_risk_bands'] = pd.Series(np.quantile(train_predictions,np.linspace(0,1,risk_band_count+1))) 
    result_model['pd_risk_bands'].name = 'Risk Bands'   

    risk_bands = result_model['pd_risk_bands']
    band_outs , bins = pd.cut(train_predictions,risk_bands,retbins=True)
    
    model_risk_band_df = pd.DataFrame([train_predictions,band_outs])
    model_risk_band_df = model_risk_band_df.T
    model_risk_band_df.columns = ['Predicted','Risk_Bands']
    
    result_model['model_risk_df_train'] = model_risk_band_df


    if get_summary_report:
        y_fitted = classifierPipeline.predict(X_train)
        confusion_matrix = skme.confusion_matrix(Y_train, y_fitted)
        classification_report = skme.classification_report(Y_train, y_fitted)
        roc_auc_score = skme.roc_auc_score(Y_train, y_fitted)
        if get_Logit_summary == False:
            model_parameters = classifierPipeline.named_steps["classifier"].get_params()
        else:
            res_summary = classifierPipeline.named_steps["classifier"].summary()
            result_model['model_summary'] = str(res_summary)
        md_coeff = pd.DataFrame()
        md_coeff['model_features'] = model_features
        md_coeff['model_coefficients'] = model_coefficients
        import json
        '''
        with open("Reports/Project_1_Training_Summary.html", "w", encoding = 'utf-8') as file:
            file.write(str("Logistic Model Summary"))
            file.write(str('\n'))
            file.write(str('\n'))
            if get_Logit_summary == False:
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
            if get_Logit_summary:
                print(res_summary.as_text())
                file.write(str('\n'))
                file.write(str('Model Summary'))
                file.write(str('\n'))
                file.write(res_summary.as_text())
            file.close()
        '''
    return result_model


def create_risk_bands(bands,data):
    pass

def make_predictions(test_data, result_model_pipeline):
    from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV
    
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

    numBinTransformer = result_model_pipeline['monotonic_binning_transformer']
    dtest[numeric_features] = numBinTransformer.transform(dtest[numeric_features])


    print('Checking for nulls/nans in dataset after numeric binning')
    print(dtest.isna().sum())
    
    #replacing empty values with missing / -999 in the testing dataset
    #Fetching columns with dtypes 
    replace_str_features = list(dtest.select_dtypes(include=['object']).columns)
    dtest[replace_str_features] = dtest[replace_str_features].fillna('missing')  

    replace_categorical_features = list(dtest.select_dtypes(include=['category']).columns)

    for cols in replace_categorical_features:            
        if type(dtest[cols].cat.categories.dtype) is pd.core.dtypes.dtypes.IntervalDtype:
            dtest[cols] = dtest[cols].cat.add_categories([pd.Interval(-999, -999)])
            dtest[cols] = dtest[cols].cat.as_ordered()
            dtest[cols] = dtest[cols].fillna(pd.Interval(-999, -999))

        if type(dtest[cols].cat.categories.dtype) is object:
            dtest[cols] = dtest[cols].cat.add_categories(['missing'])
            dtest[cols] = dtest[cols].fillna('missing')

        if type(dtest[cols].cat.categories.dtype) is str:
            dtest[cols] = dtest[cols].cat.add_categories(['missing'])                            
        
        
    print('Post imputing for nulls/nans in dataset')
    print(dtest.isna().sum())

    classifierPipeline = result_model_pipeline['model_pipeline']

    y_pred = classifierPipeline.predict(dtest)
    predictions['ypred'] = y_pred
    predictions['confusion_matrix'] = skme.confusion_matrix(y_test, y_pred)
    predictions['classification_report'] = skme.classification_report(y_test, y_pred)
    predictions['roc_auc_score'] = skme.roc_auc_score(y_test, y_pred)
    predictions['binned_variables'] = dtest.copy()


    print(predictions['confusion_matrix'])
    print(predictions['classification_report'])
    print(predictions['roc_auc_score'])


    #TODO check class type of classifier.. if type skleanr LR then edited below code
    #TODO get 'classifier' step obj from classifierPipeline and do a class check
    y_pred_probability = classifierPipeline.predict_proba(dtest)    
    predictions['y_pred_probability'] = y_pred_probability

    risk_bands = result_model_pipeline['pd_risk_bands']
    band_outs , bins = pd.cut(y_pred_probability,risk_bands,retbins=True)

    model_risk_band_df = pd.DataFrame([y_pred_probability,band_outs])
    model_risk_band_df = model_risk_band_df.T
    model_risk_band_df.columns = ['Predicted','Risk_Bands']
    model_risk_band_df['Actuals'] = y_test
    model_risk_band_df.columns = ['Predicted','Risk_Bands','Actuals']
    risk_band_range = model_risk_band_df['Risk_Bands'].unique()

    predictions['model_risk_df'] = model_risk_band_df
    predictions['model_risk_bands'] = risk_band_range
    
    #print(predictions['y_pred_probability'].shape)
    #predictions['classification_report'] = skme.classification_report(y_test, y_pred)

    return predictions


## feature_selector_model
max_features = None #If None log2 of total features will be taken
numerical_feat_process = None
numerical_feat_missing = -9999 
category_feat_process = 'woe' #ordinalencoder
category_feat_missing= 'missing' #any sring in place of missing values just needs to be consistent
method = 'RFE' #backward, forward, Lasso


## create_bin_model
variableset = None #Possible list of variables to be used in modelling for subsetting
get_Logit_summary = True #/False
alpha = 0.8 #anynumeric value
penalty = 'l1' #'l2','elastinet', 'none'
param_grid = { 
    'classifier__C' : np.logspace(-3, 1, 5),
    'classifier__penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'classifier__solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'classifier__max_iter' : [100, 1000, 2500, 5000]
    } #None
tune_hyperparams = True #/False
get_summary_report = True #/False

## make_predictions
#test_data # any oos/oot data for getting variance
#result_model_pipeline # output of create_bin_model

## Monotonic Binning - yet to coded as an input function as of now hardcoded. once done will be paramter inputs to create_bin_model.
max_bins = 20 # maximum no. of bins to be used
force_bins = 5 # if no monotonicity then force use 5 bins instead
cardinality_cutoff = 5 # min number of unique values needed for binning
custom_binning = None # dictionary of custom created bins
