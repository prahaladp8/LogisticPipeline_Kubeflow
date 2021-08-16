import numpy as np
import pandas as pd
import random
import datetime
random.seed(144)



#filepath = r'Data/loan_data_2007_2014.csv'#
#data = pd.read_csv(filepath)
#data.drop(['Unnamed: 0'], axis = 1, inplace = True)
#print(data.shape)

#filepath = r'Data/loan_data_2015.csv'
#oos_data = pd.read_csv(filepath)
#oos_data.drop(['Unnamed: 0'], axis = 1, inplace = True)
#print(data.shape)

class ValidationError(Exception):
    pass

def distcheck_classifier(train, test, colstodrop = None, method = 'KNeighborsClassifier'):
    from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV, train_test_split
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, RobustScaler
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    import numpy as np
    import pandas as pd
    import random
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_selection import SelectKBest
    import sklearn.metrics as skme
    random.seed(144)

    train['is_train'] = 1
    test['is_train'] = 0
    data = pd.concat([train,test])
    data = data.sample(frac=1, random_state = 144)
    data.reset_index(drop = True)

    if colstodrop is not None:
        data.drop(colstodrop, axis = 1, inplace = True)

    random.seed(144)

    X = data.drop(['is_train'],axis=1)
    Y = data.is_train.copy()
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = .3, random_state = 144)
    x_train.reset_index(drop = True,inplace = True)
    x_test.reset_index(drop = True,inplace = True)
    y_train.reset_index(drop = True,inplace = True)
    y_test.reset_index(drop = True,inplace = True)

    factors = x_train.shape[1]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    numeric_features = list(data.select_dtypes(include=['int64', 'float64']).columns)
    numeric_features.remove('is_train')
    categorical_features = list(data.select_dtypes(include=['object']).columns)
    #categorical_features.remove('is_train')

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features)
            ,('categorical', categorical_transformer, categorical_features)
            ])

    print(f'Using {method} as classifier.')
    if method == 'GaussianNB':
        binPipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feat', TruncatedSVD(n_components=factors)),
            ('classifier', GaussianNB())])
    elif method=='LinearDiscriminantAnalysis':
        binPipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feat', TruncatedSVD(n_components=factors)),
            ('classifier', LinearDiscriminantAnalysis())])
    elif method=='QuadraticDiscriminantAnalysis':
        binPipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feat', TruncatedSVD(n_components=factors)),
            ('classifier', QuadraticDiscriminantAnalysis())])
    else:
        binPipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feat', TruncatedSVD(n_components=factors)),
            ('classifier', KNeighborsClassifier(3))])
    
    binPipeline.fit(x_train, y_train)
    y_pred = binPipeline.predict(x_test)

    # print(skme.confusion_matrix(y_test, y_pred))
    # print(skme.classification_report(y_test, y_pred))
    # print(skme.roc_auc_score(y_test, y_pred))

    score = skme.roc_auc_score(y_test, y_pred)

    return score

def input_data(projectname, train, target, targettransformlist = None, makebinomial = False, trainbinimbalance = 0.1, oos=None, oot=None, pdv=None, vars_exc=None, distcheck = False, write = False):

    return_slate = {}
    return_slate['projectname'] = projectname

    print(f'Train data check commencing')

    if target not in list(train.columns._data):
        raise ValidationError(f'{target} not present in list of columns in Training Data')

    trainrows = train.shape[0]
    traincols = train.shape[1]
    print(f'Train Shape: {train.shape}')

    if vars_exc is not None:
        train.drop(vars_exc, axis = 1, inplace = True)
        print(f'Train shape post dropping of excluded variables: {train.shape}')

    alltraincols = list(train.columns._data)
    train.dropna(how='all', axis=1, inplace = True)
    nonulltraincols = list(train.columns._data)
    traincols = train.shape[1]

    nullcols = None
    if len(nonulltraincols)<len(alltraincols):
        print(f'Train shape post dropping of null variables: {train.shape}')
        nullcols = list(set(alltraincols) - set(nonulltraincols))

    train.drop_duplicates(inplace=True)
    train.reset_index(drop = True, inplace = True)
    if trainrows > train.shape[0]:
        print(f'Train data duplicate rows removed')
    trainrows = train.shape[0]

    traincolumns = list(train.columns._data)

    if trainrows<=5000:
        print(f'Training data has less than 5000 data points')

    origtargetlist = list(train[target].unique())
    return_slate['All_Target'] = origtargetlist
    if targettransformlist is not None:
        return_slate['Level0_Target'] = targettransformlist
        if makebinomial:
            train['yTarget'] = np.where(train[target].isin(targettransformlist),"Level0","level1")
        else:
            train['yTarget'] = np.where(train[target].isin(targettransformlist),"Level0",train[target])
        return_slate['Level1_Target'] =list(set(origtargetlist) - set(targettransformlist))
    else:
        train['yTarget'] = train[target]
    
    return_slate['Train'] = train.copy()

    if write:
        print(f'Writing Train file as csv')
        return_slate['Train'].to_csv("Data/"+(return_slate['projectname']+'_Train_Data.csv'))

    targetlist = list(train['yTarget'].unique())

    if len(targetlist)<2:
        raise ValidationError(f'Target column has less than 2 categories. Not suitable for Classification.')
    elif len(targetlist)==2:
        targetmethod = 'binomial'
        targetbindist = train['yTarget'].value_counts().reset_index(drop=False)
        targetbindist['bin%'] = targetbindist['yTarget']/sum(targetbindist['yTarget'])
        if  all(i >= trainbinimbalance for i in list(targetbindist['bin%']))==False:
            print(f'Data Imbalance in Binary Target Scenario')
        #train['yTarget'] = np.where(train.target.isin(targettransformlist),"Level0",train.target)

    else:
        targetmethod = 'multinomial'

    return_slate['TargetMethod'] = targetmethod

    print(f'Train data check done')

    if oos is not None:
        print(f'OOS data check commencing')
        if vars_exc is not None:
            oos.drop(vars_exc, axis = 1, inplace = True)
        if nullcols is not None:
            oos.drop(nullcols, axis = 1, inplace = True)
        oos.drop_duplicates(inplace=True)    
        oos.reset_index(drop = True, inplace = True)
        ooscols = set(list(oos.columns._data))
        if ooscols != set(traincolumns):
            raise ValidationError(f'OOS and Train Data have different column set')
        if targettransformlist is not None:
            if makebinomial:
                oos['yTarget'] = np.where(oos[target].isin(targettransformlist),"Level0","level1")
            else:
                oos['yTarget'] = np.where(oos[target].isin(targettransformlist),"Level0",oos[target])
        else:
            oos['yTarget'] = oos[target]
        return_slate['OOS'] = oos.copy()
        #print(return_slate['OOS']['yTarget'])
        if write:
            print(f'Writing OOS file as csv')
            return_slate['OOS'].to_csv(("Data/"+return_slate['projectname']+'_Test_Data.csv'))
        if distcheck:
            print(f'Checking OOS data distribution')
            score = distcheck_classifier(train = train, test = oos, colstodrop = [target,'yTarget'], method = 'KNeighborsClassifier')
            print(f'OOS distribution check score: {score}')
            if score > 0.5:
                print(f'OOS and Train distribtion might be different')
        print(f'OOS data check done')
    else:
        return_slate['OOS'] = None

    if oot is not None:
        print(f'OOT data check commencing')
        if vars_exc is not None:
            oot.drop(vars_exc, axis = 1, inplace = True)
        if nullcols is not None:
            oot.drop(nullcols, axis = 1, inplace = True)
        oot.drop_duplicates(inplace=True)     
        oot.reset_index(drop = True, inplace = True)
        ootcols = set(list(oot.columns._data))
        if ootcols != set(traincolumns):
            print(f'OOT and Train Data have different column set')
        if targettransformlist is not None:
            if makebinomial:
                oot['yTarget'] = np.where(oot[target].isin(targettransformlist),"Level0","level1")
            else:
                oot['yTarget'] = np.where(oot[target].isin(targettransformlist),"Level0",oot[target])
        else:
            oot['yTarget'] = oot[target]
        return_slate['OOT'] = oot.copy()
        if write:
            print(f'Writing OOT file as csv')
            return_slate['OOT'].to_csv(("Data/"+return_slate['projectname']+'_OOT_Data.csv'))
        if distcheck:
            print(f'Checking OOT data distribution')
            score = distcheck_classifier(train = train, test = oot, colstodrop = [target,'yTarget'], method = 'KNeighborsClassifier')
            print(f'OOT distribution check score: {score}')
            if score > 0.5:
                print(f'OOT and Train distribtion might be different')
        print(f'OOT data check done')
    else:
        return_slate['OOT'] = None
    
    if pdv is not None:
        print(f'PDV data check commencing')
        if vars_exc is not None:
            pdv.drop(vars_exc, axis = 1, inplace = True)
        if nullcols is not None:
            pdv.drop(nullcols, axis = 1, inplace = True)
        pdv.drop_duplicates(inplace=True)     
        pdv.reset_index(drop = True, inplace = True)
        pdvcols = set(list(pdv.columns._data))
        if pdvcols != set(traincolumns):
            raise ValidationError(f'PDV and Train Data have different column set')
        if targettransformlist is not None:
            if makebinomial:
                pdv['yTarget'] = np.where(pdv[target].isin(targettransformlist),"Level0","level1")
            else:
                pdv['yTarget'] = np.where(pdv[target].isin(targettransformlist),"Level0",pdv[target])
        else:
            pdv['yTarget'] = pdv[target]
        return_slate['PDV'] = pdv.copy()
        if write:
            print(f'Writing PDV file as csv')
            return_slate['PDV'].to_csv("Data/"+(return_slate['projectname']+'_PDV_Data.csv'))
        if distcheck:
            print(f'Checking PDV data distribution')
            score = distcheck_classifier(train = train, test = pdv, colstodrop = [target,'yTarget'], method = 'KNeighborsClassifier')
            print(f'PDV distribution check score: {score}')
            if score > 0.5:
                print(f'PDV and Train distribtion might be different')
        print(f'PDV data check done')
    else:
        return_slate['PDV'] = None

    return return_slate


# input_slate = input_data(projectname = 'Project_1', train = data.copy(), target = 'loan_status', 
#                          targettransformlist = ['Charged Off','Late (31-120 days)','Default','Does not meet the credit policy. Status:Charged Off'], 
#                          makebinomial = True, trainbinimbalance = 0.1, oos = None, oot=oos_data.copy(), 
#                          pdv=None, vars_exc=None, distcheck = False, write = True)