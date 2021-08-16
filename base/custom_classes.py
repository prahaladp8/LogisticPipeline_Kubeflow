class ValidationError(Exception):
    pass

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.stats.stats as stats
import pandas.core.algorithms as algos
#from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array

pd.options.mode.chained_assignment = None

class MonotonicBinning(BaseEstimator, TransformerMixin):
    
    """Monotonically bin numeric variables based on target. The binning operation starts 
    with the "max_bins" option. It iterates by reducing the number of bins, until it finds bins 
    with monotonic relationship (either increasing or decreasing) between X and y. 
    If the module is unable to find a monotonic relationship, it forcefully creates bins 
    using the "force_bins" option.    
    
    Parameters
    ----------
    feature_names: 'all' or list (default='all')
        list of features to perform monotonic binning operation. 
        - 'all' (default): All features in the dataset will be used
        - list of features: ['age', 'income',......]
    
    max_bins: int (default=20)
        Maximum number of bins that can be created for any given variable. The final number of bins 
        created will be less than or equal to this number.
        
    force_bins: int (default=3)
        It forces the module to create bins for a variable, when it cannot find monotonic relationship 
        using "max_bins" option. The final number of bins created will be equal to the number specified.
        
    cardinality_cutoff: int (default=5)
        Cutoff to determine if a variable is eligible for monotonic binning operation. Any variable 
        which has unique levels less than this number will be treated as character variables. 
        At this point no binning operation will be performed on the variable and it will return 
        the unique levels as bins for these variable.
    
    prefix: string (default=None)
        Variable prefix to be used for the column created by monotonic binning. 
        
    custom_binning: dict (default=None)
        Dictionary structure - {'feature_name': float list}
        Example - {'age': [0., 1., 2., 3.]}
        Using this parameter, the user can perform custom binning on variables. 
        This parameter is also used to apply previously computed bins for each feature (Score new data). 
    """
    
    # Initialize the parameters for the function
    def __init__(self, feature_names='all', max_bins=20, force_bins=3, monotone_cutoff=0.001,
                 cardinality_cutoff=5, prefix=None, custom_binning=None):
        self.feature_names = feature_names
        self.max_bins = max_bins
        self.force_bins = force_bins + 1 #to make the total number of bins as specified by user
        self.cardinality_cutoff = cardinality_cutoff
        self.prefix = prefix
        self.custom_binning = custom_binning
        self.monotone_cutoff = monotone_cutoff
    
    # check input data type - Only Pandas Dataframe allowed
    def check_datatype(self, X):
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("The input data must be pandas dataframe. But the input provided is " + str(type(X)))
        return self
        
    # the fit function for monotonic binning
    def fit(self, X, y):
        
        #if the function is used as part of pipeline, then try to unpack tuple values 
        #produced in the previous step. Added as a part of pipeline feature. 
        try:
            X, y = X
        except:
            pass
        
        #check datatype of X
        self.check_datatype(X)
        
        #The length of X and Y should be equal
        if X.shape[0] != y.shape[0]:
            raise ValueError("Mismatch in input lengths. Length of X is " + str(X.shape[0]) + " \
                            but length of y is " + str(y.shape[0]) + ".")
        
        # The label must be binary with values {0,1}
        unique = np.unique(y)
        if len(unique) != 2:
            raise ValueError("The target column y must be binary. But the target contains " + str(len(unique)) \
                             + " unique value(s).")
        
        #identify the variables to bin and assign the bin mapping dictionary
        self.bins = {} #bin mapping
        custom_features = [] 
        numerical_features = list(X._get_numeric_data().columns)
        to_bin = numerical_features.copy()
        
        #if custom binning is provided by the user, use that instead of monotonic binning operation
        if self.custom_binning:
            self.bins = self.custom_binning
            custom_features = list(self.bins.keys())
            to_bin = list(set(numerical_features) - set(custom_features))
        
        #Identifying the final features to fit the binning operation
        if self.feature_names == 'all':
            fit_on_features = to_bin
            self.transform_features = numerical_features
        else:
            fit_on_features = list(set(self.feature_names) - set(custom_features))
            self.transform_features = self.feature_names
        
        temp_X = X[fit_on_features] #subset data only on features to fit
        
        #check if the dataframe is numeric. If any categorical data is present, raise errors
        try:
            fit_X = pd.DataFrame(check_array(temp_X, accept_sparse=True, force_all_finite=False, \
                                             copy=True), columns=fit_on_features)
        except:
            raise ValueError("The input feature(s) should be numeric type. Some of the input features \
                            has character values in it. Please use a encoder before performing monotonic operations.")
        
        #apply the monotonic train function on dataset
        fit_X.apply(lambda x: self.train(x, y), axis=0)
        return self
    
    #check the cardinality of each column present in input data. Input data X is a Pandas Series type. 
    #If the cardinality is less than user provided input, then the input data will be processed as is. 
    #No fit will be performed on the data. If the cardinality is greater, fit will be performed. 
    def check_cardinality(self, X):
        
        if len(pd.Series.unique(X)) > self.cardinality_cutoff: 
            fitted = True #Fit will be performed by the function
        else:
            fitted = False #No fit will be performed. Return X as is.

        return fitted
    
    #monotonic binning - The function is applied on each columns identified in the fit function. 
    #Here, the input X is a Pandas Series type.
    def train(self, X, y):
        
        fitted = self.check_cardinality(X) #check the unique value and evaluate if fit is needed or not
        mapping = {} #dictionary mapping for the current feature
        
        #Create the bins for each numeric variables
        if fitted: #fit is required
            r = 0
            max_bins = self.max_bins
            force_bins = self.force_bins
            
            """Calculate spearman correlation for the distribution identified. If the distribution is not monotonic, 
            reduce bins and reiterate. Proceed until either one of the following happens,
            a) Monotonic relationship is identified between a feature X and y is identified (np.abs(r) =1)
            b) max_bins = 0, in which case the code could not identify a monotonic relationship
            """
            while ((1-np.abs(r)) > self.monotone_cutoff) and (max_bins > 0):
                try:
                    ser, bins = pd.qcut(X, max_bins, retbins=True)
                    bins_X = pd.DataFrame({"X": X, "Y": y, "Bins": ser})
                    bins_X_grouped = bins_X.groupby('Bins', as_index=True)
                    r, p = stats.spearmanr(bins_X_grouped.mean().X, bins_X_grouped.mean().y) #spearman operation
                    max_bins = max_bins - 1 
                except Exception as e:
                    max_bins = max_bins - 1
            
            """
            Execute this block when monotonic relationship is not identified by spearman technique. 
            We still want our code to produce bins.
            """
            if len(bins_X_grouped) == 1:
                bins = algos.quantile(X, np.linspace(0, 1, force_bins)) #creates a new binnning based on forced bins
                if len(np.unique(bins)) == 2:
                    bins = np.insert(bins, 0, 1)
                    bins[1] = bins[1]-(bins[1]/2)
                bins = np.sort(np.unique(bins))
        else: #no fit is required
            bins = np.sort(pd.Series.unique(X))
        
        # map the bins corresponding input feature
        mapping[str(X.name)] = bins
        self.bins.update(mapping)
        
        return self
        
    #Transform new data or existing data based on the fit identified or custom transformation provided by user
    def transform(self, X, y=None):
        
        #if the function is used as part of pipeline, then try to unpack tuple values produced in the 
        #previous step. Added as a part of pipeline feature. 
        try:
            X, y = X
        except:
            pass
        
        self.check_datatype(X) #check input datatype. 
        outX = X.copy(deep=True) 
        
        #identify the features on which the transformation should be performed
        try:
            #check_is_fitted(self, 'transform_features')
            if self.transform_features:
                transform_features = self.transform_features
        except:
            if self.custom_binning:
                transform_features = list(self.custom_binning.keys())
            else:
                raise ValueError("Estimator has to be fitted to make monotonic transformations")
        
        #final list of features to be transformed
        transform_features = list(set(transform_features) & set(outX.columns)) 
        
        #raise error if the list is empty
        if not transform_features:
            raise ValueError("Empty list for monotonic transformation. \
                            Estimator has to be fitted to make monotonic transformations")
        
        #iterate through the dataframe and apply the bins
        for i in transform_features:
            
            tempX = outX[i] #pandas Series
            original_column_name = str(i)
            
            #create the column name based on user provided prefix
            if self.prefix:
                new_column_name = str(self.prefix) + '_' + str(i)
            else:
                new_column_name = original_column_name
            
            #use the custom bins provided by user, wherever possible
            if self.custom_binning:
                if original_column_name in self.custom_binning:
                    try:
                        self.bins[original_column_name] = self.custom_binning[original_column_name]
                    except:
                        self.bins = self.custom_binning
            
            #check if the bin mapping is present 
            #check_is_fitted(self, 'bins')
            if not self.bins:
                raise ValueError("Bin variable is not present. \
                                Estimator has to be fitted to apply monotonic transformations.")
            
            #input data cardinality check
            fitted = self.check_cardinality(tempX)
            
            #determine whether to apply bins or not
            if fitted == True: #apply bins and return
                apply_bins = self.bins[original_column_name]
                outX[new_column_name] = pd.cut(tempX, apply_bins, include_lowest=True)
            else: # no binning required
                outX[new_column_name] = tempX
            
        #transformed dataframe 
        return outX
    
    #Method that describes what we need this transformer to do
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
    
from statsmodels.discrete.discrete_model import Logit
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.estimator_checks import check_estimator

class LogitWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept = True, start_params=None, method='l1', maxiter='defined_by_method', full_output=1, disp=1, callback=None, alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=0.0001, qc_tol=0.03):

        self.fit_intercept = fit_intercept
        self.model_ = None
        self.results_ = None
        self.start_params = start_params
        self.method = method
        self.maxiter = maxiter
        self.full_output = full_output
        self.disp = disp
        self.callback = callback
        self.alpha = alpha
        self.trim_mode = trim_mode
        self.auto_trim_tol = auto_trim_tol
        self.size_trim_tol = size_trim_tol
        self.qc_tol = qc_tol

    """
    Parameters
    ------------
    column_names: list
            It is an optional value, such that this class knows 
            what is the name of the feature to associate to 
            each column of X. This is useful if you use the method
            summary(), so that it can show the feature name for each
            coefficient
    """ 

    def fit(self, X, y, column_names=()):

        if self.fit_intercept:
            X = sm.add_constant(X,has_constant='add')
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        if len(column_names) != 0:
            cols = column_names.copy()
            cols = list(cols)
            X = pd.DataFrame(X)
            cols = column_names.copy()
            cols.insert(0,'intercept')
            print('X ', X)
            X.columns = cols
        
        self.X_ = X
        self.y_ = y
        self.model_ = Logit(y, X)
        self.results_ = self.model_.fit_regularized(start_params=self.start_params, method=self.method, maxiter=self.maxiter, full_output=self.full_output, disp=self.disp, callback=self.callback, alpha=self.alpha, trim_mode=self.trim_mode, auto_trim_tol=self.auto_trim_tol, size_trim_tol=self.size_trim_tol, qc_tol=self.qc_tol)
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, 'model_')

        # Input validation
        X = check_array(X)

        if self.fit_intercept:
            X = sm.add_constant(X)
        
        return list(map(round, self.results_.predict(X)))

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self, 'model_')

        # Input validation
        X = check_array(X)

        if self.fit_intercept:
            X = sm.add_constant(X)
        
        return self.results_.predict(X)
    
    def get_intercept(self, deep = False):
        return {'fit_intercept':self.fit_intercept}

    def summary(self):
        return self.results_.summary2()