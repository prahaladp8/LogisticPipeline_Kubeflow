import numpy as np

# Set of variables the model which the model will consider regardless of previous filtering
variableset = [
'loan_amnt',
 'installment',
 'out_prncp',
 'out_prncp_inv',
 'total_rec_prncp',
 #'recoveries',
 'last_pymnt_amnt',
 'grade'
 ]         

get_Logit_summary = True
alpha = 0

# Each value in the array represents a bin range
custom_bins = {
    'loan_amnt': np.array([ 1000.,  6845., 10015., 15000., 20000., 35000.]),
    'installment': np.array([  25.44 ,  222.28 ,  329.498,  445.214,  614.13 , 1327.45 ]),
    'out_prncp': np.array([    0.   ,  2865.044,  8825.614, 31553.95 ]),
    'out_prncp_inv': np.array([    0.  ,  2860.29,  8817.17, 31531.42]),
    'total_rec_prncp': np.array([    0.   ,  3204.896,  5378.428,  8500.   , 13674.528, 35000.02 ]),
    #'recoveries':np. array([0.e+00, 1.e+00, 9.e+03]),
    'last_pymnt_amnt': np.array([    0.   ,   270.91 ,   432.434,   706.506,  5281.632, 35532.3  ]),    
}

#custom_risk_bands = None
custom_risk_bands = [0 , 0.1 , 0.2 , 0.35 , 0.5, 0.65 , 0.75 , 0.9, 0.95 , 1.0]

numeric_transformations = None  # Self explanatory, accepted values = 'log', 'sqrt'

''' #Example 
    numeric_transformations = {
    'col1': 'log',
    'col2': 'sqrt',
}
'''

#categorical_grouping = None  # Used for grouping categories

categorical_grouping =  {    
    'grade': {
        #'<old-value>' : '<combined new value>'
        'A' : 'Very Good',
        'B' : 'Good',
        'C' : 'Good',
        'D' : 'Average',
        'E' : 'Average',
        'F' : 'Poor',
        'G' : 'Poor'
    }
}

numeric_bining = True 
#Setting the above flag True ensures that even all numeric features are converted to Categorical Bins using WOE Encoding.



#---------------------------------------------------- Variables for Post Mode Reporting ------------------------------------
