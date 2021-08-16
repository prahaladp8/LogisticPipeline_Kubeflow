DROP_MISSING_THRESHOLD = 0.6 
BAD_OUTCOME = 'level1'
IV_THRESHOLD = 0.02
TARGET_VARIABLE = 'yTarget'
TARGET_VARIABLE_GOOD_OUTCOME = 'Level0'
TARGET_VARIABLE_BAD_OUTCOME = 'level1'
CORRELATION_THRESHOLD = 0.5
REPLACE_MISSING_WITH = 'mean'
VIF_THRESHOLD = 10
PSI_BUCKET_COUNT = 10
CSI_THRESHOLD = 0.1
EMPTY_CATEGORICAL_VALUE_REPLACEMENT = 'missing'
EMPTY_NUMERIC_VALUE_REPLACEMENT = -999999
FEATURE_SELECTION_METHOD = 'Lasso'
#Variables Internal to the Pipeline

pipeline_mandatory_keys = ['set_execution_steps','target_variable','target_variable_bad_values','training_dataset']
pipeline_input_file_path = "PipelineInputs.yml"

class ExecutionStepsKey():
    univariate = 'univariate'
    bivariate = 'bivariate'
    multivariate = 'multivariate'
    model_building = 'model_building'
    feature_reduction = 'feature_reduction'
    fine_tuning = 'fine_tuning'

class DefaultInfo():
    default_model_path = 'Models'
    default_log_path = 'Logs'
    default_report_path = 'Reports'
    default_staging_location = 'temp'
    default_train_test_split_ratio = 0.25
    default_converted_target_variable_name = 'yTarget'