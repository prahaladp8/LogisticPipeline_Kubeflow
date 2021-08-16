from  base import EDA as eda
import traceback
import argparse
import os
import shutil


def pre_execute():

    parser = argparse.ArgumentParser(description='Reading command line arguments for Pipeline')
    parser.add_argument('--inputpath', type=str, help='Input dir')
    parser.add_argument('--stage', type=str, help='Output dir each component')
    args = parser.parse_args()

    inputpath = args.inputpath
    stage = args.stage

    if not os.path.exists(os.getcwd()+'/inputs'):
        os.mkdir(os.getcwd()+'/inputs')

    if not os.path.exists(os.getcwd()+'/outputs'):
        os.mkdir(os.getcwd()+'/outputs')

    if not os.path.exists(os.getcwd()+'/base/Data'):
        os.mkdir(os.getcwd()+'/base/Data')

    dest_loc = os.getcwd()+'/inputs'
    for item in os.listdir(inputpath):
        shutil.copy2(inputpath+'/'+item, dest_loc)

    dest_loc = os.getcwd()+'/base/Data'
    for item in os.listdir(inputpath):
        shutil.copy2(inputpath+'/'+item, dest_loc)
    
    return {'inputpath': inputpath,'stage': stage} 

if __name__ == '__main__':
    try:
        pre_execute_dict = pre_execute()                       
        lr_pipeline = eda.LogisticPipeline('','','','')
        lr_pipeline.prepare_pipeline()
        if pre_execute_dict['stage'] == "prepare_data":
            lr_pipeline.prepare_data(os.getcwd()+'/outputs')
        elif pre_execute_dict['stage'] == "variable_reduction":
            lr_pipeline.reduce_variables(pre_execute_dict['inputpath'],os.getcwd()+'/outputs')
        elif pre_execute_dict['stage'] == "feature_reduction":
            lr_pipeline.feature_reduction(pre_execute_dict['inputpath'],os.getcwd()+'/outputs')
        elif pre_execute_dict['stage'] == "model_building":
            pass
        elif pre_execute_dict['stage'] == "fine_tuning":
            pass

    except Exception as e:
        print(e)
        print(traceback.format_exc())
