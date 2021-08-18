from  base import EDA as eda
import traceback
import argparse
import os
import shutil


def pre_execute():

    parser = argparse.ArgumentParser(description='Reading command line arguments for Pipeline')
    parser.add_argument('--inputpath', type=str, help='Input dir')
    parser.add_argument('--stage', type=str, help='Stage tobe executed')
    parser.add_argument('--outputpath', type=str, help='Output dir each component')
    args = parser.parse_args()

    inputpath = args.inputpath
    stage = args.stage
    outputpath = args.outputpath

    if not os.path.exists(os.getcwd()+'/inputs'):
        os.mkdir(os.getcwd()+'/inputs')

    if not os.path.exists(os.getcwd()+'/outputs'):
        os.mkdir(os.getcwd()+'/outputs')

    #if not os.path.exists(outputpath):
    #    os.mkdir(outputpath)

    
    os.makedirs(os.path.dirname(outputpath), exist_ok=True)
        

    if not os.path.exists(os.getcwd()+'/base/Data'):
        os.mkdir(os.getcwd()+'/base/Data')

    print(outputpath)
    print(inputpath)
    
    print('op is directory : '+str(os.path.isdir(outputpath)))
    print('in is directory : '+str(os.path.isdir(inputpath)))

    print('op is file : '+str(os.path.isfile(outputpath)))
    print('in is file : '+str(os.path.isfile(inputpath)))

    print('op exists : '+ str( os.path.exists(outputpath)))
    print('ip exists : '+ str( os.path.exists(inputpath)))

    print(type(inputpath))
    import pandas as pd    
    df = pd.read_csv(inputpath)
    print(df.shape)

    dest_loc = os.getcwd()+'/inputs'
    for item in os.listdir(inputpath):
        shutil.copy2(inputpath+'/'+item, dest_loc)

    dest_loc = os.getcwd()+'/base/Data'
    for item in os.listdir(inputpath):
        shutil.copy2(inputpath+'/'+item, dest_loc)

    
    #look to accept yaml file also
    
    return {'inputpath': inputpath,'stage': stage,'outputpath':outputpath} 

def alt_pre_execute(inputpath,stage,outputpath):

    if not os.path.exists(os.getcwd()+'/inputs'):
        os.mkdir(os.getcwd()+'/inputs')

    if not os.path.exists(os.getcwd()+'/outputs'):
        os.mkdir(os.getcwd()+'/outputs')

    #if not os.path.exists(outputpath):
    #    os.mkdir(outputpath)

    
    os.makedirs(os.path.dirname(outputpath), exist_ok=True)
        

    if not os.path.exists(os.getcwd()+'/base/Data'):
        os.mkdir(os.getcwd()+'/base/Data')

    print(outputpath)
    print(inputpath)
    
    print('op is directory : '+str(os.path.isdir(outputpath)))
    print('in is directory : '+str(os.path.isdir(inputpath)))

    print('op is file : '+str(os.path.isfile(outputpath)))
    print('in is file : '+str(os.path.isfile(inputpath)))

    print('op exists : '+ str( os.path.exists(outputpath)))
    print('ip exists : '+ str( os.path.exists(inputpath)))

    print(type(inputpath))
    import pandas as pd    
    df = pd.read_csv(inputpath)
    print(df.shape)

    dest_loc = os.getcwd()+'/inputs'
    for item in os.listdir(inputpath):
        shutil.copy2(inputpath+'/'+item, dest_loc)

    dest_loc = os.getcwd()+'/base/Data'
    for item in os.listdir(inputpath):
        shutil.copy2(inputpath+'/'+item, dest_loc)    


def execute(inputpath,stage,outputpath):
    try:        
        alt_pre_execute(inputpath,stage,outputpath)
        lr_pipeline = eda.LogisticPipeline('','','','')
        lr_pipeline.prepare_pipeline()
        if stage == "prepare_data":
            lr_pipeline.prepare_data(os.getcwd()+'/outputs')
        elif stage == "variable_reduction":
            lr_pipeline.reduce_variables(pre_execute_dict['inputpath'],os.getcwd()+'/outputs')
        elif stage == "feature_reduction":
            lr_pipeline.feature_reduction(pre_execute_dict['inputpath'],os.getcwd()+'/outputs')
        elif stage == "model_building":
            lr_pipeline.model(pre_execute_dict['inputpath'],os.getcwd()+'/outputs')
        elif stage == "fine_tuning":
            pass
        
        local_output_loc = os.getcwd()+'/outputs'
        for item in os.listdir(local_output_loc):
           shutil.copy2(local_output_loc+'/'+item, pre_execute_dict['outputpath'])

    except Exception as e:
        print(e)
        print(traceback.format_exc())


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
            lr_pipeline.model(pre_execute_dict['inputpath'],os.getcwd()+'/outputs')
        elif pre_execute_dict['stage'] == "fine_tuning":
            pass
        
        local_output_loc = os.getcwd()+'/outputs'
        for item in os.listdir(local_output_loc):
           shutil.copy2(local_output_loc+'/'+item, pre_execute_dict['outputpath'])

    except Exception as e:
        print(e)
        print(traceback.format_exc())
