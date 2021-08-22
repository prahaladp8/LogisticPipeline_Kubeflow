from  base import EDA as eda
import traceback
import argparse
import os
import shutil
import pandas as pd


def pre_execute():

    parser = argparse.ArgumentParser(description='Reading command line arguments for Pipeline')
    parser.add_argument('--inputpath', type=str, help='Input dir')
    parser.add_argument('--stage', type=str, help='Stage to be executed')
    #parser.add_argument('--outputpath', type=str, help='Output dir each component')
    args = parser.parse_args()

    print('***** Preprocessing data for the Pipeline ******')
    inputpath = args.inputpath
    stage = args.stage
    #outputpath = args.outputpath
    print(inputpath)
    print(stage)
    #Temp inputs dir
    if not os.path.exists(os.getcwd()+'/inputs'):
        os.mkdir(os.getcwd()+'/inputs')

    #Temp outputs dir
    if not os.path.exists(os.getcwd()+'/outputs'):
        os.mkdir(os.getcwd()+'/outputs')

    #if not os.path.exists(outputpath):
    #    os.mkdir(outputpath)
    
    #os.makedirs(os.path.dirname(outputpath))
        

    #Actual Data folder
    if not os.path.exists(os.getcwd()+'/base/Data'):
        os.mkdir(os.getcwd()+'/base/Data')

    #print(outputpath)
    print(inputpath)
    
    #print('op is directory : '+str(os.path.isdir(outputpath)))
    print('in is directory : '+str(os.path.isdir(inputpath)))

    #print('op is file : '+str(os.path.isfile(outputpath)))
    print('in is file : '+str(os.path.isfile(inputpath)))

    #print('op exists : '+ str( os.path.exists(outputpath)))
    print('ip exists : '+ str( os.path.exists(inputpath)))
 
    # for items in os.listdir(os.path.realpath(r''+inputpath)):
    #     print(items)

    #dest_loc = os.getcwd()+'/inputs'
    for item in os.listdir(os.path.realpath(r''+inputpath)):
        print(inputpath+'/'+item)
        if item.endswith("PipelineInputs.yml"):
            if os.path.exists(os.getcwd()+"/base/PipelineInputs.yml"):
                os.remove(os.getcwd()+"/base/"+item)
            shutil.copy2(inputpath+'/'+item, os.getcwd()+"/base/" )    
        elif item.endswith("Fine_Tuning_Inputs.py"):
            if os.path.exists(os.getcwd()+"/base/Fine_Tuning_Inputs.py"):
                os.remove(os.getcwd()+"/base/"+item)
            shutil.copy2(inputpath+'/'+item, os.getcwd()+"/base/" )
        elif item.endswith(".csv"):
            shutil.copy2(inputpath+'/'+item, os.getcwd()+'/base/Data')

    for item in os.listdir(os.path.realpath(r''+inputpath+'/Reports')):
        if item.endswith("Pre_Train_Report.xlsx"):
            if os.path.exists(os.getcwd()+"/base/Reports/Pre_Train_Report.xlsx.py"):
                os.remove(os.getcwd()+"/base/"+item)
            shutil.copy2(inputpath+'/'+item, os.getcwd()+"/base/")

        #shutil.copy2(inputpath+'/'+item, dest_loc)

    # dest_loc = os.getcwd()+'/base/Data'
    # for item in os.listdir(os.path.realpath(r''+inputpath)):
    #     shutil.copy2(inputpath+'/'+item, dest_loc)

    print('***** End of Preprocessing data for the Pipeline ******')
    #look to accept yaml file also
    
    return {'inputpath': inputpath,'stage': stage #,'outputpath':outputpath
    } 


if __name__ == '__main__':
    try:
        pre_execute_dict = pre_execute()                       
        lr_pipeline = eda.LogisticPipeline('','','','')
        lr_pipeline.prepare_pipeline()
        if pre_execute_dict['stage'] == "prepare_data":
            lr_pipeline.prepare_data(pre_execute_dict['inputpath'])
        elif pre_execute_dict['stage'] == "variable_reduction":
            lr_pipeline.reduce_variables(pre_execute_dict['inputpath'])
        elif pre_execute_dict['stage'] == "feature_selection":
            lr_pipeline.feature_selection(pre_execute_dict['inputpath'])
        elif pre_execute_dict['stage'] == "model_building":
            lr_pipeline.model_building(pre_execute_dict['inputpath'])
        elif pre_execute_dict['stage'] == "fine_tuning":
            lr_pipeline.model_fine_tuning(pre_execute_dict['inputpath'])
        
        reports_dir = os.getcwd()+"/base/Reports/Post_Train_Model_Report.xlsx"
        if os.path.exists(reports_dir):
            shutil.copy2(reports_dir, pre_execute_dict['inputpath'])

    except Exception as e:
        print(e)
        print(traceback.format_exc())
'''   
    --inputpath = pvc location for the 1st stage

'''