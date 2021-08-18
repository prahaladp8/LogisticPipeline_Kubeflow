import kfp
from kfp import components
from kfp.components import InputPath, OutputPath
import os
import kfp.dsl as dsl

# def prepare_data(inputpath: InputPath('str'), stage : str, outputpath: OutputPath('str')):
#     return dsl.ContainerOp(
#         name = 'Prepare Data', 
#         image = 'prahaladp8/lrbase', 
#         command = ['python3', 'Driver.py'],
#         arguments=[
#             '--inputpath', inputpath,
# 			'--stage', stage
#         ],
#         file_outputs={
#             'outputpath': '/univariate.pkl',
#         }
#     )

def prep_data_new(inputpath:InputPath('str'),outputpath:OutputPath('str')):
	import pandas as pd
	import os	
	print('input path'+inputpath)
	print('output path'+outputpath)

	print(os.getcwd())

	#for item in os.listdir(os.getcwd()):
	#	print(item)
	
	if not os.path.exists(outputpath):
		os.mkdir(outputpath)
	
	df1 = pd.DataFrame({"x":[0,1,2,3]})
	
	# fullname = os.path.join(outputpath, 'out.csv')  
	# df1.to_csv(fullname)	
	fullname = os.path.join(outputpath, 'out.pkl')  
	df1.to_pickle(fullname)
	df1.to_pickle(outputpath+"/out2.pkl")
	df1.to_csv(outputpath+"/out4.csv")
	print(df1.shape)

def t3(inputpath:InputPath('str'),outputpath:OutputPath('str')):
	import pandas as pd 
	import os	
	from Driver import execute
	print(inputpath)

	print(os.path.isdir(inputpath))	

	execute(inputpath,'prepare_data',outputpath)

def t2(inputpath:InputPath('str'),outputpath:OutputPath('str')):
	import pandas as pd 
	import os	
	print(inputpath)

	print(os.path.isdir(inputpath))

	if os.path.isdir(inputpath):
		for items in os.listdir(inputpath):
			print(items)

	t2 = pd.read_pickle(inputpath+"/out.pkl")
	print(t2)
	print(t2.shape)

	if not os.path.exists(outputpath):
		os.mkdir(outputpath)

	t2.to_pickle(outputpath+"/out2.pkl")
	t2.to_pickle(outputpath+"/out3.pkl")
	t2.to_csv(outputpath+"/out4.csv")

test3_op = components.create_component_from_func(
		func=prep_data_new,
		base_image='prahaladp8/lrbase',
		output_component_file='sample1.yaml',		
	)	

test4_op = components.create_component_from_func(
		func=t2,
		base_image='prahaladp8/lrbase',		
		output_component_file='sample2.yaml',
	)	

test5_op = components.create_component_from_func(
		func=t3,
		base_image='prahaladp8/lrbase',		
		output_component_file='sample3.yaml',
	)	


#prepare_data_op = components.load_component_from_file(os.getcwd()+'/components/prepare_data.yaml')
#variable_reduction_op = components.load_component_from_file(os.getcwd()+'/components/variable_reduction.yaml')
#feature_selection_op = components.load_component_from_file('components/feature_selection.yaml')
#model_building_op = components.load_component_from_file('components/model_building.yaml')
#fine_tuning_op = components.load_component_from_file('components/fine_tuning.yaml')

@kfp.dsl.pipeline(name='LogisticPipeline',description='An example pipeline that performs addition calculations.')
def LogisticPipeline(inputpath:InputPath('str'),outputpath:OutputPath('str')):	
	# if os.path.isdir(inputpath):
	# 	for items in os.listdir(inputpath):
	# 		print(items)
	#files_task = test3_op(inputpath)
	#t2_func  = test4_op(files_task.output)
	temp = test5_op(inputpath)
	#prep_output = prepare_data_op(inputpath,'prepare_data')
	#variable_reduction = variable_reduction_op(prep_output.outputs['output_path'],'variable_reduction')

if __name__ == '__main__':
	kfp.compiler.Compiler().compile(pipeline_func=LogisticPipeline,package_path='LogisticPipeline-temp.yaml')




	# 1. pre load the dataset
	# 2. download the dataset
	# 3. shared volumne
	# 4. gcloud / other storage