from openicl import PromptTemplate
from openicl import DatasetReader
from openicl import PPLInferencer, AccEvaluator
from datasets import load_dataset, concatenate_datasets
from accelerate import Accelerator
import os
from utils import templates, input_columns, output_columns, test_split, score_mat_2_rank_mat

# set the model and dataset path
dir1 = 'result/'
dataset_path = 'data/'
 
task_names = ["cms"]
model_names = ['meta-llama/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct']
seeds = [1]

for task_name in task_names:
    print('\n#######################################################################  task: {}  #######################################################################'.format(task_name))

    for model_name in model_names:
        print('\n#########################  model: {}  #########################\n'.format(model_name))

        # load dataset
        train_path = dataset_path + task_name + '/train.jsonl'
        test_name = test_split[task_name]
        test_path = dataset_path + task_name + '/' + test_name + '.jsonl'
        # test_path = dataset_path + task_name + '/train.jsonl'

        combined_dataset = load_dataset("json", data_files={"train": train_path, "test": test_path})

        train_dataset = combined_dataset["train"]
        test_dataset = combined_dataset["test"]

        # Construct the DatasetReader
        data = DatasetReader(combined_dataset, input_columns=input_columns[task_name], output_column=output_columns[task_name])
    
        prediction_dir = dir1 + model_name + '/' + task_name
        if os.path.exists(prediction_dir) and os.path.isdir(prediction_dir):
        #     print('\n#########################  task_name: {}  #########################\n'.format(task_name))

            file_checked = []
            for k in [8]:
                best = 0
                best_file = None
                for file_name in sorted(os.listdir(prediction_dir)):
                    if 'process' in file_name:
                        continue
                    
                    if str(k)+'_shot' in file_name:
                        prediction_path = os.path.join(prediction_dir, file_name)
                
                        import json
                        with open(prediction_path, 'r') as f:
                            try:
                                data1 = json.load(f)
                            except:
                                print("An exception occurred: ", prediction_path)

                        num = 0
                        predictions = []
                        for i in range(len(data1)):
                            predictions.append(data1[str(i)]['prediction'])
                        
                        for i in range(len(data.references)):
                            if data.references[i] == predictions[i]:
                                num += 1.0                    
                        # if task_name in ["cms"]:
                        score = num / len(data.references)
                        print('{} ### score: {}'.format(file_name, score))

                        if score > best:
                            best = score
                            best_file = file_name
                        file_checked.append(file_name)
                    
                if best_file is not None:
                    print("### k: {}, score: {:.5f}, file: {} \n".format(k, best, best_file))
                else:
                    print("### k: {}, ##### \n".format(k))