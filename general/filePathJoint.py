from filePathShared import *


# path where to store the files for training the model
# change this to your folder
training_data_joint_path = '/Users/gong/Documents/MTG document/dataset/syllableSeg/'

primarySchool_wav_path = join(primarySchool_dataset_root_path, 'wav')
primarySchool_textgrid_path = join(primarySchool_dataset_root_path, 'textgrid')

joint_cnn_model_path = join(root_path, 'cnnModels', 'joint')

scaler_joint_model_path = join(joint_cnn_model_path, 'scaler_joint.pkl')

# results path for outputing the metrics
cnnModel_name = 'jan_joint'
eval_results_path = join(root_path, 'eval', 'results', 'joint', cnnModel_name)

# results path to save the .pkl for the evaluation
primarySchool_results_path = join(root_path, 'eval', 'joint', 'results')

full_path_keras_cnn_0 = join(joint_cnn_model_path, cnnModel_name)

nacta_wav_path = join(nacta_dataset_root_path, 'wav')
nacta_textgrid_path = join(nacta_dataset_root_path, 'textgrid')

nacta2017_wav_path = join(nacta2017_dataset_root_path, 'wav')
nacta2017_textgrid_path = join(nacta2017_dataset_root_path, 'textgridDetails')