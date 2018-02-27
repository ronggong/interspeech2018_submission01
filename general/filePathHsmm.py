from filePathShared import *


# acoustic model training dataset path
# change this to your folder
training_data_hsmm_path = '/Users/gong/Documents/MTG document/dataset/acousticModels'

primarySchool_wav_path = join(primarySchool_dataset_root_path, 'wav')
primarySchool_textgrid_path = join(primarySchool_dataset_root_path, 'textgrid')

cnn_file_name = 'hsmm_am_timbral'
eval_results_path = join(root_path, 'eval', 'results', 'hsmm', cnn_file_name)

primarySchool_results_path = join(root_path, 'eval', 'joint', 'results')

kerasScaler_path = join(root_path, 'cnnModels', 'hsmm', 'scaler_'+cnn_file_name+'.pkl')
kerasModels_path = join(root_path, 'cnnModels', 'hsmm', cnn_file_name)

nacta2017_wav_path = join(nacta2017_dataset_root_path, 'wav')
nacta2017_textgrid_path = join(nacta2017_dataset_root_path, 'textgridDetails')

nacta_wav_path = join(nacta_dataset_root_path, 'wav')
nacta_textgrid_path = join(nacta_dataset_root_path, 'textgrid')
