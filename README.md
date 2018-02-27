# INTERSPEECH 2018 submission 01
Singing voice phoneme segmentation by jointly learning syllable and phoneme onset positions.

The code in the repository is for the conference review process.

## First thing to do
* Use python 2.7.* I haven't test the code on python3  
* Install the requirements.txt

## Download the 3 jingju a cappella singing voice datasets
[part 1](https://doi.org/10.5281/zenodo.780559)  
[part 2](https://doi.org/10.5281/zenodo.842229)  
[part 3](https://doi.org/10.5281/zenodo.1185123)  
If you only want to reproduce the experiment results in the paper, 
you only need to download the part 3 because the part 1 and 2 are used
for training the models.

## Set the paths
Once datasets are downloaded, you need to set the paths to let the 
program knows where are they. 

What you need to set in `./general/filePathShared.py` are:
* Set `path_jingju_dataset` to the parent path of these three datasets.
* Set `primarySchool_dataset_root_path` to the path of the interspeech2018 dataset (the current dataset).
* Set `nacta_dataset_root_path` to the path of the jingju dataset part1.
* Set `nacta2017_dataset_root_path` to the path the jingju dataset part2.

And in both `./general/filePathHsmm.py` and `./general/filePathJoint.py`:
* Set `training_data_joint_path` to where putting the training features, labels
for the proposed joint model.
* Set `training_data_hsmm_path` to where putting these files for the baseline
HSMM emission model.

## How to use pre-trained models to reproduce the results?
As you may see, there is a _cnnModels_ folder in the repo, where we store all
the pre-trained models. To use these models, you should run the following scripts:
* `proposed_method_pipeline.py` will calculate the syllable and phoneme onset
results using the proposed method, then save them to `./eval/results/joint/`.
* `baseline_forced_alignment.py` will calculate those results using the baseline
 HSMM forced alignment, then save them to `./eval/results/hsmm/`.

For each model, we have trained five times, to get the mean and std statistics, you
need to run `eval_stats.py`. The final results will be put in `./eval/hsmm/` or 
`./eval/joint/` folds.
* _*phoneme_onset_all.txt_: phoneme onset detection results
* _*phoneme_segment_all.txt_: phoneme segmentation results
* _*syllable_onset_all.txt_: syllable onset detection results
* _*syllalbe_segment_all.txt_: syllable segmentation results

There are two columns in each result file, the 1st column is the mean, the 2nd
is the std. For onset detection results, the 3rd row is the f1-measure
without considering the label and the tolerance is 0.025s.

## How to get the features, labels and samples weights?
Make sure that you have downloaded all three datasets and set the `training_data_joint_path`
and `training_data_hsmm_path`. In `./training_feature_collection` folder, you can:
* Run `training_sample_collection_joint.py` for the proposed method
* Run `training_sample_collection_hsmm.py` for the baseline method

The training materials will be stored in the paths you have set.

## How to train the models?
We have provided the training scripts. You can find them in `./model_training/train_scripts` folder.
Before running them, you need change the necessary paths to direct to the training materials
which you obtained in the previous step.

## Questions?
Feel free to open an issue or email me: rong.gong\<at\>upf.edu
