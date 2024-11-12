# Operation Process

The **real** floder is a deep learning model  which based on the 2s-AGCN and the **HDBN** floder is another deep learning model based on Mix_former improvement .

- Generate a skeleton dataset using the original joint dataset and extract 2d pose from the test dataset
- Train models:

	* **MIXCTR_GCN**: Load the MIX_CTR_GCN configuration file and train on the specified GPU device.
	* **Mix_Former:** Train **joint** and **bone** models separately, load the corresponding configuration files and train.
- Test models:
	* **MIXCTR_GCN**: Load the test configuration files for joints and bones separately and conduct testing.
	* **Mix_Former**: Test the model separately using joint and bone data, and save the test results.
- Ensemble：
	* Use simulated annealing algorithm as optimization method to gradually find the optimal weight combination
	* Load the prediction scores of multiple models and apply different weighted combinations to the prediction scores of each model




# Process_data

First, we need to generate bone_dataset from origin joint dataset. 

```cmd
python gen_modal.py --modal bone --use_mp True
```
and 2d_pose dataset is also used.
(in HDBN)

```cmd
python temp.py
cd ./Process_data
python extract_2d_pose.py
```
**change** different parameter in order to generate three_views_datasets



# Training dataset



## MS_CTR_GCN

Enter the directory of the MS_CTR_GCN model and run the training script. This command loads the configuration file MS_CTR_GCN.config.yaml and trains on the specified device ( GPU device 0).

```cmd
cd ./MS-CTR-GCN-master
python main.py --config ./config/uav/joint.yaml --device 0

python main.py --config ./config/uav/bone.yaml --device 0

python main.py --config ./config/uav/motion.yaml --device 0

python main.py --config ./config/uav/longtail.yaml --device 0
```



## TD_GCN
```cmd
cd ./TD_GCN
python main.py --config ./config/tdgcn_V1_J.yaml --device 0
```





## Mix_Former

Similarly, enter the directory of the Mix_Former model and run the training scrips to train the joint and bone models respectively.

**Mix** dosen't mean mix formers, only one type of former is used, mix means mix different views.

```cmd
cd ./Three_Views_Former/Mix_Former
python main.py --config ./config/mixformer_V1_J_1.yaml --device 0

python main.py --config ./config/mixformer_V1_J_2.yaml --device 0

python main.py --config ./config/mixformer_V1_J_3.yaml --device 0

python main.py --config ./config/mixformer_V1_B_1.yaml --device 0

python main.py --config ./config/mixformer_V1_B_2.yaml --device 0

python main.py --config ./config/mixformer_V1_B_3.yaml --device 0
```



# Model test



## MS_CTR_GCN

Run the following command to load the corresponding test configuration file to test the Mix_GCN model using joint and bone data and output the test results.

```cmd
cd ./MS-CTR-GCN
python main.py --config ./config/test_joint.yaml

python main.py --config ./config/test_bone.yaml

python main.py --config ./config/test_motion.yaml

python main.py --config ./config/test_longtail.yaml
```

## TD_GCN
```cmd
cd ./TD_GCN
python main.py --config ./config/test_joint.yaml
```

## Mix_Former

Similarly, for the Mix_Former model, run the following commands to test using joint and bone data separately and save the test scores.


```cmd
cd ./Three_Views_Former/Mix_Former
python main.py --config ./config/mixformer_V1_J_1.yaml --phase test --save-score True --weights ./output/skmixf_V1_J_1/runs-56-7280.pt --device 0 

python main.py --config ./config/mixformer_V1_J_2.yaml --phase test --save-score True --weights ./output/skmixf_V1_J_2/runs-56-7280.pt --device 0 

python main.py --config ./config/mixformer_V1_J_3.yaml --phase test --save-score True --weights ./output/skmixf_V1_J_3/runs-56-7280.pt --device 0 

python main.py --config ./config/mixformer_V1_B_1.yaml --phase test --save-score True --weights ./output/skmixf_V1_B_1/runs-68-8840.pt --device 0 

python main.py --config ./config/mixformer_V1_B_2.yaml --phase test --save-score True --weights ./output/skmixf_V1_B_2/runs-68-8840.pt --device 0 

python main.py --config ./config/mixformer_V1_B_3.yaml --phase test --save-score True --weights ./output/skmixf_V1_B_3/runs-70-9100.pt --device 0 
```

**for time reason** only one axis former_joint was training, axis in 2 performs much much better than other axis, so provide skmixf_V1_J in output file, actually equals to skmixf_V1_J_2.



# Ensemble

We generate each model's weight depending on valset
```cmd
python ensemble2B.py
```
