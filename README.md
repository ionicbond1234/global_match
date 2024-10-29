# Operation Process

The **real** floder is a deep learning model  which based on the 2s-AGCN and the **HDBN** floder is another deep learning model based on Mix_former improvement .

- Generate a skeleton dataset using the original joint dataset and extract 2d pose from the test dataset
- Train models:

	* **Mix_GCN**: Load the Mix_GCN configuration file and train on the specified GPU device.
	* **Mix_Former:** Train **joint** and **bone** models separately, load the corresponding configuration files and train.
- Test models:
	* **Mix_GCN**: Load the test configuration files for joints and bones separately and conduct testing.
	* **Mix_Former**: Test the model separately using joint and bone data, and save the test results.
- Ensemble：
	* Use simulated annealing algorithm as optimization method to gradually find the optimal weight combination
	* Load the prediction scores of multiple models and apply different weighted combinations to the prediction scores of each model




# Process_data

First, we need to generate bone_dataset from origin joint dataset. 

(in 2s_AGCN)

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


# Training dataset

## Mix_GCN

Enter the directory of the Mix_GCN model and run the training script. This command loads the configuration file mix_GCN.config.yaml and trains on the specified device ( GPU device 0).

```cmd
cd ./Model_inference/Mix_GCN
python main.py --config ./config/tdgcn_V2_J.yaml --device 0
python main.py --config ./config/tdgcn_V2_B.yaml --device 0
python main.py --config ./config/mstgcn_V2_J.yaml --device 0
python main.py --config ./config/mstgcn_V2_B.yaml --device 0
python main.py --config ./config/ctrgcn_V2_J.yaml --device 0
python main.py --config ./config/ctrgcn_V2_B.yaml --device 0
python main.py --config ./config/train_joint.yaml --device 0
```

## Mix_Former

Similarly, enter the directory of the Mix_Former model and run the training scrips to train the joint and bone models respectively.

```cmd
cd ./Model_inference/Mix_Former
python main.py --config ./config/mixformer_V2_J.yaml --device 0
python main.py --config ./config/mixformer_V2_B.yaml --device 0
```

# Model test

## Mix_GCN

Run the following command to load the corresponding test configuration file to test the Mix_GCN model using joint and bone data and output the test results.

```cmd
python main.py --config ./config/test_joint.yaml
python main.py --config ./config/test_bone.yaml
```

## Mix_Former

Similarly, for the Mix_Former model, run the following commands to test using joint and bone data separately and save the test scores.

```cmd
python main.py --config ./config/mixformer_V2_J.yaml --phase test --save-score True --weights ./checkpoints/mixformer_V2_J.pt --device 0 
python main.py --config ./config/mixformer_V2_B.yaml --phase test --save-score True --weights ./checkpoints/mixformer_V2_B.pt --device 0 
```



# Ensemble


## ensemble

We generate each model's weight depending on testset_A
```cmd
python ensemble2B.py
```

After getting these weights, we merge all these score_pklfile and reformat the shape.
```cmd
python ensemble2.py
```