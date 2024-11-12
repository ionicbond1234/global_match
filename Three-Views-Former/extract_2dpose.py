import numpy as np
import sys


CS_train_V2_data = np.load("/data/ices/ionicbond/global_match-main/HDBN/train_joint_new.npy")
CS_train_V2_label = np.load("/data/ices/ionicbond/global_match-main/real/data/data/train_label.npy")
CS_test_V2_data = np.load("/data/ices/ionicbond/global_match-main/HDBN/test_joint_new.npy")
CS_test_V2_label = np.load("/data/ices/ionicbond/global_match-main/real/data/data/val_label.npy")





np.savez('./save_2d_pose/V2.npz', x_train=CS_train_V2_data, y_train=CS_train_V2_label,
         x_test=CS_test_V2_data, y_test=CS_test_V2_label)