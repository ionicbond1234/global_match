import argparse
import pickle
import numpy as np
from tqdm import tqdm



# 读取数据

label = open('/data/ices/ionicbond/MS-CTR-GCN-master/data/uav/v1/test_label2.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open('/data/ices/ionicbond/MS-CTR-GCN-master/work_dir/generate/bone/epoch1_test_score.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open('/data/ices/ionicbond/MS-CTR-GCN-master/work_dir/generate/joint/epoch1_test_score.pkl', 'rb')
r2 = list(pickle.load(r2).items())
r3 = open('/data/ices/ionicbond/MS-CTR-GCN-master/work_dir/generate/motion/epoch1_test_score.pkl', 'rb')
r3 = list(pickle.load(r3).items())
r4 = open('/data/ices/ionicbond/MS-CTR-GCN-master/work_dir/generate/longtail/epoch1_test_score.pkl', 'rb')
r4 = list(pickle.load(r4).items())
r5 = open('/data/ices/ionicbond/global_match-main/HDBN/Mix_Former/output/test2/skmixf__V1_B_test2/epoch1_test_score.pkl', 'rb')
r5 = list(pickle.load(r5).items())
r6 = open('/data/ices/ionicbond/global_match-main/real/work_dir/tdgcn_test_joint/epoch1_test_score.pkl', 'rb')
r6 = list(pickle.load(r6).items())
r7 = open('/data/ices/ionicbond/global_match-main/HDBN/Mix_Former/output/test2/skmixf__V1_J2/epoch1_test_score.pkl', 'rb')
r7 = list(pickle.load(r7).items())


# 初始化统计变量
right_num = total_num = right_num_5 = 0
weighted_scores = []

# 计算加权分数
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    _, r33 = r3[i]
    _, r44 = r4[i]
    _, r55 = r5[i]
    _, r66 = r6[i]
    _, r77= r7[i]


    r = 0.9 * r11 + 0.6 * r22 + 0.6 * r33 + 0.7 * r44 + 0.6 * r55 + 0.9 * r66 + 0.9  * r77
    weighted_scores.append(r)
    # 计算 top-5 准确率
# 保存加权分数到新的Pickle文件
with open('./work_dir/weighted_test_scoresB.pkl', 'wb') as f:
    pickle.dump(weighted_scores, f)

import pickle
import numpy as np

# 路径到你的Pickle文件
pickle_file_path = './work_dir/weighted_test_scoresB.pkl'

# 读取Pickle文件
with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

# 确保数据是NumPy数组
if not isinstance(data, np.ndarray):
    data = np.array(data)

# 设置保存的.npy文件路径
npy_file_path = 'pred.npy'

# 保存为.npy格式
np.save(npy_file_path, data)

print(f"Data has been successfully saved to {npy_file_path}")


# 输出准确率
print("done！！！")