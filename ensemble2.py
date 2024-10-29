import argparse
import pickle
import numpy as np
from tqdm import tqdm



# 读取数据

label = open('./data/test_label_B.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open('./work_dir/ctrgcn_test_boneB/epoch1_test_score.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open('./work_dir/ctrgcn_test_jointB/epoch1_test_score.pkl', 'rb')
r2 = list(pickle.load(r2).items())
r3 = open('./work_dir/tdgcn_test_boneB/epoch1_test_score.pkl', 'rb')
r3 = list(pickle.load(r3).items())
r4 = open('./work_dir/tdgcn_test_jointB/epoch1_test_score.pkl', 'rb')
r4 = list(pickle.load(r4).items())
r5 = open('./work_dir/tdgcn_test_jointB/epoch1_test_score.pkl', 'rb')
r5 = list(pickle.load(r5).items())
r6 = open('./work_dir/mstgcn_test_boneB/epoch1_test_score.pkl', 'rb')
r6 = list(pickle.load(r6).items())
r7 = open('./work_dir/mstgcn_test_jointB/epoch1_test_score.pkl', 'rb')
r7 = list(pickle.load(r7).items())
r8 = open('./work_dir/transformer/joint.pkl', 'rb')
r8 = list(pickle.load(r8).items())
r9 = open('./work_dir/transformer/bone.pkl', 'rb')
r9 = list(pickle.load(r9).items())


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
    _, r77 = r7[i]
    _, r88 = r8[i]
    _, r99 = r9[i]


    r = 3.65945 * r11 + 2.84651 * r22 + 1.44252 * r33 + -0.65244 * r44 + -1.0 * r55 + -0.21555 * r66 + 3.81894 * r77 + 1.54528 * r88 + 0.96178 * r99
    weighted_scores.append(r)
    # 计算 top-5 准确率
# 保存加权分数到新的Pickle文件
with open('./work_dir/weighted_test_scoresB.pkl', 'wb') as f:
    pickle.dump(weighted_scores, f)


# 输出准确率
print("done！！！")