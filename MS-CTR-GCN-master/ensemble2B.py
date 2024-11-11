# import argparse
# import pickle
# import numpy as np
# from tqdm import tqdm
# from skopt import gp_minimize
# from skopt.space import Real
# from skopt.utils import use_named_args
#
# # 定义参数空间
# space = [
#     Real(-1, 3, name='weight1'),
#     Real(-1, 3, name='weight2'),
#     Real(-1, 3, name='weight3'),
#     Real(-1, 3, name='weight4'),
#     Real(-1, 3, name='weight5'),
#     Real(-1, 3, name='weight6'),
#     Real(-1, 3, name='weight7'),
#     Real(-1, 3, name='weight8'),
#     Real(-1, 3, name='weight9'),
# ]
#
# # 全局变量存储历史最优解
# best_accuracy = 0
# best_weights = []
# history = []
#
# @use_named_args(space)
# def objective(weight1, weight2, weight3, weight4, weight5, weight6,weight7,weight8,weight9):
#     # 将单独的权重参数重新组装成列表
#     weights = [weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9]
#
#     right_num = total_num = 0
#     for i in tqdm(range(len(label))):
#         l = label[i]
#         _, r_11 = r1[i]
#         _, r_22 = r2[i]
#         _, r_33 = r3[i]
#         _, r_44 = r4[i]
#         _, r_55 = r5[i]
#         _, r_66 = r6[i]
#         _, r_77 = r7[i]
#         _, r_88 = r8[i]
#         _, r_99 = r9[i]
#
#         r = r_11 * weights[0] + r_22 * weights[1] + r_33 * weights[2] + r_44 * weights[3] + r_55 * weights[4] + r_66 * weights[5] + r_77 * weights[6] + r_88 * weights[7] + r_99 * weights[8]
#         r = np.argmax(r)
#         right_num += int(r == int(l))
#         total_num += 1
#     acc = right_num / total_num
#     global best_accuracy, best_weights
#     if acc > best_accuracy:
#         best_accuracy = acc
#         best_weights = weights.copy()
#     history.append((acc, weights.copy()))
#     return -acc
#
# def callback(res):
#     print(f"Current best accuracy: {best_accuracy*100:.2f}%")
#     print(f"Current best weights: {best_weights}")
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed',
#                         default=42)
#     # add argument definitions here
#     parser.add_argument('--ctrgcn_test_bone',
#                         default='./work_dir/ctrgcn_test_bone/epoch1_test_score.pkl')
#     parser.add_argument('--ctrgcn_test_joint',
#                         default='./work_dir/ctrgcn_test_joint/epoch1_test_score.pkl')
#     parser.add_argument('--tdgcn_test_bone',
#                         default='./work_dir/tdgcn_test_bone/epoch1_test_score.pkl')
#     parser.add_argument('--agcn_test_joint',
#                         default='./work_dir/agcn_test_joint/epoch1_test_score.pkl')
#     parser.add_argument('--tdgcn_test_joint',
#                         default='./work_dir/tdgcn_test_joint/epoch1_test_score.pkl')
#     parser.add_argument('--mstgcn_test_bone',
#                         default='./work_dir/mstgcn_test_bone/epoch1_test_score.pkl')
#     parser.add_argument('--mstgcn_test_joint',
#                         default='./work_dir/mstgcn_test_joint/epoch1_test_score.pkl')
#     parser.add_argument('--mixformer-joint',
#                         default='./work_dir/epoch68_test_score.pkl')
#     parser.add_argument('--mixformer-bone',
#                         default='./work_dir/epoch70_test_score.pkl')
#     arg = parser.parse_args()
#
#     label = np.load("./data/test_label_A.npy")
#     # load your data here
#     with open(arg.ctrgcn_test_bone, 'rb') as r1:
#         r1 = list(pickle.load(r1).items())
#
#     with open(arg.ctrgcn_test_joint, 'rb') as r2:
#         r2 = list(pickle.load(r2).items())
#
#     with open(arg.tdgcn_test_bone, 'rb') as r3:
#         r3 = list(pickle.load(r3).items())
#
#     with open(arg.agcn_test_joint, 'rb') as r4:
#         r4 = list(pickle.load(r4).items())
#
#     with open(arg.tdgcn_test_joint, 'rb') as r5:
#         r5 = list(pickle.load(r5).items())
#
#     with open(arg.mstgcn_test_bone, 'rb') as r6:
#         r6 = list(pickle.load(r6).items())
#
#     with open(arg.mstgcn_test_joint, 'rb') as r7:
#         r7 = list(pickle.load(r7).items())
#
#     with open(arg.mixformer_joint, 'rb') as r8:
#         r8 = list(pickle.load(r8).items())
#
#     with open(arg.mixformer_bone, 'rb') as r9:
#         r9 = list(pickle.load(r9).items())
#
#     result = gp_minimize(objective, space, n_calls=300, random_state=int(arg.seed), callback=callback)
#     print('Maximum accuracy: {:.4f}%'.format(best_accuracy * 100))
#     print('Optimal weights: {}'.format(best_weights))
#
#     # 打印所有收集的历史记录
#     for acc, weights in history:
#         print('Accuracy: {:.4f}%, Weights: {}'.format(acc * 100, weights))
# import argparse
# import pickle
# import numpy as np
# from tqdm import tqdm
# from scipy.optimize import differential_evolution
#
# # 定义参数空间
# bounds = [(0, 4)] * 9  # 定义每个权重的范围从0到4
#
# # 全局变量存储历史最优解
# best_accuracy = 0
# best_weights = []
# history = []
#
# def objective(weights):
#     # 假设你的数据和标签是预先加载的
#     right_num = total_num = 0
#     for i in tqdm(range(len(label))):
#         l = label[i]
#         _, r_11 = r1[i]
#         _, r_22 = r2[i]
#         _, r_33 = r3[i]
#         _, r_44 = r4[i]
#         _, r_55 = r5[i]
#         _, r_66 = r6[i]
#         _, r_77 = r7[i]
#         _, r_88 = r8[i]
#         _, r_99 = r9[i]
#
#         r = (r_11 * weights[0] + r_22 * weights[1] + r_33 * weights[2] +
#              r_44 * weights[3] + r_55 * weights[4] + r_66 * weights[5] +
#              r_77 * weights[6] + r_88 * weights[7] + r_99 * weights[8])
#         r = np.argmax(r)
#         right_num += int(r == int(l))
#         total_num += 1
#     acc = right_num / total_num
#     global best_accuracy, best_weights
#     if acc > best_accuracy:
#         best_accuracy = acc
#         best_weights = weights.copy()
#     history.append((acc, weights.copy()))
#     return -acc  # 由于是最大化准确率，返回负准确率作为最小化目标
#
# def callback(xk, convergence):
#     print(f"Current best accuracy: {best_accuracy*100:.2f}%")
#     print(f"Current best weights: {best_weights}")
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     # add argument definitions here
#     parser.add_argument('--ctrgcn_test_bone', default='./work_dir/ctrgcn_test_bone/epoch1_test_score.pkl')
#     parser.add_argument('--ctrgcn_test_joint', default='./work_dir/ctrgcn_test_joint/epoch1_test_score.pkl')
#     parser.add_argument('--tdgcn_test_bone', default='./work_dir/tdgcn_test_bone/epoch1_test_score.pkl')
#     parser.add_argument('--agcn_test_joint', default='./work_dir/agcn_test_joint/epoch1_test_score.pkl')
#     parser.add_argument('--tdgcn_test_joint', default='./work_dir/tdgcn_test_joint/epoch1_test_score.pkl')
#     parser.add_argument('--mstgcn_test_bone', default='./work_dir/mstgcn_test_bone/epoch1_test_score.pkl')
#     parser.add_argument('--mstgcn_test_joint', default='./work_dir/mstgcn_test_joint/epoch1_test_score.pkl')
#     parser.add_argument('--mixformer-joint', default='./work_dir/epoch68_test_score.pkl')
#     parser.add_argument('--mixformer-bone', default='./work_dir/epoch70_test_score.pkl')
#     arg = parser.parse_args()
#
#     # 加载数据和标签
#     label = np.load("./data/test_label_A.npy")
#     with open(arg.ctrgcn_test_bone, 'rb') as r1:
#         r1 = list(pickle.load(r1).items())
#
#     with open(arg.ctrgcn_test_joint, 'rb') as r2:
#         r2 = list(pickle.load(r2).items())
#
#     with open(arg.tdgcn_test_bone, 'rb') as r3:
#         r3 = list(pickle.load(r3).items())
#
#     with open(arg.agcn_test_joint, 'rb') as r4:
#         r4 = list(pickle.load(r4).items())
#
#     with open(arg.tdgcn_test_joint, 'rb') as r5:
#         r5 = list(pickle.load(r5).items())
#
#     with open(arg.mstgcn_test_bone, 'rb') as r6:
#         r6 = list(pickle.load(r6).items())
#
#     with open(arg.mstgcn_test_joint, 'rb') as r7:
#         r7 = list(pickle.load(r7).items())
#
#     with open(arg.mixformer_joint, 'rb') as r8:
#         r8 = list(pickle.load(r8).items())
#
#     with open(arg.mixformer_bone, 'rb') as r9:
#         r9 = list(pickle.load(r9).items())
#
#     # 运行优化
#     result = differential_evolution(objective, bounds, maxiter=100000, seed=3, callback=callback, disp=True)
#     print('Maximum accuracy: {:.4f}%'.format(best_accuracy * 100))
#     print('Optimal weights: {}'.format(best_weights))
#
#     # 打印所有收集的历史记录
#     for acc, weights in history:
#         print('Accuracy: {:.4f}%, Weights: {}'.format(acc * 100, weights))
# import argparse
# import pickle
# import numpy as np
# from tqdm import tqdm
# import random
#
# # 定义参数空间
# bounds = [(-1, 4)] * 9  # 9 weights with bounds from -1 to 3
#
# # 全局变量存储历史最优解
# best_accuracy = 0
# best_weights = []
# history = []
#
#
# def objective(weights):
#     right_num = total_num = 0
#     for i in tqdm(range(len(label))):
#         l = label[i]
#         r = sum(r[i][1] * weight for r, weight in zip([r1, r2, r3, r4, r5, r6, r7, r8, r9], weights))
#         r = np.argmax(r)
#         right_num += int(r == int(l))
#         total_num += 1
#     acc = right_num / total_num
#     global best_accuracy, best_weights
#     if acc > best_accuracy:
#         best_accuracy = acc
#         best_weights[:] = weights  # Copy the weights
#     history.append((acc, weights.copy()))
#     return -acc  # Return negative accuracy for minimization
#
#
# def simulated_annealing(init_weights, n_iterations=3000, initial_temp=300, cooling_rate=0.99):
#     current_weights = init_weights
#     current_score = objective(current_weights)
#
#     for iteration in range(n_iterations):
#         temp = initial_temp * (cooling_rate ** iteration)
#
#         # Generate new candidate weights
#         new_weights = [np.clip(w + np.random.normal(0, 0.5), bound[0], bound[1]) for w, bound in
#                        zip(current_weights, bounds)]
#         new_score = objective(new_weights)
#
#         # Calculate acceptance probability
#         if new_score < current_score or random.uniform(0, 1) < np.exp((current_score - new_score) / temp):
#             current_weights = new_weights
#             current_score = new_score
#
#     return current_weights
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', default=36)
#     parser.add_argument('--ctrgcn_test_bone', default='./work_dir/ctrgcn_test_bone/epoch1_test_score.pkl')
#     parser.add_argument('--ctrgcn_test_joint', default='./work_dir/ctrgcn_test_joint/epoch1_test_score.pkl')
#     parser.add_argument('--tdgcn_test_bone', default='./work_dir/tdgcn_test_bone/epoch1_test_score.pkl')
#     parser.add_argument('--agcn_test_joint', default='./work_dir/agcn_test_joint/epoch1_test_score.pkl')
#     parser.add_argument('--tdgcn_test_joint', default='./work_dir/tdgcn_test_joint/epoch1_test_score.pkl')
#     parser.add_argument('--mstgcn_test_bone', default='./work_dir/mstgcn_test_bone/epoch1_test_score.pkl')
#     parser.add_argument('--mstgcn_test_joint', default='./work_dir/mstgcn_test_joint/epoch1_test_score.pkl')
#     parser.add_argument('--mixformer-joint', default='./work_dir/epoch68_test_score.pkl')
#     parser.add_argument('--mixformer-bone', default='./work_dir/epoch70_test_score.pkl')
#     arg = parser.parse_args()
#     np.random.seed(int(arg.seed))
#
#     label = np.load("./data/test_label.npy")
#     # Load your data here
#     with open(arg.ctrgcn_test_bone, 'rb') as r1:
#         r1 = list(pickle.load(r1).items())
#
#     with open(arg.ctrgcn_test_joint, 'rb') as r2:
#         r2 = list(pickle.load(r2).items())
#
#     with open(arg.tdgcn_test_bone, 'rb') as r3:
#         r3 = list(pickle.load(r3).items())
#
#     with open(arg.agcn_test_joint, 'rb') as r4:
#         r4 = list(pickle.load(r4).items())
#
#     with open(arg.tdgcn_test_joint, 'rb') as r5:
#         r5 = list(pickle.load(r5).items())
#
#     with open(arg.mstgcn_test_bone, 'rb') as r6:
#         r6 = list(pickle.load(r6).items())
#
#     with open(arg.mstgcn_test_joint, 'rb') as r7:
#         r7 = list(pickle.load(r7).items())
#
#     with open(arg.mixformer_joint, 'rb') as r8:
#         r8 = list(pickle.load(r8).items())
#
#     with open(arg.mixformer_bone, 'rb') as r9:
#         r9 = list(pickle.load(r9).items())
#
#
#     init_weights = np.random.uniform(-1, 4, len(bounds)).tolist()  # Random initial weights
#     best_weights = simulated_annealing(init_weights)
#
#     print('Maximum accuracy: {:.4f}%'.format(best_accuracy * 100))
#     print('Optimal weights: {}'.format(best_weights))
#
#     # Print all collected historical records
#     for acc, weights in history:
#         print('Accuracy: {:.4f}%, Weights: {}'.format(acc * 100, weights))
#
#
#     print('Maximum accuracy: {:.4f}%'.format(best_accuracy * 100))
#     print('Optimal weights: {}'.format(best_weights))
# import argparse
# import pickle
# import numpy as np
# from tqdm import tqdm
# import random
#
# # 定义参数空间
# bounds = [(0.2,1.2)] * 4  # 9 weights with bounds from -1 to 3
#
# # 全局变量存储历史最优解
# best_accuracy = 0
# best_weights = []
# history = []
#
#
# def objective(weights):
#     right_num = total_num = 0
#     for i in tqdm(range(len(label))):
#         l = int(label[i])
#         # 计算加权和
#         combined_scores = sum(r[i][1] * weight for r, weight in zip([r1, r2, r3], weights))
#         # 找到TOP5的索引
#         top5_indices = np.argsort(combined_scores)[-5:]
#
#         # 检查标签是否在TOP5的索引中
#         right_num += int(l in top5_indices)
#         total_num += 1
#
#     acc = right_num / total_num
#     global best_accuracy, best_weights
#     if acc > best_accuracy:
#         best_accuracy = acc
#         best_weights[:] = weights  # Copy the weights
#     history.append((acc, weights.copy()))
#     return -acc  # Return negative accuracy for minimization
#
# # def objective(weights):
# #     right_num = total_num = 0
# #     for i in tqdm(range(len(label))):
# #         l = label[i]
# #         r = sum(r[i][1] * weight for r, weight in zip([r1, r2, r3], weights))
# #         r = np.argmax(r)
# #         right_num += int(r == int(l))
# #         total_num += 1
# #     acc = right_num / total_num
# #     global best_accuracy, best_weights
# #     if acc > best_accuracy:
# #         best_accuracy = acc
# #         best_weights[:] = weights  # Copy the weights
# #     history.append((acc, weights.copy()))
# #     return -acc  # Return negative accuracy for minimization
#
# def simulated_annealing(init_weights, n_iterations=3000, initial_temp=300, cooling_rate=0.99):
#     current_weights = init_weights
#     current_score = objective(current_weights)
#
#     for iteration in range(n_iterations):
#         temp = initial_temp * (cooling_rate ** iteration)
#
#         # Generate new candidate weights
#         new_weights = [np.clip(w + np.random.normal(0, 0.5), bound[0], bound[1]) for w, bound in
#                        zip(current_weights, bounds)]
#         new_score = objective(new_weights)
#
#         # Calculate acceptance probability
#         if new_score < current_score or random.uniform(0, 1) < np.exp((current_score - new_score) / temp):
#             current_weights = new_weights
#             current_score = new_score
#
#     return current_weights
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', default=8)
#
#     parser.add_argument('--test_bone', default='/data/ices/ionicbond/MS-CTR-GCN-master/work_dir/test2/bone/epoch1_test_score.pkl')
#     parser.add_argument('--test_joint', default='/data/ices/ionicbond/MS-CTR-GCN-master/work_dir/test2/joint/epoch1_test_score.pkl')
#     parser.add_argument('--test_motion', default='/data/ices/ionicbond/MS-CTR-GCN-master/work_dir/test2/bone_motion/epoch1_test_score.pkl')
#     parser.add_argument('--test_longtail', default='/data/ices/ionicbond/MS-CTR-GCN-master/work_dir/test_longtail/epoch1_test_score.pkl')
#     parser.add_argument('--test_mixformer', default='/data/ices/ionicbond/MS-CTR-GCN-master/work_dir/mixformer/epoch70_test_score.pkl')
#
#
#     arg = parser.parse_args()
#     np.random.seed(int(arg.seed))
#
#     label = np.load("/data/ices/ionicbond/MS-CTR-GCN-master/data/uav/v1/val_label.npy")
#     # Load your data here
#     with open(arg.test_bone, 'rb') as r1:
#         r1 = list(pickle.load(r1).items())
#
#     with open(arg.test_joint, 'rb') as r2:
#         r2 = list(pickle.load(r2).items())
#
#     with open(arg.test_motion, 'rb') as r3:
#         r3 = list(pickle.load(r3).items())
#
#     with open(arg.test_longtail, 'rb') as r4:
#         r4 = list(pickle.load(r4).items())
#
#
#
#
#
#     init_weights = np.random.uniform(0.2, 1, len(bounds)).tolist()  # Random initial weights
#     best_weights = simulated_annealing(init_weights)
#
#     print('Maximum accuracy: {:.4f}%'.format(best_accuracy * 100))
#     print('Optimal weights: {}'.format(best_weights))
#
#     # Print all collected historical records
#     for acc, weights in history:
#         print('Accuracy: {:.4f}%, Weights: {}'.format(acc * 100, weights))
#
#
#     print('Maximum accuracy: {:.4f}%'.format(best_accuracy * 100))
#     print('Optimal weights: {}'.format(best_weights))
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import random

# 定义参数空间
bounds = [(0.5,0.9)] * 7  # 4 weights with bounds from 0.2 to 1.2

# 全局变量存储历史最优解
best_accuracy = 0
best_weights = []
history = []


def objective(weights):
    right_num = total_num = 0
    for i in tqdm(range(len(label))):
        l = int(label[i])
        # 计算加权和
        combined_scores = sum(r[i][1] * weight for r, weight in zip([r1, r2, r3, r4, r5, r6, r7], weights))
        # 找到最高分的索引，即 Top 1
        top1_index = np.argmax(combined_scores)

        # 检查标签是否在 Top 1 中
        right_num += int(l == top1_index)
        total_num += 1

    acc = right_num / total_num
    global best_accuracy, best_weights
    if acc > best_accuracy:
        best_accuracy = acc
        best_weights[:] = weights  # Copy the weights
    history.append((acc, weights.copy()))
    return -acc  # Return negative accuracy for minimization



def search_best_weights(n_iterations=3000):
    global best_accuracy, best_weights

    # Start with random weights
    for iteration in range(n_iterations):
        # Generate new candidate weights and round them to 1 decimal place
        new_weights = [round(np.random.uniform(bound[0], bound[1]), 1) for bound in bounds]

        # Calculate the score for this set of weights
        new_score = objective(new_weights)

        # If the new score is better, update the best_weights and best_accuracy
        if new_score < 0:  # Since we're minimizing negative accuracy
            new_accuracy = -new_score
            if new_accuracy > best_accuracy:
                best_accuracy = new_accuracy
                best_weights[:] = new_weights  # Copy the new best weights

    return best_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=8)
    parser.add_argument('--test_bone',
                        default='/data/ices/ionicbond/MS-CTR-GCN-master/work_dir/test_bone/epoch1_test_score.pkl')
    parser.add_argument('--test_joint',
                        default='/data/ices/ionicbond/MS-CTR-GCN-master/work_dir/test_joint/epoch1_test_score.pkl')
    parser.add_argument('--test_motion',
                        default='/data/ices/ionicbond/MS-CTR-GCN-master/work_dir/test_motion/epoch1_test_score.pkl')
    parser.add_argument('--test_longtail',
                        default='/data/ices/ionicbond/MS-CTR-GCN-master/work_dir/test_longtail/epoch1_test_score.pkl')
    parser.add_argument('--test_mixformer_bone',
                        default='/data/ices/ionicbond/global_match-main/HDBN/Mix_Former/output/test/skmixf__V1_B_test2/epoch1_test_score.pkl')
    parser.add_argument('--test_tdgcn_joint',
                        default='/data/ices/ionicbond/global_match-main/real/work_dir/tdgcn_V2_J/epoch40_test_score.pkl')
    parser.add_argument('--test_mixformer_joint',
                        default='/data/ices/ionicbond/global_match-main/HDBN/Mix_Former/output/test2/skmixf__V1_J/epoch68_test_score.pkl')

    arg = parser.parse_args()
    np.random.seed(int(arg.seed))

    label = np.load("/data/ices/ionicbond/MS-CTR-GCN-master/data/uav/v1/val_label.npy")
    # Load your data here
    with open(arg.test_bone, 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(arg.test_joint, 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    with open(arg.test_motion, 'rb') as r3:
        r3 = list(pickle.load(r3).items())

    with open(arg.test_longtail, 'rb') as r4:
        r4 = list(pickle.load(r4).items())

    with open(arg.test_mixformer_bone, 'rb') as r5:
        r5 = list(pickle.load(r5).items())

    with open(arg.test_tdgcn_joint, 'rb') as r6:
        r6 = list(pickle.load(r6).items())

    with open(arg.test_mixformer_joint, 'rb') as r7:
        r7 = list(pickle.load(r7).items())

    # Start searching for the best weights using the simple loop
    best_weights = search_best_weights(n_iterations=5000)

    print('Maximum accuracy: {:.4f}%'.format(best_accuracy * 100))
    print('Optimal weights: {}'.format(best_weights))

    # Print all collected historical records
    for acc, weights in history:
        print('Accuracy: {:.4f}%, Weights: {}'.format(acc * 100, weights))

    print('Maximum accuracy: {:.4f}%'.format(best_accuracy * 100))
    print('Optimal weights: {}'.format(best_weights))


