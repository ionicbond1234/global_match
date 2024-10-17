# import pickle
#
# import numpy as np
#
# # 读取原数据
# with open('data/data_A/train_label.npy', 'rb') as f:
#     data = np.load(f)  # 这里传递的是文件对象f
# max_label = 0
# # print(data[:])
# #
# skeleton_list = [f"{i+1}.skeleton" for i in range(len(data))]
#
# # 组合成所需的格式
# result = [skeleton_list, data.tolist()]
#
# # 输出结果
# print(result)
# # 将结果写回原文件
#
# with open('data/data_A/train_label.pkl', 'wb') as f:
#     pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
#
# # import pickle
# #
# # def check_pickle_file(file_path):
# #     try:
# #         with open(file_path, 'rb') as f:
# #             # 尝试加载数据
# #             data = pickle.load(f)
# #         print("Pickle file is valid.")
# #         return data  # 如果需要返回数据
# #     except (OSError, pickle.UnpicklingError) as e:
# #         print(f"Error loading pickle file: {e}")
# #         return None  # 或者其他适当的错误处理
# #
# # # 示例用法
# # file_path = './data/test_label_A.pkl'
# # data = check_pickle_file(file_path)
import numpy as np
data = np.load("pred.npy")
print(data.shape)


