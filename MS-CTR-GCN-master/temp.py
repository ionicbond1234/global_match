import os
import datetime

# 替换为你想查询的文件路径
file_path = '/data/ices/ionicbond/global_match-main/HDBN/Mix_Former/output/skmixf__V1_J/epoch70_test_score.pkl'

# 获取从纪元以来的修改时间(秒)
modification_time = os.path.getmtime(file_path)

# 转换为日期时间格式
modification_datetime = datetime.datetime.fromtimestamp(modification_time)

print("文件的最后修改时间是:", modification_datetime)
# import numpy as np
# data = np.load("/data/ices/ionicbond/MS-CTR-GCN-master/data/uav/v1/test_joint2.npy")
# print(data.shape)
# import numpy as np
# import pickle
#
# ones_array = np.ones(4307, dtype=int)
# skeleton_list = [f"{i+1}.skeleton" for i in range(0,4307)]
# result = [skeleton_list, ones_array.tolist()]
# with open('/data/ices/ionicbond/MS-CTR-GCN-master/data/uav/v1/test_label2.pkl', 'wb') as f:
#      pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)