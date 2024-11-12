# 示例代码
import numpy as np

# 加载 npz 文件
data = np.load('/data/ices/ionicbond/global_match-main/real/data/data/train_joint.npy')


# # 打印包含的数组名称
# print(data.files
#       )  # 输出: ['arr1', 'arr2']
#
# # 访问数组
# arr1 = data['x_train']
# # arr2 = data['arr2']
print(data.shape)

# 删除第三维度的第二个索引
# axis=2 表示操作的是第三个维度，index=1 表示删除该维度的第二个索引
updated_arr = np.delete(data, 2, axis=1)
swapped = updated_arr.swapaxes(1, 2)
# print("交换轴后的数组:\n", swapped)
print(swapped.shape)
np.save('train_joint_new.npy', swapped)


