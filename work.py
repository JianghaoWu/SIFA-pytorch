import numpy as np
cat_data = np.load('./data10/T2/vs_gk_4_t2_seg_z007.npz')
print(cat_data.files)
a = cat_data['arr_1']
no0 = np.where(a > 0 ,1,0)
b = no0.sum()
print(a, b)