#Use tensorflow2
# convert tf record to npz for SIFAdata

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os 

##tfrecord to numpy
def decode_tfrecords(example):
    features = {
        'data_vol':tf.io.FixedLenFeature([],tf.string),
        'label_vol':tf.io.FixedLenFeature([],tf.string)
    }
    feature_dict = tf.io.parse_single_example(example,features)
    img = tf.io.decode_raw(feature_dict['data_vol'],out_type = tf.float32)
    lab = tf.io.decode_raw(feature_dict['label_vol'],out_type = tf.float32)
    img = tf.reshape(img,[256,256,3])
    lab = tf.reshape(lab,[256,256,3])
    return img,lab
'''
#one case to test
files = 'ct_train_slice169.tfrecords'
rawdata = tf.data.TFRecordDataset(files)
dataset = rawdata.map(decode_tfrecords)
itera = tf.compat.v1.data.make_one_shot_iterator(dataset)
img,lab = itera.get_next()
img = img.numpy()
lab = lab.numpy()
print(img.shape)
print(lab.shape)
print(np.unique(lab))
img = img[:,:,1]
lab = lab[:,:,1]
plt.figure()
plt.imshow(img,cmap='gray')
plt.figure()
plt.imshow(lab)
plt.show()
'''
##convert tfrecord to npz
#for CT
path = 'SIFAdata/train/ct_train/'
pathlist = os.listdir(path)
for i in range(len(pathlist)):
    files = pathlist[i]
    name = files.split('.')[0]
    rawdata = tf.data.TFRecordDataset(path+files)
    dataset = rawdata.map(decode_tfrecords)
    itera = tf.compat.v1.data.make_one_shot_iterator(dataset)
    img,lab = itera.get_next()
    img = img.numpy()
    lab = lab.numpy()
    np.savez('UDA/data/SIFAdata/ct_train/'+name,img,lab)

#for MR    
path = 'SIFAdata/train/mr_train/'
pathlist = os.listdir(path)
for i in range(len(pathlist)):
    files = pathlist[i]
    name = files.split('.')[0]
    rawdata = tf.data.TFRecordDataset(path+files)
    dataset = rawdata.map(decode_tfrecords)
    itera = tf.compat.v1.data.make_one_shot_iterator(dataset)
    img,lab = itera.get_next()
    img = img.numpy()
    lab = lab.numpy()
    np.savez('UDA/data/SIFAdata/mr_train/'+name,img,lab)    
