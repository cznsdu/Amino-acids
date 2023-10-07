#导入模块
import os
import csv
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


#加载数据
Test_File_Path = "E:/pycharm_data/anjisuan/5_Anjisuan_OH_dosage_test50.csv"
tdf = pd.read_csv(Test_File_Path)        #读取测试数据
tdf_f= tdf.astype("float64")
# print(df_f.shape)      #(10,77)


#数据分类
# N = 5
# csv_batch_data = df_f.tail(N)      #取后五行
# print(csv_batch_data.shape)      #(5,77)
# print(csv_batch_data)
# train_batch_data = df_f.iloc[:,0:24]    #取前24列
# print(train_batch_data)

#创建数据集
Xt =  tdf_f.iloc[:,0:24].values
Yt =  tdf_f.iloc[:,24:87].values
# print(X)
# print(Y)

#创建训练集、验证集和测试集
x_test = Xt
y_test = Yt



#创建训练批次
batch_size = 1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'             #消除警告
test_dataset = tf.data.Dataset.from_tensor_slices(
    (x_test,y_test)).batch(batch_size)



#构建tensorflow模型

#构建模型
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=x_test.shape[1]),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(y_test.shape[1]),
])


#编译模型
from keras import metrics,optimizers
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.0001),
              metrics=['mae', 'mse', 'accuracy'])


#加载模型
from sklearn import model_selection, metrics
checkpoint_filepath = os.path.join("E:/Users/shuha/PycharmProjects/anjisuan_fivelayers_128/saved_model_lose50_5.128")
model.load_weights(os.path.join(checkpoint_filepath, 'variables/variables'))
preds = model.predict(test_dataset)
evals = model.evaluate(x=test_dataset, return_dict=True)
# r2 = metrics.r2_score(y_test,preds)
rmse = evals['accuracy']
print(f'Test RMSE: {evals}')
# print(f'Test R2: {r2}')




# 数据输出
np.savetxt('E:/Users/shuha/PycharmProjects/anjisuan_fivelayers_128/test50.csv',preds,delimiter=',')


#结果预览图


#picture 1
a = y_test[0]
b = preds[0]
l1 = plt.plot(range(0,22), a[0:22],'ro--', label="True1")
l2 = plt.plot(range(0,22), b[0:22],'bo--', label="Pred1")
plt.legend(['True1','Pred1'])
plt.savefig('./image_lose50/Met_OH_50')
plt.show()

#picture 2
a = y_test[1]
b = preds[1]
l3 = plt.plot(range(22,35), a[22:35],'ro--', label="True1")
l4 = plt.plot(range(22,35), b[22:35],'bo--', label="Pred1")
plt.legend(['True1','Pred1'])
plt.savefig('./image_lose50/Phe_OH_50')
plt.show()

#picture 3
a = y_test[2]
b = preds[2]
l5 = plt.plot(range(35,53), a[35:53],'ro--', label="True1")
l6 = plt.plot(range(35,53), b[35:53],'bo--', label="Pred1")
plt.legend(['True1','Pred1'])
plt.savefig('./image_lose50/Cys_OH_50')
plt.show()

#picture 4
a = y_test[3]
b = preds[3]
l7 = plt.plot(range(53,58), a[53:58],'ro--', label="True1")
l8 = plt.plot(range(53,58), b[53:58],'bo--', label="Pred1")
plt.legend(['True1','Pred1'])
plt.savefig('./image_lose50/Glu_OH_50')
plt.show()

#picture 5
a = y_test[4]
b = preds[4]
l9 = plt.plot(range(58,63), a[58:63],'ro--', label="True1")
l10 = plt.plot(range(58,63), b[58:63],'bo--', label="Pred1")
plt.legend(['True1','Pred1'])
plt.savefig('./image_lose50/Pro_OH_50')
plt.show()

#picture 6
a = y_test[5]
b = preds[5]
l11 = plt.plot(range(0,22), a[0:22],'ro--', label="True1")
l12 = plt.plot(range(0,22), b[0:22],'bo--', label="Pred1")
plt.legend(['True1','Pred1'])
plt.savefig('./image_lose50/Met_O_50')
plt.show()

#picture 7
a = y_test[6]
b = preds[6]
l13 = plt.plot(range(22,35), a[22:35],'ro--', label="True1")
l14 = plt.plot(range(22,35), b[22:35],'bo--', label="Pred1")
plt.legend(['True1','Pred1'])
plt.savefig('./image_lose50/Phe_O_50')
plt.show()

#picture 8
a = y_test[7]
b = preds[7]
l15 = plt.plot(range(35,53), a[35:53],'ro--', label="True1")
l16 = plt.plot(range(35,53), b[35:53],'bo--', label="Pred1")
plt.legend(['True1','Pred1'])
plt.savefig('./image_lose50/Cys_O_50')
plt.show()

#picture 9
a = y_test[8]
b = preds[8]
l17 = plt.plot(range(53,58), a[53:58],'ro--', label="True1")
l18 = plt.plot(range(53,58), b[53:58],'bo--', label="Pred1")
plt.legend(['True1','Pred1'])
plt.savefig('./image_lose50/Glu_O_50')
plt.show()

#picture 10
a = y_test[9]
b = preds[9]
l19 = plt.plot(range(58,63), a[58:63],'ro--', label="True1")
l20 = plt.plot(range(58,63), b[58:63],'bo--', label="Pred1")
plt.legend(['True1','Pred1'])
plt.savefig('./image_lose50/Pro_O_50')
plt.show()

#picture 11
a = y_test[10]
b = preds[10]
l21 = plt.plot(range(0,22), a[0:22],'ro--', label="True1")
l22 = plt.plot(range(0,22), b[0:22],'bo--', label="Pred1")
plt.legend(['True1','Pred1'])
plt.savefig('./image_lose50/Met_O3_50')
plt.show()

#picture 12
a = y_test[11]
b = preds[11]
l23 = plt.plot(range(22,35), a[22:35],'ro--', label="True1")
l24 = plt.plot(range(22,35), b[22:35],'bo--', label="Pred1")
plt.legend(['True1','Pred1'])
plt.savefig('./image_lose50/Phe_O3_50')
plt.show()

#picture 13
a = y_test[12]
b = preds[12]
l25 = plt.plot(range(35,53), a[35:53],'ro--', label="True1")
l26 = plt.plot(range(35,53), b[35:53],'bo--', label="Pred1")
plt.legend(['True1','Pred1'])
plt.savefig('./image_lose50/Cys_O3_50')
plt.show()

#picture 14
a = y_test[13]
b = preds[13]
l27 = plt.plot(range(53,58), a[53:58],'ro--', label="True1")
l28 = plt.plot(range(53,58), b[53:58],'bo--', label="Pred1")
plt.legend(['True1','Pred1'])
plt.savefig('./image_lose50/Glu_O3_50')
plt.show()

#picture 15
a = y_test[14]
b = preds[14]
l29 = plt.plot(range(58,63), a[58:63],'ro--', label="True1")
l30 = plt.plot(range(58,63), b[58:63],'bo--', label="Pred1")
plt.legend(['True1','Pred1'])
plt.savefig('./image_lose50/Pro_O3_50')
plt.show()


#picture 16
a = y_test[15]
b = preds[15]
l31 = plt.plot(range(0,22), a[0:22],'ro--', label="True1")
l32 = plt.plot(range(0,22), b[0:22],'bo--', label="Pred1")
plt.legend(['True1','Pred1'])
plt.savefig('./image_lose50/Met_H2O2_50')
plt.show()

#picture 17
a = y_test[16]
b = preds[16]
l33 = plt.plot(range(22,35), a[22:35],'ro--', label="True1")
l34 = plt.plot(range(22,35), b[22:35],'bo--', label="Pred1")
plt.legend(['True1','Pred1'])
plt.savefig('./image_lose50/Phe_H2O2_50')
plt.show()

#picture 18
a = y_test[17]
b = preds[17]
l35 = plt.plot(range(35,53), a[35:53],'ro--', label="True1")
l36 = plt.plot(range(35,53), b[35:53],'bo--', label="Pred1")
plt.legend(['True1','Pred1'])
plt.savefig('./image_lose50/Cys_H2O2_50')
plt.show()

#picture 19
a = y_test[18]
b = preds[18]
l37 = plt.plot(range(53,58), a[53:58],'ro--', label="True1")
l38 = plt.plot(range(53,58), b[53:58],'bo--', label="Pred1")
plt.legend(['True1','Pred1'])
plt.savefig('./image_lose50/Glu_H2O2_50')
plt.show()

#picture 20
a = y_test[19]
b = preds[19]
l39 = plt.plot(range(58,63), a[58:63],'ro--', label="True1")
l40 = plt.plot(range(58,63), b[58:63],'bo--', label="Pred1")
plt.legend(['True1','Pred1'])
plt.savefig('./image_lose50/Pro_H2O2_50')
plt.show()

