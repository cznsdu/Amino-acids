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
CSV_File_Path = "E:/pycharm_data/anjisuan/5_Anjisuan_OH_dosage_pred50.csv"  #根据数据路径调整
Test_File_Path = "E:/pycharm_data/anjisuan/5_Anjisuan_OH_dosage_test50.csv"
df = pd.read_csv(CSV_File_Path)          #读取训练数据
df_f = df.astype("float64")
tdf = pd.read_csv(Test_File_Path)        #读取测试数据
tdf_f= tdf.astype("float64")
# print(df_f.shape)      #(10,77)
print(df_f)


#数据分类
# N = 5
# csv_batch_data = df_f.tail(N)      #取后五行
# print(csv_batch_data.shape)      #(5,77)
# print(csv_batch_data)
# train_batch_data = df_f.iloc[:,0:24]    #取前24列
# print(train_batch_data)

#创建数据集
X =  df_f.iloc[:,0:24].values
Y =  df_f.iloc[:,24:87].values
Xt =  tdf_f.iloc[:,0:24].values
Yt =  tdf_f.iloc[:,24:87].values
print(X.shape, Y.shape)
# print(X)
# print(Y)

#创建训练集、验证集和测试集
x_train = X
y_train = Y
x_test = Xt
y_test = Yt
print(x_train.shape, y_train.shape)


#创建训练批次
batch_size = 1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'             #消除警告
train_dataset = tf.data.Dataset.from_tensor_slices(
    (x_train,y_train)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (x_test,y_test)).batch(batch_size)
#检查数据是否有误
print(train_dataset.cardinality().numpy())


#构建tensorflow模型
#设置模型参数
epochs = 100000

# #引用先前模型
# model.load_weights(os.path.join(checkpoint_filepath, 'variables/variables'))

#构建模型
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=x_train.shape[1]),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(y_train.shape[1]),
])


#编译模型
from keras import metrics,optimizers
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.0001),
              metrics=['mae', 'mse', 'accuracy'])

#设置调回最小误差的循环，保存其模型的权重和偏置
checkpoint_filepath = os.path.join(os.getcwd(), 'saved_model_lose50_5.128')    #保存模型至当前目录
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='loss',
    mode='min',
    save_best_only=True)

# 训练模型
history = model.fit(train_dataset,
                    epochs=epochs,
                    callbacks=[model_checkpoint_callback])
                    # validation_data=valid_dataset)       #无验证模型


#展示最佳训练周期
best_epoch = np.argmin(np.array(history.history['loss']))
print(best_epoch)


#加载模型
from sklearn import model_selection, metrics
model.load_weights(os.path.join(checkpoint_filepath, 'variables/variables'))
preds = model.predict(test_dataset)
evals = model.evaluate(x=test_dataset, return_dict=True)
# r2 = metrics.r2_score(y_test,preds)
rmse = evals['accuracy']
print(f'Test RMSE: {evals}')
# print(f'Test R2: {r2}')

#绘图
# l1 = plt.plot([range(0,24)], y_test,'--ro', label="True")
# l2 = plt.plot([range(0,24)], preds,'--bo', label="Pred")
#
# plt.legend(["True","Pred"])
# plt.show()

# #picture 1
# a = y_test[0]
# b = preds[0]
# l1 = plt.plot(range(0,22), a[0:22],'ro--', label="True1")
# l2 = plt.plot(range(0,22), b[0:22],'bo--', label="Pred1")
# plt.legend(['True1','Pred1'])
# plt.show()


