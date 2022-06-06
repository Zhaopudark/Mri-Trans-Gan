import numpy as np
import tensorflow as tf

#这里使用TensorFlow自带的数据集作为测试，以下是导入数据集代码 
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 784).astype('float32')/255.0
train_labels = tf.one_hot(train_labels,depth=10)

Xtrain = tf.data.Dataset.from_tensor_slices(train_images).take(5000)
Ytrain = tf.data.Dataset.from_tensor_slices(train_labels).take(5000)
Xtest = tf.data.Dataset.from_tensor_slices(train_images).skip(5000).take(200)
Ytest = tf.data.Dataset.from_tensor_slices(train_labels).skip(5000).take(200)
# def one_hot(x):
#     return tf.one_hot()
# train_images,train_labels =  tf.data.Dataset.from_tensor_slices(basic).take(5000),

# Xtest =  tf.data.Dataset.from_tensor_slices(basic).skip(5000).take(200)
# for item in train_images:
#     print(tf.reduce_mean(item))
#     break

# Xtrain, Ytrain = mnist.train.next_batch(5000)  #从数据集中选取5000个样本作为训练集
# Xtest, Ytest = mnist.test.next_batch(200)    #从数据集中选取200个样本作为测试集


# 输入占位符
xtr = tf.compat.v1.placeholder("float", [None, 784])
xte = tf.compat.v1.placeholder("float", [784])


# 计算L1距离
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# 获取最小距离的索引
pred = tf.arg_min(distance, 0)

#分类精确度
accuracy = 0.

# 初始化变量
init = tf.global_variables_initializer()

# 运行会话，训练模型
with tf.Session() as sess:

    # 运行初始化
    sess.run(init)

    # 遍历测试数据
    for i in range(len(Xtest)):
        # 获取当前样本的最近邻索引
        nn_index = sess.run(pred, feed_dict={xtr: Xtrain, xte: Xtest[i, :]})   #向占位符传入训练数据
        # 最近邻分类标签与真实标签比较
        print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
            "True Class:", np.argmax(Ytest[i]))
        # 计算精确度
        if np.argmax(Ytrain[nn_index]) == np.argmax(Ytest[i]):
            accuracy += 1./len(Xtest)

    print("Done!")
    print("Accuracy:", accuracy)
