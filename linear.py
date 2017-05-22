# -*- coding: UTF-8 -*-

# 简化调用库名
import tensorflow as tf
import numpy as np

# 模拟生成100对数据对, 对应的函数为y = x * 0.1 + 0.3
#x_data = np.random.rand(100).astype("float32")
#y_data = x_data * 0.1 + 0.3
#print x_data, y_data
import numpy as np
import matplotlib.pyplot as plt

num_points = 1000
vectors_set = []
for i in xrange(num_points):
	x1 = np.random.normal(0.0, 0.55)
	y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
	vectors_set.append([x1, y1])


x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

# 指定w和b变量的取值范围（注意我们要利用TensorFlow来得到w和b的值）
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

w_value = []
b_value = []
step_value = []

# 最小化均方误差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

# 初始化TensorFlow参数
init = tf.initialize_all_variables()

# 运行数据流图（注意在这一步才开始执行计算过程）
sess = tf.Session()
sess.run(init)

# 观察多次迭代计算时，w和b的拟合值
for step in xrange(201):
    step_value.append(step)
    a1 = sess.run(W)[0]
    a2 = sess.run(b)[0]
    w_value.append(a1)
    b_value.append(a2)
    sess.run(train)
    if step % 20 == 0:
    	print(step, a1, a2)

# 最好的情况是w和b分别接近甚至等于0.1和0.3

#plt.plot(x_data, y_data, 'ro', label='Original data')
#plt.plot(x_data, sess.run(W)*x_data+sess.run(b))
plt.plot(step_value, w_value)
plt.plot(step_value, b_value)
plt.legend()
plt.show()

