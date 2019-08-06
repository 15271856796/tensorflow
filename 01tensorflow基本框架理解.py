# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

#creat data
x_data = np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3


#creat tensorflow structure satrt

Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases=tf.Variable(tf.zeros([1]))
y=Weights*x_data+biases

loss = tf.reduce_mean(tf.square(y-y_data))             #误差计算，也就是折损函数
optimizer=tf.train.GradientDescentOptimizer(0.4)       #新建一个优化器,一般用梯度下降的算法实现
train= optimizer.minimize(loss)                        #用优化器对误差进行优化，下面循环的调用,使得下一次的误差要比上一次的

init = tf.initialize_all_variables()

#create tensorflow structure end

sess=tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 ==0:
        print(step,sess.run(Weights),sess.run(biases),sess.run(loss))

