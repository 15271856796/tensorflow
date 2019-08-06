#首先要知道什么是激励函数 由于输入和输出之间并不一定非要是线性关系,即下一个神经元的输入不一定非得是
#上一个神经元的weight*x+biases的关系,所以要判断这个输入到底的值是什么,也就是通过激励函数进行激活

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):                          #构建神经层
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))                             #初始化weights,Variable是之后需要优化的参数
    biases=tf.Variable(tf.zeros([1,out_size]) + 0.1)                                      #初始化biases
    Wx_plus_b=tf.matmul(inputs,Weights) + biases                                          #通过weight和biases得到结果值
    if activation_function is None:
        outputs=Wx_plus_b                                                                 #以下步骤是指要通过激励函数
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs                                                                         #得到输出值


x_data=np.linspace(-1,1,300)[:,np.newaxis]                                                 #初始化输入的数据,一个有300个例子
noise = np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise                                                         #初始化输出的正确的值,此时x_data与y_data并不是线性的

xs=tf.placeholder(tf.float32,[None,1])                                                     #None代表无论给多少列子都可以,这里是x_data是300个例子
ys= tf.placeholder(tf.float32,[None,1])
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)                                       #构建好隐藏层
predition=add_layer(l1,10,1,activation_function=None)                                      #构建好输出层
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition),reduction_indices=[1]))          #误差函数
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)                         #优化器,目标是最小化误差
init = tf.initialize_all_variables()                                                       #创建初始化所以变量的手柄
sess = tf.Session()                                                                        #上面的所有过程均没有被执行,只有在会话run之后才会被执行
sess.run(init)                                                                             #初始化的是Variable变量

#可视化显示
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()                                          #可以让plt.show()后不会暂停程序
plt.show()


for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})                                   #由于执行train_step时候用到xs和ys所以得进行赋值
    if i %  50==0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))                              #由于执行loss的时候也用到了xs,ys 所以也需要赋值,注意不要觉得loss可以直接sess.run(loss)
        # try:
        #     ax.line.remove(line[0])
        # except Exception:
        #     pass
        # prediction_value=sess.run(predition,feed_dict={xs:x_data})
        # line = ax.plot(x_data,prediction_value,'r-',lw=5)
        # plt.pause(0.1)                        #暂停0.1秒
