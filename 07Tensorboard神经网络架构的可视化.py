import tensorflow as tf
import numpy as np


def add_layer(inputs,in_size,out_size,activation_function=None):                   #定义好隐藏层（这里输出层用的也是这个代码）
    with tf.name_scope('layer'):
        with tf.name_scope('weigths'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]),name='weight')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases

        if activation_function is None:
            outputs=Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b)
        return outputs


x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

with tf.name_scope('inputs'):                                                     #我的理解是运行到这里的时候,会添加一个名叫inputs的节点,这个节点点开后包含一个名叫x_input和y_input节点
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')


l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
predition=add_layer(l1,10,1,activation_function=None)
with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition),reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
writer=tf.summary.FileWriter("log1",sess.graph)               #把刚刚建立好的架构加载到一个文件中，这里就是把架构加载到log文件夹下面，也就是运行之后会自动产生一个文件在log目录下(log不存在时也会自动产生一个log文件夹)

#writer=tf.train.SummaryWriter("log",sess.graph)                  #加载架构也可以用这种方法(具体还是要看tensorflow的版本)


