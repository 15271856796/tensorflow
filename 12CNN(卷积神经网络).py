#卷积神经网络的提出:为什么要加一层卷积,是为了降低训练的复杂度,先对需要训练的数据进行特征提取,也就是通过卷积拿到我们
# 有用的数据,降低数据维度后再放入神经网络中,这个卷积矩阵也是训练的参数

#CNN与RNN的应用场景不一样,RNN一般用于文本生成,比如知道'这顿饭很好' 后面补全的话,可以是'吃',因为RNN是记忆机制


import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)                         #当没有这个数据的时候,会在网上自动下载数据,自动下载数据的工具


def compute_accuracy(v_xs,v_ys):
    global prediction                                                                #拿已经训练好的参数来预测测试数据
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})                                 #输出的是一行十列,没一个元素都是概率(0与1之间的数)
    correct_prediction=tf.equal(tf.arg_max(y_pre,1),tf.arg_max(v_ys,1))              #判断测试正确与否
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))                #测试正确的个数所占的比列
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

def weight_variable(shape):
    inital = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(inital)
def bias_variable(shape):
    inital=tf.constant(0.1,shape=shape)
    return tf.Variable(inital)

def conv2d(x,W):
    #stride[1,x_movement,y_movement,1]
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')       #padding有两种方式 'VALID'和'SAME'

def max_pool_2x2(x):                                           #在卷积层的时候尽可能的保留多的信息,在池化层再进行信息提取
    # stride[1,x_movement,y_movement,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#define placeholder for inputs to network
xs=tf.placeholder(tf.float32,[None,784])   #28*28
ys=tf.placeholder(tf.float32,[None,10])                                              #ys有10个分类,所以是10列,只有一个1
x_image=tf.reshape(xs,[-1,28,28,1])                                                  #黑白图片是rgb=1,所以最后一个参数是1,也就是图片的通道为1

#conv1 layer
W_conv1 = weight_variable([5,5,1,32])                                                 #5*5*1*32 的意思是每个卷积核是5*5*1的,但是有32个
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)     #output size 28*28*32
h_pool1=max_pool_2x2(h_conv1)                           #outpur size 14*14*32

#conv2 layer
W_conv2 = weight_variable([5,5,32,64])                                                  #把高度从32变成64,每个卷积核的通道是32,且有64个卷积核
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)                                     #output size 14*14*64
h_pool2=max_pool_2x2(h_conv2)                                                           #outpur size 7*7*64

#func1 layer
W_fcl=weight_variable([7*7*64,1024])
b_fcl=bias_variable([1024])
#[n_samples,7,7,64]->>[n_samples,7*7*64]
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])                                              #将数据拍平
h_fcl=tf.nn.relu(tf.matmul(h_pool2_flat,W_fcl)+b_fcl)

#func2 layer
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
prediction=tf.nn.softmax(tf.matmul(h_fcl,W_fc2)+b_fc2)


#add loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))     #一般搭配softmax来做分类算法
train_step = tf .train.AdadeltaOptimizer(0.0001).minimize(cross_entropy)


sess= tf.Session()

sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)                             #提取一部分的x与y,每次训练就只训练其中的100个列子
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i % 50==0:
        #准确度的输出(准确度用test数据来测试)
        print(compute_accuracy(mnist.test.images,mnist.test.labels))          #mnist中的整个数据分为训练train和测试数据test


