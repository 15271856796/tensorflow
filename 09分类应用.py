import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)                         #当没有这个数据的时候,会在网上自动下载数据,自动下载数据的工具

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b=tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs=Wx_plus_b                                                                 #以下步骤是指要通过激励函数
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs                                                                         #得到输出值

def compute_accuracy(v_xs,v_ys):
    global prediction                                                                #拿已经训练好的参数来预测测试数据
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})                                 #输出的是一行十列,没一个元素都是概率(0与1之间的数)
    correct_prediction=tf.equal(tf.arg_max(y_pre,1),tf.arg_max(v_ys,1))              #判断测试正确与否
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))                #测试正确的个数所占的比列
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

#define placeholder for inputs to network
xs=tf.placeholder(tf.float32,[None,784])   #28*28
ys=tf.placeholder(tf.float32,[None,10])                                               #ys有10个分类,所以是10列,只有一个1

#add output layer
prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)    #softmax一般来处理分类

#add loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))     #一般搭配softmax来做分类算法
train_step = tf .train.GradientDescentOptimizer(0.5).minimize(cross_entropy)



sess= tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)                             #提取一部分的x与y,每次训练就只训练其中的100个列子
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i % 50==0:
        #准确度的输出(准确度用test数据来测试)
        print(compute_accuracy(mnist.test.images,mnist.test.labels))          #mnist中的整个数据分为训练train和测试数据test


