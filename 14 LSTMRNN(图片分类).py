
#为什么在出现深度神经网络(CNN,DNN如果不加卷积层的话,那么就和NN没什么区别了,所以CNN是DNN的典型代表)之后,又提出
#RNN呢?因为有些问题的输入是有时序关系的,而NN和DNN的输入是平行关系,没有任何的时序而言(DNN可以理解是在NN的基础上多了几个隐藏层或者还多了卷积层)
#而LSTM RNN(长短期记忆)可以理解是RNN的一种最为流行的形式之一,为了解决输入是有序数据的问题
#传统的RNN有两个问题,记忆力差(对于长序列效果很差),不易训练(权重初始值不好设定,梯度消失,梯度爆炸),而LSTM可以很好的解决这个问题


#经过LSTM层之后的结果形式

# x_input = [batch,n_steps,n_input] #先将数据解压成输入格式
# tmp = []
# for i in range(x_input.shape[1]): #x_input.shape[1]=n_steps
#     tmp.append(x_input[:,i,:])
# #tmp -> [[batch,n_input],[batch,n_input]...] -> len(tmp)=n_steps
# #tmp.shape -> [n_steps, batch, n_input]
# outputs = [] #保存每次的输出结果
# state = cell.zero_state()
# for x in tmp: # x -> [batch,n_input],循环n_steps次
#     output,state = cell(x,state)
#     outputs.append(output) #outputs -> [y_1,y_2,...y_steps]
# return outputs,state
#最后的输出变成了 outputs=(n_steps,batch,n_hidden) 所以我们需要tf.transpose(outputs,[1,0,2])，这样就可以取到最后一步的output

#上面的代码就能看到是一步步的计算所有的batch之后再进行下一步计算,并把每一步中的多个结果都存起来,放在一个list里,所以最后得到的list的结构是(n_steps,batch,n_hidden),所以想得到最后一步的所有batch的结果就得tf.transpose(outputs,[1,0,2])



#判断一张图片是1到10中的那个数
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#this is data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)                      #每张图片是28*28个像素点

#hyperparameters(超参数)
lr=0.001
training_iters=100000
batch_size=128                  #自定义有一次有多少数据进行训练

n_inputs=28                     #minst data input(image shape:28*28)
n_steps=28                      #因为数据是28*28,每次输入一行的28,要输入28次
n_hidden_unis=128               #自定义的隐藏层的神经元个数
n_classes=10                    #输出类别(判断图片到底是是1到10数据中的哪个数)


#tf Graph input
x=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y=tf.placeholder(tf.float32,[None,n_classes])

#define weights
weights={#(28,128)
        'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_unis])),
        #(128,10)
        'out':tf.Variable(tf.random_normal([n_hidden_unis,n_classes]))
}

biases={
    #(128,)
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_unis,])),
    'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}

def RNN(x,weights,biases):
    #hidden layer for input to cell

    #x(128 batch,28 steps,28 inputs)
    #====>(128*28,28 inputs)
    x=tf.reshape(x,[-1,n_inputs])
    #x_in=====>(128 batch *28 steps,128 hidden)
    x_in=tf.matmul(x,weights['in']+biases['in'])
    #x_in ===> (128batch,28 steps,128 hidden)
    x_in=tf.reshape(x_in,[-1,n_steps,n_hidden_unis])

    #cell
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis,forget_bias=1.0,state_is_tuple=True)       #cell也相当于一个隐藏层,里面有n_hidden_unis=128个神经元,forget_bias是忘记门的初始化偏置值
    #lstm cell is divided into parts(c_state,m_state)
    _init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)                                   #128个data的转态全部初始为0

    #outputs是一个list,每一步的结果都存在outputs里,而states是最后一步的states
    outputs,states=tf.nn.dynamic_rnn(lstm_cell,x_in,initial_state=_init_state,time_major=False)     #time_major来表示数据维度,是指要考虑多少个时间序列,这里的x_in,是(128batch,28 steps,128 hidden),步数在第二个位置(次要的位置),不是第一个位置,所以是False


    #hidden layer for output as the final results
    #方式一:results= tf.matmul(states[1],weights['out'])+biases['out']       #states[1]=outputs[-1]
    #方式二:unpack to list[(batch,outputs)....]*steps
    outputs = tf.unpack(tf.transpose(outputs,[1,0,2]))
    results=tf.matmul(outputs[-1],weights['out'])+biases['out']
    return results

pred=RNN(x,weights,biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
train_op=tf.train.AdamOptimizer(lr).minimize(cost)

#有多少是预测对的
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step=0
    while step*batch_size<training_iters:
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        batch_xs=batch_xs.reshape([batch_size,n_steps,n_inputs])
        sess.run(train_op,feed_dict={x:batch_xs,
                                     y:batch_ys})
        if step % 20 ==0:
            print(sess.run(accuracy,feed_dict={x:batch_xs,
                                               y:batch_ys}))
        step += 1

