#先对数据进行压缩(将784->256->128,将元素点变少),再原路径的返回进行解压,
# 拿解压后的数据和原数据进行对比,计算误差,利用误差训练参数
#最后训练好后,拿测试图片去进行实验,上面一行输出原测试图片,下面一行输出解压再还原后的图片


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Import MNIST data
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
mnist =input_data.read_data_sets("/encoder/data/",one_hot=False)
tf.logging.set_verbosity(old_v)

#Visualize decoder detting
#Parameters
learning_rate=0.01
training_epochs=5
batch_size=256
display_step=1
example_to_show=10

#Network Parameters
n_input=784   #MNIST data input(img shape:28*28)

#tf Graph input(only pictures)
X=tf.placeholder('float32',[None,n_input])
n_hidden_1=256
n_hidden_2=128

weights={
    "encoder_h1":tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    "encoder_h2":tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    "decoder_h1":tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
    "decoder_h2":tf.Variable(tf.random_normal([n_hidden_1,n_input]))
}

biases={
    'encoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2':tf.Variable(tf.random_normal([n_input]))
}

#Building the encoder
def encoder(x):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),biases['encoder_b1']))
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['encoder_h2']),biases['encoder_b2']))
    return layer_2

#Building the decoder 注意压缩和解码的过程一样,都是sigmoid
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2


#Construct model

encoder_op=encoder(X)
decoder_op=decoder(encoder_op)

#Prediction
y_pred=decoder_op
#Targes(Lables)are the input data
y_true=X

#Define oss and optimizer,minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true-y_pred,2))
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

#Initializing the variable
init = tf.initialize_all_variables()

#Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch=int(mnist.train.num_examples/batch_size)
    #Training cycle
    for epoch in range(training_epochs):
        #Loop over all batches
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)   #max(x)=1,min(x)=0
            #Run optimization op(backprop) and cose op(to get loss value)
            _,c=sess.run([optimizer,cost],feed_dict={X:batch_xs})
            #Display logs per epoch step:
            if epoch % display_step == 0:
                print("Epoch:",'%04d'%(epoch+1),"cost=","{:.9f}".format(c))
    print("Optimization Finished!")

    encode_decode=sess.run(y_pred,feed_dict={X:mnist.test.images[:example_to_show]})
    #Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(example_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
        a[1][i].imshow(np.reshape(encode_decode[i],(28,28)))
    plt.show()