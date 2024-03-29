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
learning_rate=0.001
training_epochs=20
batch_size=256
display_step=1


#Network Parameters
n_input=784   #MNIST data input(img shape:28*28)

#tf Graph input(only pictures)
X=tf.placeholder('float32',[None,n_input])
n_hidden_1=128
n_hidden_2=64
n_hidden_3=10
n_hidden_4=2

weights={
    "encoder_h1":tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    "encoder_h2":tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    "encoder_h3":tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3])),
    "encoder_h4":tf.Variable(tf.random_normal([n_hidden_3,n_hidden_4])),
    "decoder_h1":tf.Variable(tf.random_normal([n_hidden_4,n_hidden_3])),
    "decoder_h2":tf.Variable(tf.random_normal([n_hidden_3,n_hidden_2])),
    "decoder_h3":tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
    "decoder_h4":tf.Variable(tf.random_normal([n_hidden_1,n_input]))
}

biases={
    'encoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3':tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4':tf.Variable(tf.random_normal([n_hidden_4])),
    'decoder_b1':tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b3':tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b4':tf.Variable(tf.random_normal([n_input]))
}

#Building the encoder
def encoder(x):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),biases['encoder_b1']))
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['encoder_h2']),biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']), biases['encoder_b4'])
    return layer_4

#Building the decoder 注意压缩和解码的过程一样,都是sigmoid
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']), biases['decoder_b4']))
    return layer_4


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


    encoder_result=sess.run(encoder_op,feed_dict={X:mnist.test.images})
    plt.scatter(encoder_result[:,0],encoder_result[:,1],c=mnist.test.labels)    #c=mnist.test.lables代表颜色分类
    plt.show()