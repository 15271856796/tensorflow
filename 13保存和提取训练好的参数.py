import tensorflow as tf
import numpy as np

#save variables to file
W=tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32)
b=tf.Variable([[1,2,3]],dtype=tf.float32)
init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path=saver.save(sess,'my_net/save_net.ckpt')
    print('Save to path',save_path)




#restore variables
#redefine the same shape and same type for your variables
W=tf.Variable(np.arange(1,7).reshape((2,3)),dtype=tf.float32)
b=tf.Variable(np.arange(1,4).reshape((1,3)),dtype=tf.float32)

#not need init step

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,'my_net/save_net.ckpt')                              #与保存好的参数的名字一直就行
    print('w:',sess.run(W))
    print('b:',sess.run(b))

