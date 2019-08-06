import tensorflow as tf

input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)

output = tf.add(input1,input2)
with tf.Session() as sess:
    for i in range(3):
        print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))


