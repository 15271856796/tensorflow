import tensorflow as tf
matrix1 =tf.constant([[3,3]])                             #矩阵的表示方法
matrix2=tf.constant([[2],                                 #矩阵的表示方法
                     [2]])

product=tf.matmul(matrix1,matrix2)                         #矩阵运算

with tf.Session() as sess:
    result = sess.run(product)
    print(result)