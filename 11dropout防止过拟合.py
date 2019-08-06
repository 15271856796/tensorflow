import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

dights=load_digits()
# O到1的数字图
X=dights.data

#y是例如[0,1,0,0,0,0,0,0,0,0,0]的形式
y=dights.target
y=LabelBinarizer().fit_transform(y)

#划分训练集和测试集
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=3)



def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b=tf.matmul(inputs,Weights) + biases
    Wx_plus_b=tf.nn.dropout(Wx_plus_b,keeps_prob)                                          #随机的某些神经元赋值概率为0
    if activation_function is None:
        outputs=Wx_plus_b                                                                 #以下步骤是指要通过激励函数
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

xs=tf.placeholder(tf.float32,[None,64])
ys=tf.placeholder(tf.float32,[None,10])
keeps_prob=tf.placeholder(tf.float32)                                                      #定义随机保留的参数的多少

l1=add_layer(xs,64,100,activation_function=tf.nn.tanh)                                      #构建好隐藏层
predition=add_layer(l1, 100, 10, activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(predition),reduction_indices=[1]))     #一般搭配softmax来做分类算法
tf.summary.scalar('loss',cross_entropy)
train_step = tf .train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

sess= tf.Session()

merged=tf.summary.merge_all()

train_writer=tf.summary.FileWriter("log3/train",sess.graph)
test_writer=tf.summary.FileWriter("log3/test",sess.graph)

sess.run(tf.initialize_all_variables())

for i in range(1000):
    sess.run(train_step,feed_dict={xs:X_train,ys:y_train,keeps_prob:0.6})                                   #由于执行train_step时候用到xs和ys所以得进行赋值
    if i %  50==0:
        #record loss
        train_result=sess.run(merged,feed_dict={xs:X_train,ys:y_train,keeps_prob:1})
        test_result=sess.run(merged,feed_dict={xs:X_test,ys:y_test,keeps_prob:1})
        train_writer.add_summary(train_result,i)
        test_writer.add_summary(test_result,i)

