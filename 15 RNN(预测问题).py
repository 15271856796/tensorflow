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




import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_START=0
TIME_STEPS=20
BATCH_SIZE=50
INPUT_SIZE=1
OUTPUT_SIZE=1
CELL_SIZE=10
LR=0.006
BATCH_START_TEST=0

#生成数据的function
def get_batch():
    global BATCH_START,TIME_STEPS
    #xs shape(50batch,20steps)
    xs=np.arange(BATCH_START,BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE,TIME_STEPS))
    seq=np.sin(xs)
    res=np.cos(xs)
    BATCH_START+=TIME_STEPS
    ##plt.plot(xs[0,:],res[0,:],'r',xs[0,:],seq[0,:],'b--')
    ##plt.show()
    return [seq[:,:,np.newaxis],res[:,:,np.newaxis],xs]

def LSTMRNN(n_steps,input_size,output_size,cell_size,batch_size):
    def __init__(self,n_steps,input_size,output_size,cell_size,batch_size):
        self.n_steps=n_steps
        self.input_size=input_size
        self.output_size=output_size
        self.batch_size=batch_size
        with tf.name_scope('inputs'):
            self.xs=tf.placeholder(tf.float32,[None,n_steps,input_size],name='xs')
            self.ys=tf.placeholder(tf.floats32,[None,n_steps,output_size],name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op=tf.train.AdamOptimizer(LR).minimize(self.cost)

def add_input_layer(self,):
    l_in_x=tf.reshape(self.xs,[-1,self.input_size],name='2_2D')  #(batch*n_step,input_size)(50*20,1)
    Ws_in=self._weight_variables([self.input_size,self.cell_size])
    bs_in = self._bias_variable([self.cell_size])
    with tf.name_scope('Wx_plus_b'):
        l_in_y=tf.matmul(l_in_x,Ws_in)+bs_in
    self.l_in_y=tf.reshape(l_in_y,[-1,self.n_steps,self.cell_size],name='2_3D')    #(50,20,10)  x_shape=(batch,n_steps,n_input)

def add_cell(self,):
    lstm_cell = tf.nn.rnn.cell.BasicLSTMCell(self.cell_size,forget_bias=1.0,state_is_tulpe=True)
    with tf.name_scope('initial_state'):
        self.cell_init_state=lstm_cell.zero_state(self.batch_size,dtype=tf.float32)
    self.cell.output,cell_final_state=tf.nn.dnnamic_rnn(lstm_cell,self.l_in_y,initial_state=self.cell_init_state,time_major=False)  # y_shape=(n_steps,batch,n_hidden)


def add_output_layer(self,):
    l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='3_2D')  # (batch*n_step,input_size) (50,10)
    Ws_out = self._weight_variables([self.cell_size, self.output_size])
    bs_out = self._bias_variable([self.output_size,])
    with tf.name_scope('Wx_plus_b'):
        self.pred=tf.matmul(l_out_x,Ws_out)+bs_out

def compute_cost(self):
    losses=tf.nn.seq2seq.sequence_loss_by_example(
        [tf.reshape(self.pred,[-1],name='reshape_pred')],
        [tf.reshape(self.ys,[-1],name='reshape_target')],
        [tf.ons([self.batch_size*self.n_steps],dtype=tf.float32)],
        average_acress_timesteps=True,
        softmax_loss_function=self.msr_error,
        name='losses'
    )
    with tf.name_scope('average_cose'):
        self.cost=tf.div(
            tf.reduce_sum(losses,name='losses_sum'),
            tf.cast(self.batch_size,tf.float32),
            name='average_cost'
        )
        tf.scalar_summary('cost',self.cost)

def msr_error(self,y_pre,y_target):
    return tf.square(tf.sub(y_pre,y_target))

def _weight_variable(self,shape,name='weights'):
    initialize = tf.random_normal_initializer(mean=0,stddev=1.,)
    return tf.get_variable(shape=shape,initializer=initialize,name=name)


def _bias_vaeiable(self,shape,name='biases'):
    initialize=tf.constant_initializer(0,1)
    return tf.get_variable(name=name,shape=shape,initializer=initialize)

if __name__=='__main__':
    model=LSTMRNN(TIME_STEPS,INPUT_SIZE,OUTPUT_SIZE,CELL_SIZE,BATCH_SIZE)
    sess=tf.Session()
    merged=tf.summary.merge_all()
    # $ tensorboard -logdir='logs'
    writer=tf.summary.FileWriter("logRNN",sess.graph)


    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.show()
    for i in range(200):
        seq,res,xs=get_batch()
        if i==0:
            feed_dict={
                model.xs:seq,
                model.ys:res
                #creat initial state
            }
        else:
            feed_dict={
                model.xs:seq,
                model.ys:res,
                model.cell_init_state:state
            }
        cost,state,pred=sess.run(
            [model.train_op,model.cost,model.cell_final_state,model.pred],
            feed_dict=feed_dict
        )
        #plotting
        plt.plot(xs[0,:],res[0].flatten(),'r',xs[0,:],pred.flatten()[:TIME_STEPS],'b--')
        plt.ylim(-1,2,1,2)
        plt.draw()
        plt.pause(0.3)

        if i%20==0:
            print('cost:',round(cost,4))
            result=sess.run(merged,feed_dict)
            writer.add_summary(result,i)




