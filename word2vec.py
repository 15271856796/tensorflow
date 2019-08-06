import jieba
import collections
import numpy as np
import tensorflow as tf
import os
import pickle as pkl

class word2vec():
    def __init__(self,
                 vocab_list=None,
                 embedding_size=200,
                 win_len=3,
                 learning_rate=1.0,
                 num_samples=1000,
                 logdir='',
                 model_path=None):
        self.batch_size=None
        if model_path != None:
            self.load_model(model_path)
        else:
            assert type(vocab_list) == list
            self.vocab_list=vocab_list
            self.vocab_size=len(vocab_list)
            self.embedding_size=embedding_size
            self.win_len=win_len
            self.num_sampled=num_samples
            self.learning_rate=learning_rate
            self.logdir=logdir
            self.word2id={}
            for i in range(self.vocab_size):
                self.word2id[self.vocab_list[i]]=i

            self.train_words_num=0
            self.train_sents_num=0
            self.train_times_num=0

            self.train_loss_records=collections.deque(maxlen=10)
            self.train_loss_k10=0
        self.build_graph()
        self.init_op()
        if model_path != None:
            tf_modesl_path=os.path.join(model_path,'tf_vars')
            self.saver.restore(self.sess,tf_modesl_path)




    def train_by_sentence(self,input_sentence=[]):
        sent_num=len(input_sentence)
        batch_inputs=[]
        batch_labels=[]
        for sent in input_sentence:
            for i in range(len(sent)):
                start=max(0,i-self.win_len)
                end=min(len(sent),i+self.win_len)
                for index in range(start,end):
                    if index==i:
                        continue
                    else:
                        input_id=self.word2id.get(sent[i])
                        label_id=self.word2id.get(sent[index])
                        if not (input_id and label_id):
                            continue
                        batch_inputs.append(input_id)
                        batch_labels.append(label_id)
        if len(batch_inputs)==0:
            return
        batch_inputs=np.array(batch_inputs,dtype=np.int32)
        batch_labels=np.array(batch_labels,dtype=np.int32)
        batch_labels=np.reshape(batch_labels,[batch_labels.__len__(),1])

        feed_dict={self.train_inputs:batch_inputs,
                   self.train_labels:batch_labels}

        #_,loss_val,summary_str=self.sess.run(self.train_op,self.loss,self.merged_symmary_op,feed_dict=feed_dict)
        self.train_loss_k10=np.mean(self.train_loss_records)
        if self.train_sents_num%1000==0:
            #self.summary_writer.add_summary(summary_str,self.train_sents_num)
            print('(a)sentencess dealed,loss:(b)'.format(a=self.train_sents_num,b=self.train_loss_k10))



        self.loss = self.sess.run(self.train_op,feed_dict=feed_dict)


        self.train_words_num+=len(batch_inputs)
        self.train_sents_num+=len(input_sentence)
        self.train_times_num+=1

    def init_op(self):
        self.sess=tf.Session(graph=self.graph)
        self.sess.run(self.init)
        self.summary_writer=tf.summary.FileWriter(self.logdir,self.sess.graph)


    def build_graph(self):
        self.graph=tf.Graph()
        with self.graph.as_default():
            self.train_inputs=tf.placeholder(tf.int32,shape=[self.batch_size])
            self.train_labels=tf.placeholder(tf.int32,shape=[self.batch_size,1])
            self.embedding_dict=tf.Variable(tf.random_uniform([self.vocab_size,self.embedding_size],-1,0,1,0))
            self.nec_weight=tf.Variable(tf.truncated_normal(shape=[self.vocab_size,self.embedding_size]))
            self.bias=tf.Variable(tf.zeros([self.vocab_size]))
            embed= tf.nn.embedding_lookup(self.embedding_dict,self.train_inputs)
            self.loss=tf.reduce_mean(tf.nn.nce_loss(weights=self.nec_weight,
                                                    biases=self.bias,
                                                    labels=self.train_labels,
                                                    inputs=embed,
                                                    num_sampled=self.num_sampled,
                                                    num_classes=self.vocab_size))
            tf.summary.scalar('loss',self.loss)
            self.train_op=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss=self.loss)

            self.test_word_id=tf.placeholder(tf.int32,shape=[None])
            vec_l2_model=tf.sqrt(tf.reduce_sum(tf.square(self.embedding_dict),1,keep_dims=True))
            avg_l2_model=tf.reduce_mean(vec_l2_model)
            tf.summary.scalar('avg_l2_model',avg_l2_model)
            self.normed_embedding=self.embedding_dict/vec_l2_model
            test_embed=tf.nn.embedding_lookup(self.embedding_dict,self.test_word_id)
            self.similarity=tf.matmul(test_embed,self.normed_embedding,transpose_b=True)

            self.init = tf.global_variables_initializer()
            self.merged_symmary_op=tf.summary.merge_all()
            self.saver=tf.train.Saver()



    def save_model(self,save_path):
        if os.path.isfile(save_path):
            raise RuntimeError('the save path should be a dir')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        model={}
        var_names=['vocab_size',
                   'vocab_list',
                   'learning_rate',
                   'word2id',
                   'embedding_size',
                   'logdir',
                   'win_len',
                   'num_sampled']
        for var in var_names:
            model[var] = eval('self.'+var)
            param_path=os.path.join(save_path,'params.pkl')
            if os.path.exists(param_path):
                os.remove(param_path)
            with open(param_path,'wb') as f:
                pkl.dump(model,f)
        tf_path=os.path.join(save_path,'tf.vars')
        if os.path.exists(tf_path):
            os.remove(tf_path)
        self.saver.save(self.sess,tf_path)

    def load_model(self,model_path):
        if not os.path.exists(model_path):
            raise RuntimeError('file not exists')
        param_path=os.path.join(model_path,'params.pkl')
        with open(param_path,'rb') as f:
            model=pkl.load(f)
            self.vocab_list=model['vocab_list']
            self.vocab_size=model['vocab_size']
            self.logdir=model['logdir']
            self.word2id=model['word2id']
            self.embedding_size=model['embedding_size']
            self.learning_rate=model['learning_rate']
            self.win_len=model['win_len']
            self.num_sampled=model['num_sampled']



    def cal_similsrity(self,test_word_id_list,top_k=10):
        sim_matrix=self.sess.run(self.similarity,feed_dict={self.test_word_id:test_word_id_list})
        sim_mean=np.mean(sim_matrix)
        sim_var=np.mean(np.square(sim_matrix=sim_mean))
        test_words=[]
        near_words=[]
        for i in range(test_word_id_list._len_()):
            test_words.append(self.vocab_list[test_word_id_list][i])
            nearest_id=[sim_matrix[i,:].argsort()[1:top_k+1]]
            nearest_word=[self.vocab_list[x] for x in nearest_id]
            near_words.append(nearest_word)
        return test_words,near_words,sim_mean,sim_var




if __name__=='__main__':
    #读取停用词文档
    stop_words = []
    with open('stop_word.txt',encoding='utf-8') as f:
        line=f.readline()
        while line:
            stop_words.append(line[:-1])
            line=f.readline()
        stop_words=set(stop_words)

    all_word_list=[]
    sentence_list=[]
    with open('2800.txt',encoding='utf-8') as f:
        line=f.readline()
        while line:
            while '\n' in line:
                line=line.replace('\n','')
            if len(line)>0:
                raw_words=list(jieba.cut(line))
                dealed_word=[]
                for word in raw_words:
                    if word not in stop_words:
                        all_word_list.append(word)
                        dealed_word.append(word)
                sentence_list.append((dealed_word))
            line=f.readline()
    word_count=collections.Counter(all_word_list)
    word_count=word_count.most_common(30000)
    word_list=[x[0] for x in word_count]
    w2v=word2vec(vocab_list=word_list,
                 embedding_size=200,
                 learning_rate=1,
                 num_samples=100)
    num_steps=1
    for i in range(num_steps):
        sent=sentence_list[i]
        w2v.train_by_sentence([sent])
    #save_path='save.txt'
    # model_path='model.txt'
    # w2v.save_model(save_path)
    # w2v.load_model(model_path)
    #test_word=['剑气','无敌']
    #test_id=[word_list.index(x) for x in test_word]
    #test_words,near_words,sim_mean,sim_var=w2v.cal_similarity(test_id)

