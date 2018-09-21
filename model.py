# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 11:02:09 2018

@author: yiweigao

该语言模型的目的是给一个句子打分，
训练语料如何准备  输入：一个句子   输出：一个概率，表示得分情况，该概率的取值在（0,1）之间吧

输入：北京 故宫 展品 失窃 事件  可是该如何标注呢，难道所有语料都标为1
错误query：
    北京  故宫 展品 失窃  时间  得分很低才对

输出：故宫 展品 失窃 事件 end


tensorflow中语言模型的目的（PTB数据） 是得到词向量吧
"""

import tensorflow as tf

class LanguageModel(object):
    def __init__(self,config,is_training,initial):
        
        #param
        self.words_num=config.words_num
        self.wordEmbeddingSize=config.wordEmbeddingSize
        self.vocab_size=config.vocab_size
        self.hidesize=config.hidesize
        self.keep_prob=config.keep_prob
        self.level=config.level
        self.lr=config.lr
        self.l2=config.l2
        self.batch=config.batchsize
        
        #input
        self.x=tf.placeholder(tf.int32,[None,self.words_num])
        self.y=tf.placeholder(tf.int32,[None,self.words_num])
        
        input_shape=tf.shape(self.x)
        batch_size=input_shape[0]
        #initialize
        self.ini_variable(initial)
        
        self.x_emb=tf.nn.embedding_lookup(self.wordEmbeddings,self.x) #b,w,emsize
        
        
        #model
        with tf.name_scope("lstm"):
            lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(self.hidesize)
            if self.keep_prob<1:
                lstm_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=self.keep_prob)
            cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*self.level)
            (hidden_state,_)=tf.nn.dynamic_rnn(cell,self.x_emb,dtype=tf.float32) #batch,words,emb
            
        #final predict
        hidden_state_reshape=tf.reshape(hidden_state,[-1,self.hidesize])
        self.logits=tf.matmul(hidden_state_reshape,self.final_w)+self.final_b #[batch*words_num,vocab_size]
        
        self.logits=tf.reshape(self.logits,[-1,self.words_num,self.vocab_size])
        
        #loss
        loss=tf.contrib.seq2seq.sequence_loss(self.logits,self.y,tf.ones([batch_size,self.words_num]))
        loss_batch=tf.reduce_sum(loss)/self.batch #all train batch_size dataset ,average perplexity
        
        l2_loss=tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])*self.l2
        self.cost=loss_batch + l2_loss
        
        
        
        if not is_training:
            return
        
        #train
        self.global_step=tf.Variable(0,name="global_step",trainable=False)
        tvars=tf.trainable_variables()
        grads,_=tf.clip_by_global_norm(tf.gradients(self.cost,tvars),config.max_grad_norm)
        optimizer=tf.train.AdadeltaOptimizer(self.lr)
        self.train_op=optimizer.apply_gradients(zip(grads,tvars),global_step=self.global_step)
        
        
    def ini_variable(self,initial):
        self.wordEmbeddings=tf.get_variable(name="embeddings",shape=[self.vocab_size,self.wordEmbeddingSize],initializer=initial)

        self.final_w=tf.get_variable("final_w",[self.hidesize,self.vocab_size],initializer=initial)
        self.final_b=tf.get_variable("final_bias",[self.vocab_size],initializer=tf.zeros_initializer())