import tensorflow as tf
import os
from pre_data import comm_clean, get_x
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class TextCNNModel:    
    def __init__(self,train_x,train_y,embed_size,output_size,vocab_size,n_filters,lr,top_k):
        self.X = tf.placeholder(tf.int64,[None,None])
        self.Y = tf.placeholder(tf.int64,[None,output_size])
        embeddings = tf.Variable(tf.random.uniform([vocab_size,embed_size]))
        embedded_X = tf.nn.embedding_lookup(embeddings,self.X)
        kernels = [3,4,5]
        covres = []
        for size in kernels:
            cov = tf.layers.conv1d(inputs=embedded_X,filters=n_filters,kernel_size=size,strides=1,\
                                  use_bias=True,activation=tf.nn.relu,padding='valid')
            # cov: batchsize,(step-size)/strides +1,filters
            cov = tf.transpose(cov,[0,2,1])
            topk = tf.nn.top_k(cov,top_k).values
            # cov transpose的目的是为了获取filters结果的topk的值
            topk = tf.transpose(topk,[0,2,1])
            # 得到topk的值之后，再返回回来，转变为conv卷积完成之后的形式
            covres.append(topk)
            # covres shape是3，batchsize，topk,filters
        covres = tf.concat(covres,axis=-1)
        # 最后一维处拼接后shape: batchsize,topk,filters*3
        covres = tf.reshape(covres,[-1,top_k*n_filters*len(kernels)])
        # covres压扁为二维：batchsize,topk叠加结果，之后要接全连接了
        self.logits = tf.layers.dense(covres,output_size)
        # 全连接输出维度 logits: batchsize,2
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.Y))
        # softmax交叉熵损失，把logits的二维输出变为概率后进行计算
        self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.cost)
        # 得到logits的二维输出以后，取大的值就是预测结果，这里用的是【01】代表1,positive；【10】代表0negative
        self.pred = tf.arg_max(self.logits,1)
        true_Y = tf.arg_max(self.Y,1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred,true_Y),tf.float32))
        
    @staticmethod       
    def predict(sentences,model,word2id,max_len,sess):
        input_X = [sen for sen in map(comm_clean,sentences)]
        pred_X = []
        for sen in input_X:
            pred_X.append(get_x(sen,word2id,max_len))
        predict = sess.run(model.pred,feed_dict={model.X:pred_X})
        return predict  