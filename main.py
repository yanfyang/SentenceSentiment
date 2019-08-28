from pre_data import *
from model import TextCNNModel
import tensorflow as tf



if __name__ == '__main__':
    
    # 准备数据
    negfile = 'data/neg.xls'
    posfile = 'data/pos.xls'
    print('read data ...')
    shuffledata = prepare_data(negfile,posfile)
    shuffledata.head()
    shuffledata['comm'] = shuffledata['comm'].apply(comm_clean)
    print('complete data loading, start building vocab ...')
    # 构建词典
    word2id, id2word = build_vocab(list(shuffledata['comm']))
    # 留一法分出训练集和测试集7:3
    train_frac = 0.7
    train_size = int(train_frac * shuffledata.shape[0])
    train_x,test_x,train_y,test_y = list(shuffledata['comm'][:train_size]),list(shuffledata['comm'][train_size:]),\
                                    list(shuffledata['label'][:train_size]),list(shuffledata['label'][train_size:])

    # 训练模型
    epoch = 14
    top_k = 5
    output_size = 2
    batch_size = 64
    max_len = 50
    embed_size = 128
    lr = 1e-3
    n_filters = 50
    vocab_size = len(word2id)
    print('start training model ...')
    tf.reset_default_graph()
    sess = tf.Session()
    model = TextCNNModel(train_x,train_y,embed_size,output_size,vocab_size,n_filters,lr,top_k)
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        batch_train = get_batch(train_x,train_y,output_size,batch_size,word2id,max_len)
        loss, acc = 0.0,0.0
        for batch_X,batch_Y in batch_train:
            batch_loss,batch_accuracy,_ = sess.run([model.cost,model.accuracy, model.optimizer],\
                                                  feed_dict={model.X:batch_X,model.Y:batch_Y})
            if loss == 0.0:
                loss += batch_loss
                acc =+ batch_accuracy
            else:
                loss = (loss + batch_loss)/2
                acc = (acc + batch_accuracy)/2
        # ↓ 每隔2个epoch打印一下结果
        if i%2 == 0:
            batch_test = get_batch(test_x,test_y,output_size,batch_size,word2id,max_len)
            test_loss, test_acc = 0.0,0.0
            for test_X,test_Y in batch_test:
                batch_test_loss,batch_test_accuracy,_ = sess.run([model.cost,model.accuracy, model.optimizer],\
                                                      feed_dict={model.X:test_X,model.Y:test_Y})
                if test_loss == 0.0:
                    test_loss += batch_test_loss
                    test_acc =+ batch_test_accuracy
                else:
                    test_loss = (test_loss + batch_test_loss)/2
                    test_acc = (test_acc + batch_test_accuracy)/2
            print('train loss:{:.5f}, train acc:{:.5f}%, test loss:{:5f},test acc:{:5f}%'.format(loss,acc*100,test_loss,test_acc*100))

    # 预测新的句子
    print('model complete, test some cases ...')
    start = 90
    end = 100
    sentences = test_x[start:end]
    for sen in sentences:
        sen_predict = TextCNNModel.predict([sen],model,word2id,max_len,sess)
        print(re.sub(' ','',sen),'\nPredict result : ',sen_predict[0],' Truly result : ',test_y[start])
        start += 1
