import tensorflow as tf
import numpy as np
import os, sys, shutil
from tqdm import tqdm
from capsLayer2 import CapsLayer2 as CapsLayer
from utils import reduce_sum
from config import cfg
import utils as utils
import matplotlib.pyplot as plt
import loadCIFAR




'''Hyper-parameters'''
lr = 0.0005
epochs = 5




train_set, train_setY, test_set, test_setY = loadCIFAR.getData()


def evaluator_model(input):
    epsilon = 1e-9
    with  tf.variable_scope('CapsuleNet', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('Conv1_layer'):
            # conv1 = tf.contrib.layers.conv2d(input, num_outputs=128, kernel_size=9, stride=1, padding='VALID')
            # conv1 = tf.contrib.layers.conv2d(conv1, num_outputs=256, kernel_size=5, stride=1, padding='VALID')
            conv1 = tf.contrib.layers.conv2d(input, num_outputs=512, kernel_size=9, stride=1, padding='VALID')

        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(num_outputs=32, vec_len=16, with_routing=False, layer_type='CONV')
            caps1 = primaryCaps(conv1, kernel_size=8, stride=2)

        with tf.variable_scope('SecondaryCaps_Layer'):
            DigitCaps = CapsLayer(num_outputs=10, vec_len=32, with_routing=True, layer_type='FC')
            Caps2 = DigitCaps(caps1)
            v_length = tf.sqrt(reduce_sum(tf.square(Caps2), axis=2, keepdims=True) + epsilon, name='v_length')
            print(v_length)
        #
        # with  tf.variable_scope('Masking'):
        #     masked_v = tf.multiply(tf.squeeze(Caps2), tf.reshape(y, (-1, 10, 1)), name='masked_v')
        #     print('Masked_V: ', masked_v)
        #
        # with tf.variable_scope('Decoder'):
        #     vector_j = tf.reshape(masked_v, shape=(cfg.batch_size, -1))
        #     fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
        #     assert fc1.get_shape() == [cfg.batch_size, 512]
        #     fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
        #     assert fc2.get_shape() == [cfg.batch_size, 1024]
        #     decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)

    return v_length, Caps2

def decoder(capsules, y):
    with  tf.variable_scope('Masking'):
        masked_v = tf.multiply(tf.squeeze(capsules), tf.reshape(y, (-1, 10, 1)), name='masked_v')
        print('Masked_V: ', masked_v)

    with tf.variable_scope('Decoder'):
        vector_j = tf.reshape(masked_v, shape=(cfg.batch_size, -1))
        fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=1024)
        assert fc1.get_shape() == [cfg.batch_size, 1024]
        fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=4096)
        assert fc2.get_shape() == [cfg.batch_size, 4096]
        decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=3072, activation_fn=tf.sigmoid)

    return decoded



def marginal_loss(v_length, Y):
    max_l = tf.square(tf.maximum(0., cfg.m_plus - v_length))
    max_r = tf.square(tf.maximum(0., v_length - cfg.m_minus))
    max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
    max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))
    T_c = Y
    L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r
    loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1), name='marginal_loss')
    return loss

def reconstruction_loss(decoded, X):
    reshaped = tf.transpose(X, perm=[0,3,1,2])
    original = tf.reshape(reshaped, shape=(cfg.batch_size,-1))
    loss = tf.reduce_mean(tf.square(tf.subtract(original,decoded)), name='reconstruction_loss')
    return loss

x = tf.placeholder(tf.float32, shape=[cfg.batch_size, 32, 32, 3], name='x')
y = tf.placeholder(tf.float32, shape=[cfg.batch_size, 10], name='y')

capsules, capsForDecoder = evaluator_model(x)
decoded = decoder(capsForDecoder, y)
#loss = tf.add(marginal_loss(capsules, y), cfg.regularization_scale * reconstruction_loss(decoded,x), name='total_loss')
loss = marginal_loss(capsules, y)
print('Capsule Output:  ', capsules, ' Decoded: ', decoded)
print('Loss output: ', loss)

optimizer = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(loss)
v_softmax = tf.nn.softmax(logits=capsules, axis=1, name='v_softmax')
predictions = tf.reshape(tf.argmax(input=v_softmax, axis=1), shape=(cfg.batch_size,), name='predictions')
actual = tf.reshape(tf.argmax(input=y, axis=1), shape=(cfg.batch_size,),name='actual')
accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, actual), dtype=tf.float32), name='accuracy')
init = tf.global_variables_initializer()

'''Save the model'''
inputs_dict = {'x' : x}
outputs_dict = {'capsules': capsules, 'predictions': predictions}
export_dir = os.path.dirname(sys.argv[0])+'\\Saved_Model'
if os.path.isdir(export_dir):
    shutil.rmtree(export_dir)
os.makedirs(export_dir)


#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
with tf.Session() as sess:
    sess.run(init)
    losses=[]
    train_accuracy = []
    test_accuracy = []
    num_batches = len(train_setY) // cfg.batch_size
    num_test_batches = len(test_setY) // cfg.batch_size
    #num_batches = 5
    #num_test_batches = 4
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        for batch in tqdm(range(num_batches)):
            X = train_set[batch * cfg.batch_size : (batch + 1) * cfg.batch_size]
            Y = train_setY[batch * cfg.batch_size: (batch + 1) * cfg.batch_size]
            batch_loss, batch_accuracy, _ = sess.run([loss, accuracy, optimizer], feed_dict={x : X, y : Y})
            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
        epoch_loss = epoch_loss / num_batches
        epoch_accuracy = epoch_accuracy / num_batches
        losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        print('Epoch: ', epoch + 1, ' Loss: ', epoch_loss, ' Accuracy: ', epoch_accuracy)
        if (epoch + 1) % 1 == 0:
            t_acc = 0
            for batch in tqdm(range(num_test_batches)):
                X = test_set[batch * cfg.batch_size : (batch + 1) * cfg.batch_size]
                Y = test_setY[batch * cfg.batch_size: (batch + 1) * cfg.batch_size]
                acc, act, pred = sess.run([accuracy, actual, predictions], feed_dict={x : X, y : Y})
                # for pair in list(zip(act, pred)):
                #     print(pair)
                t_acc += acc
            t_acc = t_acc / num_test_batches
            test_accuracy.append(t_acc)
            print('Test Accuracy: ', t_acc)
        print('Saving Model...')
        tf.saved_model.simple_save(sess, export_dir + '\\Model_Epoch_' + str(epoch), inputs_dict, outputs_dict)
        print('...Model Saved')

    print(losses)
    print(train_accuracy)
    print(test_accuracy)
    '''Loss Figure'''
    plt.plot(list(range(1,epochs+1)), losses)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(export_dir+'\\Training Loss.png', format='png')
    plt.close()

    '''Training Accuracy Figure'''
    plt.plot(list(range(1,epochs+1)), train_accuracy)
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(export_dir+'\\Training Accuracy.png', format='png')
    plt.close()

    '''Test Accuracy Figure'''
    plt.plot(list(range(1,epochs+1)), test_accuracy)
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(export_dir+'\\Test Accuracy.png', format='png')
    plt.close()