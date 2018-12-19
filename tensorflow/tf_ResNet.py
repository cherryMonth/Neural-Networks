import tensorflow as tf
import keras
from tensorflow.contrib.slim import nets

slim = tf.contrib.slim
from keras.datasets.cifar10 import load_data


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test


def scheduler(epoch):
    if epoch < 81:
        return 0.1
    if epoch < 122:
        return 0.01
    return 0.001


class ShowProcess():
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """
    i = 0  # 当前的处理进度
    max_steps = 0  # 总共需要处理的次数
    max_arrow = 50  # 进度条的长度
    infoDone = 'done'

    # 初始化函数，需要知道总共的处理次数
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 0

    # 显示函数，根据当前的处理进度i显示进度
    # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, info, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps)  # 计算显示多少个'>'
        num_line = self.max_arrow - num_arrow  # 计算显示多少个'-'
        percent = self.i * 100.0 / self.max_steps  # 计算完成进度，格式为xx.xx%
        process_bar = '\r[' + '>' * num_arrow + '-' * num_line + ']' \
                      + '%.2f' % percent + '%  ' + info  # 带输出的字符串，'\r'表示不换行回到最左边
        print(process_bar, end='')  # 这两句打印字符到终端
        if self.i > self.max_steps:
            self.close()

    def close(self):
        print("\n")  # 训练完一行记录之后跳转到下一行
        self.i = 0


batch_size = 64
import numpy as np

(x_train, y_train), (x_test, y_test) = load_data()  # 50000, 32,32,3
y_train = np.array(keras.utils.to_categorical(y_train, 10).tolist())
y_test = np.array(keras.utils.to_categorical(y_test, 10).tolist())
x_train, x_test = color_preprocessing(x_train, x_test)
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool, shape=[])

with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
    net, endpoints = nets.resnet_v2.resnet_v2_50(x, 10, is_training=is_training)

net = tf.squeeze(net, axis=[1, 2])
# net = slim.fully_connected(net, num_outputs=10,  activation_fn=None, scope='Predict')

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=net))

global_step = slim.create_global_step()
learning_rate = tf.train.exponential_decay(0.000005, global_step, 400000, 0.9)  # 0.000001迭代150轮
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net, 1), tf.argmax(y, 1)), dtype=tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# saver = tf.train.Saver()
# saver.restore(sess, "sample_data/model")
n_epochs = 200
iterations = 50000 // batch_size + 1
p = ShowProcess(iterations)

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(x_train)

for epoch_i in range(n_epochs):
    result = None
    average_result = 0.0
    average_loss = 0.0
    count = 0
    k = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=batch_size):
        result = sess.run([optimizer, accuracy, loss], feed_dict={x: x_batch, y: y_batch, is_training: True})
        average_result += result[1]
        average_loss += result[2]
        p.show_process('epoch: {}, step: {}, loss: {:.3f}, acc: {:.2f}'.format(epoch_i + 1,
                                                                               k + 1, result[2], result[1], 3), k)
        k += 1
        if k == iterations: break

    average_result /= iterations
    average_loss /= iterations
    if epoch_i % 10 == 0:
        saver = tf.train.Saver()
        save_path = saver.save(sess, "sample_data/model")
    index = np.random.randint(10000, size=(1000,))
    result = sess.run([optimizer, accuracy, loss], feed_dict={x: x_test[index], y: y_test[index], is_training: False})
    info = "epoch: {}, step:{}, average-loss: {:.3f}, average-acc: {:.2f}, val_loss: {:.3f}, val_acc: {:.2f}".format(
        epoch_i,
        k + 1,
        average_loss,
        average_result,
        result[2],
        result[1])
    p.show_process(info, iterations)
    p.close()
