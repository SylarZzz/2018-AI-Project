import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import imageio
import os


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def read_data(file_name):
    imageList = []
    labelList = []

    name1 = "/Users/wenjiezhang/Desktop/Fruit-Images-Dataset-master/"
    name = os.listdir(name1 + file_name + "/")
    name.sort()
    count = len(name)

    for i in range(count):
        listdir = os.listdir(name1 + file_name + "/" + name[i])
        count2 = len(listdir)

        for j in range(count2):
            img = imageio.imread(
                name1
                + file_name + "/" + name[i] + "/" + listdir[j])
            img = img / 255.0
            imageList.append(img)

            a = np.zeros(45)
            a[i] = 1
            labelList.append(a)

    print(len(imageList))
    print(len(labelList))

    return imageList, labelList


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1.})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1.})
    return result


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")


def weights_variables(shape):
    initial = tf.truncated_normal(shape, stddev=1e-2, dtype=tf.float32)
    return tf.Variable(initial)


def biases_variables(shape):
    initial = tf.constant(shape, dtype=tf.float32)
    return tf.Variable(initial)


def max_pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def batch_norm(inputs):
    return tf.layers.batch_normalization(inputs=inputs, training=True)


def conv_v1(inputs, out_size):
    shortcut = inputs

    inputs1 = slim.conv2d(inputs, out_size, kernel_size=1, stride=1, padding='SAME')
    inputs1 = batch_norm(inputs1)
    #inputs1 = tf.nn.relu(inputs1)

    inputs2 = slim.conv2d(inputs1, out_size, kernel_size=3, stride=1, padding='SAME')
    inputs2 = batch_norm(inputs2)
    #inputs2 = tf.nn.relu(inputs2)

    output = tf.nn.relu(shortcut + inputs2)

    return output


def conv_v2(inputs, out_size):
    shortcut = inputs

    inputs1 = slim.conv2d(inputs, out_size, kernel_size=3, stride=2, padding='SAME')
    inputs1 = batch_norm(inputs1)
    #inputs1 = tf.nn.relu(inputs1)

    inputs2 = slim.conv2d(inputs1, out_size, kernel_size=3, stride=1, padding='SAME')
    inputs2 = batch_norm(inputs2)
    #inputs2 = tf.nn.relu(inputs2)

    shortcut = slim.conv2d(shortcut, out_size, kernel_size=3, stride=2, padding='SAME')
    shortcut = batch_norm(shortcut)

    output = tf.nn.relu(shortcut + inputs2)

    return output


xs = tf.placeholder(tf.float32, [None, 100, 100, 3]) / 255.
ys = tf.placeholder(tf.float32, [None, 45])
keep_prob = tf.placeholder(tf.float32)

# res-34
conv1 = slim.conv2d(xs, 64, kernel_size=7, stride=2, padding='SAME')
conv1 = batch_norm(conv1)
conv1_r = tf.nn.relu(conv1)
conv1 = tf.nn.max_pool(conv1_r, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

conv2_1 = conv_v1(conv1_r, 64)
conv2_3 = conv_v1(conv2_1, 64)

conv3_1 = conv_v2(conv2_3, 128)
conv3_4 = conv_v1(conv3_1, 128)

conv4_1 = conv_v2(conv3_4, 256)
conv4_6 = conv_v1(conv4_1, 256)

conv5_1 = conv_v2(conv4_6, 512)
conv5_3 = conv_v1(conv5_1, 512)

global_pool = tf.nn.avg_pool(conv5_3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

W_fc1 = weights_variables([4 * 4 * 512, 100])
b_fc1 = biases_variables([100])
h5_pooling_flat = tf.reshape(global_pool, [-1, 4 * 4 * 512])

fc1_h = tf.nn.tanh(batch_norm(tf.matmul(h5_pooling_flat, W_fc1) + b_fc1))
fc1_h_dropout = tf.nn.dropout(fc1_h, keep_prob)

W_fc2 = weights_variables([100, 45])
b_fc2 = biases_variables([45])
prediction = tf.nn.softmax(tf.matmul(fc1_h_dropout, W_fc2) + b_fc2)

#lnr = tf.train.exponential_decay(15e-3, global_step=10000, decay_steps=100, decay_rate=0.90, staircase=True)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction, 0, 1.0)),
                                              reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


init = tf.global_variables_initializer()
sess.run(init)

train_images, train_labels = read_data("Training")
indexList = np.arange(len(train_images), dtype=np.dtype(int))
np.random.shuffle(indexList)
batch_size = 50


test_images, test_labels = read_data("Testing")
indexList_t = np.arange(len(test_images), dtype=np.dtype(int))
np.random.shuffle(indexList_t)
batch_size_t = 50


for epoch in range(10000):
    for i in range(int(len(train_images) / batch_size)):
        if (i + 1) * batch_size < len(train_images):
            batch_xs = np.array(train_images)[indexList[i * batch_size: (i + 1) * batch_size]]
            batch_ys = np.array(train_labels)[indexList[i * batch_size: (i + 1) * batch_size]]
            __, pred = sess.run([train_step, prediction], feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.6})
            print(i, compute_accuracy(batch_xs, batch_ys))
        else:
            batch_xs = np.array(train_images)[indexList[i * batch_size:]]
            batch_ys = np.array(train_labels)[indexList[i * batch_size:]]
            __, pred = sess.run([train_step, prediction], feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.6})
            print(i, compute_accuracy(batch_xs, batch_ys))

    pre_right = 0

    for j in range(int(len(test_images) / batch_size_t)):
        if (j + 1) * batch_size_t < len(test_images):
            batch_xt = np.array(test_images)[indexList_t[j * batch_size_t: (j + 1) * batch_size_t]]
            batch_yt = np.array(test_labels)[indexList_t[j * batch_size_t: (j + 1) * batch_size_t]]
            pre_right += compute_accuracy(batch_xt, batch_yt) * batch_size_t

        else:
            batch_xt = np.array(test_images)[indexList_t[j * batch_size_t:]]
            batch_yt = np.array(test_labels)[indexList_t[j * batch_size_t:]]
            pre_right += compute_accuracy(batch_xt, batch_yt) * batch_xt.shape[0]

    test_acc = pre_right / len(test_images)
    print("Epoch:{} Test_error: {:.10f}".format(epoch, test_acc))
