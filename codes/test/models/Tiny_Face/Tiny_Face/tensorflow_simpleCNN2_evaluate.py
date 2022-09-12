# -*- coding: utf-8 -*-
import cv2
from skimage import io, transform
import glob
import os
import tensorflow as tf
import numpy as np
import time
#Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2 报错忽略
import os
from scipy.special import expit
import matplotlib.pyplot as plt
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'


#path = 'D:/service2019/train/train_hatorperson'

model_path = 'models\Tiny_Face\model/test_model3'


w = 100
h = 100
c = 3



def read_img(path):
    cate = [path +'/'+ x for x in os.listdir(path) if os.path.isdir(path +'/'+ x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            print('reading the images:%s' % (im))
            img = io.imread(im)
            img = transform.resize(img, (w, h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


#data, label = read_img(path)


#num_example = data.shape[0]
#arr = np.arange(num_example)
#np.random.shuffle(arr)
#data = data[arr]
#label = label[arr]


#ratio = 0.8
#s = np.int(num_example * ratio)
#x_train = data[:s]
#y_train = label[:s]
#x_val = data[s:]
#y_val = label[s:]


x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')


conv1 = tf.layers.conv2d(
    inputs=x,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)


conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)


conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)


conv4 = tf.layers.conv2d(
    inputs=pool3,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])


dense1 = tf.layers.dense(inputs=re1,
                         units=1024,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
dense2 = tf.layers.dense(inputs=dense1,
                         units=512,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
logits = tf.layers.dense(inputs=dense2,
                         units=5,
                         activation=None,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))


loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


saver = tf.train.Saver()


'''
n_epoch = 10
batch_size = 1
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver.restore(sess, model_path)



# validation
val_loss, val_acc, n_batch = 0, 0, 0
for x_val_a, y_val_a in minibatches(data, label, batch_size, shuffle=False):
    #print(y_val_a)
    #print(x_val_a)
    err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: [1]})
    val_loss += err
    val_acc += ac
    n_batch += 1
print("   validation loss: %f" % (val_loss / n_batch))
print("   validation acc: %f" % (val_acc / n_batch))
'''

'''
def evaluate(image):
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)

    err, ac = sess.run([loss, acc], feed_dict={x: image, y_: [0]})
    sess.close()
    if ac == 0: return 'person'
    else: return 'hat'


def evaluate_all(img,boxes):
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)
    imgs = []
    final_boxes = []
    count = 0
    for i in boxes:
        img_head = img[int(i[2]):int(i[4]),int(i[1]):int(i[3])]
        imgs.append(img_head)
    for i in imgs:
        try:
            i = transform.resize(i, (w, h))
            kkk = []
            kkk.append(i)
            err, ac = sess.run([loss, acc], feed_dict={x: kkk, y_: [0]})
            if ac == 0: final_boxes.append(boxse[count])
        except:
            pass
        count += 1
    sess.close()
    return final_boxes
'''

def evaluate_mulboxes(mulboxes):
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)
    final_mulboxes = []
    for boxes in mulboxes:
        try:
            img = io.imread(boxes[0])
            final_boxes = []
            for box in boxes[2:len(boxes)]:
                img_head = img[int(box[2]):int(box[4]),int(box[1]):int(box[3])]
                img_head = transform.resize(img_head, (w, h))
                err, ac = sess.run([loss, acc], feed_dict={x: [img_head], y_: [0]})#[0] is hat
                if ac == 0: final_boxes.append(box)
        except: pass

        final_boxes.insert(0,boxes[1])
        final_boxes.insert(0,boxes[0])
        final_mulboxes.append(final_boxes)

    sess.close()
    return final_mulboxes

def cnn_addlabel(mulboxes):
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)
    final_mulboxes = []
    for boxes in mulboxes:
        try:
            img = io.imread(boxes[0])
            final_boxes = []
            for box in boxes[2:len(boxes)]:
                img_head = img[int(box[2]):int(box[4]),int(box[1]):int(box[3])]
                img_head = transform.resize(img_head, (w, h))
                scores = sess.run(logits,feed_dict={x: [img_head]})
                score_hat = expit(scores[0][0])
                score_person = expit(scores[0][1])
                if score_hat>score_person:
                    box.insert(1,'hat')
                    box.append(score_hat)
                else:
                    box.insert(1,'person')
                    box.append(score_person)
                final_boxes.append(box)
        except: pass
        final_mulboxes.append(final_boxes)
    sess.close()
    return final_mulboxes


def cnn_for_vedio(mulboxes,vedio_path):
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)
    cap = cv2.VideoCapture(vedio_path) 
    ret, frame = cap.read()
    counting = 0.0
    final_mulboxes = []
    for boxes in mulboxes:
        if(len(boxes)==0):
            continue
        if ret == False: break
        while ret and boxes[0][0] != int(counting):
            ret, frame = cap.read()
            counting += 1
        final_boxes = []
        for box in boxes:
            img_head = frame[int(box[2]):int(box[4]),int(box[1]):int(box[3])]
            img_head = cv2.cvtColor(img_head, cv2.COLOR_RGB2BGR)
            #plt.imshow(img_head)
            #plt.show()
            img_head = transform.resize(img_head, (w, h))
            scores = sess.run(logits,feed_dict={x: [img_head]})
            score_hat = expit(scores[0][0])
            score_person = expit(scores[0][1])
            if score_hat>score_person:
                box.insert(1,'hat')
                box.append(score_hat)
            else:
                box.insert(1,'person')
                box.append(score_person)
            final_boxes.append(box)
        final_mulboxes.append(final_boxes)
    cap.release()
    sess.close()
    return final_mulboxes

'''
#sess.close()
def main():
    im = 'D:/service2019/train/person/20.jpg'
    img = cv2.imread(im)
    img = transform.resize(img, (w, h))
    #kkk = []
    #kkk.append(img)
    print(evaluate([img]))

if __name__ == '__main__':
    main()
'''