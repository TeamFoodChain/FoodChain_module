import tensorflow as tf
import numpy as np
from fc_moduleserver import preprocess, im_module1


FLAGS = tf.app.flags.FLAGS
n_classes = 8

def test(img_path):
    with tf.device('/device:GPU:0'):
        with tf.Graph().as_default():
            x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
            y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])

            im_model = im_module1.VGG19(32, n_classes, 0.001)
            prob = im_model.build(x)

            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prob, labels=y))
            prediction = tf.arg_max(prob, 1)

            saver = tf.train.Saver()

            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
                saver.restore(sess, './fc_moduleserver/checkpoint_momentum/149_model.ckpt')
                test_data = preprocess.get_test_tensor(32, img_path)
                data = test_data.__next__()
                bx = data[0]
                by = data[1]
                probability, loss, predict = sess.run([prob, cross_entropy, prediction],
                                                      feed_dict={x:bx, y:by})
                print('probability : %s ' %probability[0])
                print('prediction : %s' %predict[0])
                for label, id in preprocess.subclass_dict.items():
                    if id == predict[0]:
                        return label

    return None

