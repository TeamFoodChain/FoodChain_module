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

                saver.restore(sess, 'put your model in here')
                test_data = preprocess.get_test_tensor(32, img_path)
                data = test_data.__next__()
                bx = data[0]
                by = data[1]
                probability, loss, predict = sess.run([prob, cross_entropy, prediction],
                                                      feed_dict={x:bx, y:by})
                # print('probability : %s ' %probability[0])
                # print('prediction : %s' %predict[0])
                # print(preprocess.superclass_dict)
                for label, id in preprocess.subclass_dict.items():
                    if id == predict[0]:
                        # print('id : %d' %id)
                        if id == 0 or id == 1:
                            superclass = preprocess.superclass_dict[0]
                            print('%s, %s' %(label, superclass))
                            return label, superclass
                        elif id == 2 or id == 3:
                            superclass = preprocess.superclass_dict[1]
                            print('%s, %s' % (label, superclass))
                            return label, superclass
                        elif id == 4 or id == 5:
                            superclass = preprocess.superclass_dict[2]
                            print('%s, %s' % (label, superclass))
                            return label, superclass
                        elif id == 6 or id == 7:
                            superclass = preprocess.superclass_dict[3]
                            print('%s, %s' % (label, superclass))
                            return label, superclass

    return None

