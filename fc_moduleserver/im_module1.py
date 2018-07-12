import numpy as np
import tensorflow as tf

class VGG19:
    def __init__(self, batch_size, n_classes, learning_rate):
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.IS_PRETRAIN = True

    def build(self, x):
        self.conv1_1 = self.conv('conv1_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=self.IS_PRETRAIN)
        self.conv1_2 = self.conv('conv1_2', self.conv1_1, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=self.IS_PRETRAIN)
        with tf.name_scope('pool1'):
            self.pool1 = self.pool('pool1', self.conv1_2, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        self.conv2_1 = self.conv('conv2_1', self.pool1, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=self.IS_PRETRAIN)
        self.conv2_2 = self.conv('conv2_2', self.conv2_1, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=self.IS_PRETRAIN)
        with tf.name_scope('pool2'):
            self.pool2 = self.pool('pool2', self.conv2_2, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        self.conv3_1 = self.conv('conv3_1', self.pool2, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=self.IS_PRETRAIN)
        self.conv3_2 = self.conv('conv3_2', self.conv3_1, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=self.IS_PRETRAIN)
        self.conv3_3 = self.conv('conv3_3', self.conv3_2, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=self.IS_PRETRAIN)
        self.conv3_4 = self.conv('conv3_4', self.conv3_3, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=self.IS_PRETRAIN)
        with tf.name_scope('pool3'):
            self.pool3 = self.pool('pool3', self.conv3_4, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        self.conv4_1 = self.conv('conv4_1', self.pool3, 512, kernel_size=[3, 3], stride=[1, 2, 2, 1], is_pretrain=self.IS_PRETRAIN)
        self.conv4_2 = self.conv('conv4_2', self.conv4_1, 512, kernel_size=[3, 3], stride=[1, 2, 2, 1], is_pretrain=self.IS_PRETRAIN)
        self.conv4_3 = self.conv('conv4_3', self.conv4_2, 512, kernel_size=[3, 3], stride=[1, 2, 2, 1], is_pretrain=self.IS_PRETRAIN)
        self.conv4_4 = self.conv('conv4_4', self.conv4_3, 512, kernel_size=[3, 3], stride=[1, 2, 2, 1], is_pretrain=self.IS_PRETRAIN)
        with tf.name_scope('pool4'):
            self.pool4 = self.pool('pool4', self.conv4_4, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        self.conv5_1 = self.conv('conv5_1', self.pool4, 512, kernel_size=[3, 3], stride=[1, 2, 2, 1], is_pretrain=self.IS_PRETRAIN)
        self.conv5_2 = self.conv('conv5_2', self.conv5_1, 512, kernel_size=[3, 3], stride=[1, 2, 2, 1], is_pretrain=self.IS_PRETRAIN)
        self.conv5_3 = self.conv('conv5_3', self.conv5_2, 512, kernel_size=[3, 3], stride=[1, 2, 2, 1], is_pretrain=self.IS_PRETRAIN)
        self.conv5_4 = self.conv('conv5_4', self.conv5_3, 512, kernel_size=[3, 3], stride=[1, 2, 2, 1], is_pretrain=self.IS_PRETRAIN)
        with tf.name_scope('pool5'):
            self.pool5 = self.pool('pool5', self.conv5_4, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        self.fc6 = self.FC_layer('fc6', self.pool5, out_nodes=4096)
        with tf.name_scope('batch_norma1'):
            self.fc6 = self.batch_norm(self.fc6)  # batch norm can avoid overfit, more efficient than dropout
        self.fc7 = self.FC_layer('fc7', self.fc6, out_nodes=4096)
        with tf.name_scope('batch_norm2'):
            self.fc7 = self.batch_norm(self.fc7)
        self.fc8 = self.FC_layer('fc8', self.fc7, out_nodes=self.n_classes)
        self.prob = tf.nn.softmax(self.fc8, name='prob')
        return self.prob

    def conv(self, layer_name, x, out_channels, kernel_size=None,  stride=None, is_pretrain=True):
        """
        Convolution op wrapper, the Activation id ReLU
        :param layer_name: layer name, eg: conv1, conv2, ...
        :param x: input tensor, size = [batch_size, height, weight, channels]
        :param out_channels: number of output channel (convolution kernel)
        :param kernel_size: convolution kernel size, VGG use [3,3]
        :param stride: paper default = [1,1,1,1]
        :param is_pretrain: whether you need pre train, if you get parameter from other, you don not want to train again,
                            so trainable = false. if not trainable = true
        :return: 4D tensor
        """
        kernel_size = kernel_size if kernel_size else [3, 3]
        stride = stride if stride else [1, 1, 1, 1]

        in_channels = x.get_shape()[-1]

        with tf.variable_scope(layer_name):
            w = tf.get_variable(name="weights",
                                shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer(),
                                trainable=is_pretrain)
            b = tf.get_variable(name='biases',
                                shape=[out_channels],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0.0),
                                trainable=is_pretrain)
            x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
            x = tf.nn.bias_add(x, b, name='bias_add')
            x = tf.nn.relu(x, name='relu')

            return x

    def pool(self, layer_name, x, ksize=None, stride=None, is_max_pool=True):
        """
        Pooling op
        :param layer_name: layer name, eg:pool1, pool2,...
        :param x:input tensor
        :param ksize:pool kernel size, VGG paper use [1,2,2,1], the size of 2X2
        :param stride:stride size, VGG paper use [1,2,2,1]
        :param is_max_pool: default use max pool, if it is false, the we will use avg_pool
        :return: tensor
        """
        ksize = ksize if ksize else [1, 2, 2, 1]
        stride = stride if stride else [1, 2, 2, 1]

        if is_max_pool:
            x = tf.nn.max_pool(x, ksize, strides=stride, padding='SAME', name=layer_name)
        else:
            x = tf.nn.avg_pool(x, ksize, strides=stride, padding='SAME', name=layer_name)

        return x

    def batch_norm(self, x):
        """
        Batch Normalization (offset and scale is none). BN algorithm can improve train speed heavily.
        :param x: input tensor
        :return: norm tensor
        """
        epsilon = 1e-3
        batch_mean, batch_var = tf.nn.moments(x, [0])
        x = tf.nn.batch_normalization(x,
                                      mean=batch_mean,
                                      variance=batch_var,
                                      offset=None,
                                      scale=None,
                                      variance_epsilon=epsilon)

        return x

    def FC_layer(self, layer_name, x, out_nodes):
        """
        Wrapper for fully-connected layer with ReLU activation function
        :param layer_name: FC layer name, eg: 'FC1', 'FC2', ...
        :param x: input tensor
        :param out_nodes: number of neurons for FC layer
        :return: tensor
        """
        shape = x.get_shape()
        if len(shape) == 4:
            size = shape[1].value * shape[2].value * shape[3].value
        else:
            size = shape[-1].value

        with tf.variable_scope(layer_name):
            w = tf.get_variable('weights',
                                shape=[size, out_nodes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('biases',
                                shape=[out_nodes],
                                initializer=tf.constant_initializer(0.0))
            # flatten into 1D
            flat_x = tf.reshape(x, [-1, size])

            x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
            x = tf.nn.relu(x)

            return x

    def loss(self, logits, labels):
        """
        Compute loss
        :param logits: logits tensor, [batch_size, n_classes]
        :param labels: one_hot labels
        :return:
        """
        with tf.name_scope('loss') as scope:
            # use softmax_cross_entropy_with_logits(), so labels must be one-hot coding
            # if use sparse_softmax_cross_entropy_with_logits(), the labels not be one-hot
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels,
                                                                       name='cross-entropy')
            loss_temp = tf.reduce_mean(cross_entropy, name='loss')
            tf.summary.scalar(scope + '/loss', loss_temp)

            return loss_temp

    def optimize(self, loss, learning_rate, global_step):
        """
        optimization, use Gradient Descent as default
        :param loss:
        :param learning_rate:
        :param global_step:
        :return:
        """
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op

    def accuracy(self, logits, labels):
        """
        Evaluate quality of the logits at predicting labels
        :param logits: logits tensor, [batch_size, n_class]
        :param labels: labels tensor
        :return:
        """
        with tf.name_scope('accuracy') as scope:
            correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
            correct = tf.cast(correct, tf.float32)
            accuracy_temp = tf.reduce_mean(correct) * 100.0
            tf.summary.scalar(scope + '/accuracy', accuracy_temp)

            return accuracy_temp

    def num_correct_prediction(self, logits, labels):
        """
        Evaluate quality of the logits at predicting labels
        :param logits:
        :param labels:
        :return: number of correct prediction
        """
        correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
        correct = tf.cast(correct, tf.int32)
        n_correct = tf.reduce_sum(correct)

        return n_correct

    def load(self, data_path, session):
        """
        load the VGG16_pretrain parameters file
        :param data_path:
        :param session:
        :return:
        """
        data_dict = np.load(data_path, encoding='latin1').item()

        keys = sorted(data_dict.keys())
        for key in keys:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    session.run(tf.get_variable(subkey).assign(data))

    def load_with_skip(self, data_path, session, skip_layer):
        """
        Only load some layer parameters
        :param data_path:
        :param session:
        :param skip_layer:
        :return:
        """
        data_dict = np.load(data_path, encoding='latin1').item()

        for key in data_dict:
            if key not in skip_layer:
                with tf.variable_scope(key, reuse=True):
                    for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                        session.run(tf.get_variable(subkey).assign(data))

    def test_load(self):
        """
        test load vgg16.npy, print the shape of data
        :return:
        """
        data_path = './/VGG16_pretrain//vgg16.npy'

        data_dict = np.load(data_path, encoding='latin1').item()
        keys = sorted(data_dict.keys())
        for key in keys:
            weights = data_dict[key][0]
            biases = data_dict[key][1]
            print('\n')
            print(key)
            print('weights shape: ', weights.shape)
            print('biases shape: ', biases.shape)

    def print_all_variables(self, train_only=True):
        """Print all trainable and non-trainable variables
        without tl.layers.initialize_global_variables(sess)
        Parameters
        ----------
        train_only : boolean
            If True, only print the trainable variables, otherwise, print all variables.
        """
        # tvar = tf.trainable_variables() if train_only else tf.all_variables()
        if train_only:
            t_vars = tf.trainable_variables()
            print("  [*] printing trainable variables")
        else:
            try:  # TF1.0
                t_vars = tf.global_variables()
            except:  # TF0.12
                t_vars = tf.all_variables()
            print("  [*] printing global variables")
        for idx, v in enumerate(t_vars):
            print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))
