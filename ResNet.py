import time
from ops import *
from utils import *
import horovod.tensorflow as hvd

class ResNet(object):
    def __init__(self, args):
        self.model_name = 'ResNet'
        self.dataset_name = args.dataset

        if self.dataset_name == 'cifar10' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_cifar10()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 10

        if self.dataset_name == 'cifar100' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_cifar100()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 100

        if self.dataset_name == 'mnist' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_mnist()
            self.img_size = 28
            self.c_dim = 1
            self.label_dim = 10

        if self.dataset_name == 'fashion-mnist' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_fashion()
            self.img_size = 28
            self.c_dim = 1
            self.label_dim = 10

        if self.dataset_name == 'tiny' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_tiny()
            self.img_size = 64
            self.c_dim = 3
            self.label_dim = 200

        if self.dataset_name == 'imagenet' :
            self.img_size = 224
            self.c_dim = 3
            self.label_dim = 1000
            self.train_x = np.array([1]*args.batch_size*self.img_size*self.img_size*self.c_dim).reshape((-1, self.img_size, self.img_size, self.c_dim))
            self.train_y = np.array([1]*args.batch_size*self.label_dim).reshape((-1, self.label_dim))
            self.test_x = self.train_x
            self.test_y = self.train_y


        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir

        self.res_n = args.res_n

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.iteration = len(self.train_x) // self.batch_size

        self.init_lr = args.lr

        self.max_iteration = args.iteration
        self.amp = args.amp


    ##################################################################################
    # Generator
    ##################################################################################

    def network(self, x, is_training=True, reuse=False):
        with tf.variable_scope("network", reuse=reuse):

            if self.res_n < 50 :
                residual_block = resblock
            else :
                residual_block = bottle_resblock

            residual_list = get_residual_layer(self.res_n)

            ch = 32 # paper is 64
            x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')

            for i in range(residual_list[0]) :
                x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')

            for i in range(1, residual_list[1]) :
                x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')

            for i in range(1, residual_list[2]) :
                x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')

            for i in range(1, residual_list[3]) :
                x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))

            ########################################################################################################


            x = batch_norm(x, is_training, scope='batch_norm')
            x = relu(x)

            x = global_avg_pooling(x)
            x = fully_conneted(x, units=self.label_dim, scope='logit')

            return x

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Graph Input """
        self.train_inptus = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='train_inputs')
        self.train_labels = tf.placeholder(tf.float32, [self.batch_size, self.label_dim], name='train_labels')

        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Model """
        self.train_logits = self.network(self.train_inptus)

        self.train_loss, self.train_accuracy = classification_loss(logit=self.train_logits, label=self.train_labels)

        reg_loss = tf.losses.get_regularization_loss()
        self.train_loss += reg_loss

        """ Training """
        lr_scaler = hvd.size()
        self.optim = tf.train.AdamOptimizer(0.001 * lr_scaler, epsilon=1e-8)
        # self.optim = tf.train.MomentumOptimizer(self.lr, momentum=0.9).minimize(self.train_loss)
        # auto mixed precision training
        if self.amp:
            self.optim = tf.train.experimental.enable_mixed_precision_graph_rewrite(self.optim)
        self.optim = hvd.DistributedOptimizer(self.optim, op=hvd.Average)

        gradients = self.optim.compute_gradients(self.train_loss)
        global_step = tf.train.get_or_create_global_step()
        self.train_op = self.optim.apply_gradients(gradients, global_step=global_step)

        """" Summary """
        self.summary_train_loss = tf.summary.scalar("train_loss", self.train_loss)
        self.summary_train_accuracy = tf.summary.scalar("train_accuracy", self.train_accuracy)
        self.train_summary = tf.summary.merge([self.summary_train_loss, self.summary_train_accuracy])

    ##################################################################################
    # Train
    ##################################################################################

    def train(self, _sess):
        # initialize all variables
        tf.global_variables_initializer().run()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, _sess.graph)

        epoch_lr = self.init_lr
        start_epoch = 0
        start_batch_id = 0
        counter = 1

        # loop for epoch
        start_time = time.time()
        epoch = 0
        while True:
            if epoch == int(self.epoch * 0.5) or epoch == int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.1

            # get batch data
            for idx in range(start_batch_id, self.iteration):
                batch_x = self.train_x[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_y = self.train_y[idx*self.batch_size:(idx+1)*self.batch_size]

                batch_x = data_augmentation(batch_x, self.img_size, self.dataset_name)

                train_feed_dict = {
                    self.train_inptus : batch_x,
                    self.train_labels : batch_y,
                    self.lr : epoch_lr
                }

                # update network
                _, summary_str, train_loss, train_accuracy = _sess.run(
                    [self.train_op, self.train_summary, self.train_loss, self.train_accuracy], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, train_accuracy: %.2f, test_accuracy: %.2f, learning_rate : %.4f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, train_accuracy, -1, epoch_lr))

                if counter > self.max_iteration:
                    break
            if counter > self.max_iteration:
                break
            epoch += 1
            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0
        print("Training speed: {} img/s".format(self.max_iteration * self.batch_size / (time.time() - start_time)))


    @property
    def model_dir(self):
        return "{}{}_{}_{}_{}".format(self.model_name, self.res_n, self.dataset_name, self.batch_size, self.init_lr)
