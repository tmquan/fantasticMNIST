import argparse
import glob2
import numpy as np
import os
import cv2
import tensorflow as tf
from natsort import natsorted

from tensorpack import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.utils.viz import stack_patches

from GAN import GANModelDesc, GANTrainer

# tf.compat.v1.disable_eager_execution()

class ImageMNISTFromFolder(RNGDataFlow):
    # https://github.com/tensorpack/tensorpack/blob/master/tensorpack/dataflow/image.py
    """ Produce images read from a list of files as (h, w, c) arrays. """
    def __init__(self, folder, is_train=False, channel=1, resize=None, shuffle=False):
        """
        Args:
            files (list): list of file paths.
            channel (int): 1 or 3. Will convert grayscale to RGB images if channel==3.
                Will produce (h, w, 1) array if channel==1.
            resize (tuple): int or (h, w) tuple. If given, resize the image.
        """
        assert folder, "No folder given to ImageFromFolder!"
        self.folder = folder
        self.channel = int(channel)
        assert self.channel in [1, 3], self.channel
        self.imread_mode = cv2.IMREAD_GRAYSCALE if self.channel == 1 else cv2.IMREAD_COLOR
        if resize is not None:
            resize = shape2d(resize)
        self.resize = resize
        self.shuffle = shuffle
        self.is_train = is_train
        if self.is_train:
            self.folder = os.path.join(self.folder, 'train')
        else:
            self.folder = os.path.join(self.folder, 'valid')
        self.files = natsorted(glob2.glob(self.folder+'/*/*.png'))
        
    def __len__(self):
        # print(self.folder, self.files)
        return len(self.files)

    def size(self):
        return len(self.files)

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.files)

        

        for _ in self.files:
            rng_class = self.rng.randint(10)
            a_class = rng_class     
            b_class = (a_class + 1) % 10
            a_folder = os.path.join(self.folder, str(a_class)) # train/0
            b_folder = os.path.join(self.folder, str(a_class)) # train/0
            a_files = natsorted(glob2.glob(a_folder+'/*.png'))
            b_files = natsorted(glob2.glob(b_folder+'/*.png'))

            a_file = self.rng.choice(a_files)
            b_file = self.rng.choice(b_files)
            # print(a_file, b_file)

            assert a_file is not None and b_file is not None

            a_image = cv2.imread(a_file, self.imread_mode)
            b_image = cv2.imread(b_file, self.imread_mode)

            yield a_image, b_image #, a_class, b_class



BATCH = 1
# IN_CH = 3
# OUT_CH = 3
LAMBDA = 100
NF = 64  # number of filter


def BNLReLU(x, name=None):
    x = BatchNorm('bn', x)
    return tf.nn.leaky_relu(x, alpha=0.2, name=name)


def visualize_tensors(name, imgs, scale_func=lambda x: (x + 1.) * 128., max_outputs=1):
    """Generate tensor for TensorBoard (casting, clipping)
    Args:
        name: name for visualization operation
        *imgs: multiple tensors as list
        scale_func: scale input tensors to fit range [0, 255]
    Example:
        visualize_tensors('viz1', [img1])
        visualize_tensors('viz2', [img1, img2, img3], max_outputs=max(30, BATCH))
    """
    xy = scale_func(tf.concat(imgs, axis=2))
    xy = tf.cast(tf.clip_by_value(xy, 0, 255), tf.uint8, name='viz')
    tf.summary.image(name, xy, max_outputs=30)


class Model(GANModelDesc):
    def inputs(self):
        SHAPE = 32
        return [tf.TensorSpec((None, SHAPE, SHAPE, 1), tf.float32, 'a_image'),
                tf.TensorSpec((None, SHAPE, SHAPE, 1), tf.float32, 'b_image'), 
                ]

    def generator(self, imgs):
        # imgs: input: 256x256xch
        # U-Net structure, it's slightly different from the original on the location of relu/lrelu
        with argscope(BatchNorm, training=True), \
                argscope(Dropout, is_training=True):
            # always use local stat for BN, and apply dropout even in testing
            # with argscope(Conv2D, kernel_size=4, strides=2, activation=BNLReLU):
            #     e1 = Conv2D('conv1', imgs, NF, activation=tf.nn.leaky_relu)
            #     e2 = Conv2D('conv2', e1, NF * 2)
            #     e3 = Conv2D('conv3', e2, NF * 4)
            #     e4 = Conv2D('conv4', e3, NF * 8)
            #     e5 = Conv2D('conv5', e4, NF * 8)
            #     e6 = Conv2D('conv6', e5, NF * 8)
            #     # e7 = Conv2D('conv7', e6, NF * 8)
            #     e8 = Conv2D('conv8', e6, NF * 8, activation=BNReLU)  # 1x1
            # with argscope(Conv2DTranspose, activation=BNReLU, kernel_size=4, strides=2):
            #     return (LinearWrap(e8)
            #             # .Conv2DTranspose('deconv1', NF * 8)
            #             # .Dropout()
            #             # .ConcatWith(e7, 3)
            #             .Conv2DTranspose('deconv2', NF * 8)
            #             .Dropout()
            #             # .ConcatWith(e6, 3)
            #             .Conv2DTranspose('deconv3', NF * 8)
            #             .Dropout()
            #             # .ConcatWith(e5, 3)
            #             .Conv2DTranspose('deconv4', NF * 8)
            #             # .ConcatWith(e4, 3)
            #             .Conv2DTranspose('deconv5', NF * 4)
            #             # .ConcatWith(e3, 3)
            #             .Conv2DTranspose('deconv6', NF * 2)
            #             # .ConcatWith(e2, 3)
            #             .Conv2DTranspose('deconv7', NF * 1)
            #             # .ConcatWith(e1, 3)
            #             .Conv2DTranspose('deconv8', 1, activation=tf.tanh)())
            with argscope(Conv2D, kernel_size=4, strides=1, activation=BNLReLU):
                logits = (LinearWrap(imgs)
                      .Conv2D('conv1_1', 64)
                      .Conv2D('conv1_2', 64, strides=2)
                      # .MaxPooling('pool1', 2)
                      # 112
                      .Conv2D('conv2_1', 128)
                      .Conv2D('conv2_2', 128, strides=2)
                      # .MaxPooling('pool2', 2)
                      # 56
                      .Conv2D('conv3_1', 256)
                      .Conv2D('conv3_2', 256)
                      .Conv2D('conv3_3', 256, strides=2)
                      # .MaxPooling('pool3', 2)
                      # 28
                      .Conv2D('conv4_1', 512)
                      .Conv2D('conv4_2', 512)
                      .Conv2D('conv4_3', 512, strides=2)
                      # .MaxPooling('pool4', 2)
                      # 14
                      .Conv2D('conv5_1', 512)
                      .Conv2D('conv5_2', 512)
                      .Conv2D('conv5_3', 512, strides=2)
                      # .MaxPooling('pool5', 2)
                      ())
            with argscope(Conv2DTranspose, activation=BNReLU, kernel_size=4, strides=1):
                output =  (LinearWrap(logits)
                      .Conv2DTranspose('deconv5_1', 512)
                      .Conv2DTranspose('deconv5_2', 512)
                      .Conv2DTranspose('deconv5_3', 512, strides=2)
                      .Conv2DTranspose('deconv4_1', 512)
                      .Conv2DTranspose('deconv4_2', 512)
                      .Conv2DTranspose('deconv4_3', 512, strides=2)
                      .Conv2DTranspose('deconv3_1', 256)
                      .Conv2DTranspose('deconv3_2', 256)
                      .Conv2DTranspose('deconv3_3', 256, strides=2)
                      .Conv2DTranspose('deconv2_1', 128)
                      .Conv2DTranspose('deconv2_2', 128, strides=2)
                      .Conv2DTranspose('deconv1_1', 64)
                      .Conv2DTranspose('deconv1_2', 64, strides=2)
                      .Conv2DTranspose('deconv0', 1, activation=tf.tanh)
                      ())
                return output

    @auto_reuse_variable_scope
    def discriminator(self, inputs, outputs):
        """ return a (b, 1) logits"""
        l = tf.concat([inputs, outputs], 3)
        with argscope(Conv2D, kernel_size=4, strides=1, activation=BNLReLU):
            l = (LinearWrap(l)
                 # .Conv2D('conv0', NF, activation=tf.nn.leaky_relu)
                 # .Conv2D('conv1', NF * 2)
                 # .Conv2D('conv2', NF * 4)
                 # .Conv2D('conv3', NF * 8, strides=1, padding='VALID')
                 # .Conv2D('convlast', 1, strides=1, padding='VALID', activation=tf.identity)())
                .Conv2D('conv1_1', 64, activation=tf.nn.leaky_relu)
                .Conv2D('conv1_2', 64, strides=2)
                # .MaxPooling('pool1', 2)
                # 112
                .Conv2D('conv2_1', 128)
                .Conv2D('conv2_2', 128, strides=2)
                # .MaxPooling('pool2', 2)
                # 56
                .Conv2D('conv3_1', 256)
                .Conv2D('conv3_2', 256)
                .Conv2D('conv3_3', 256, strides=2)
                # .MaxPooling('pool3', 2)
                # 28
                .Conv2D('conv4_1', 512)
                .Conv2D('conv4_2', 512)
                .Conv2D('conv4_3', 512, strides=2)
                # .MaxPooling('pool4', 2)
                # 14
                .Conv2D('conv5_1', 512)
                .Conv2D('conv5_2', 512)
                .Conv2D('conv5_3', 512, strides=2)
                 # .Conv2D('conv3', NF * 8, strides=1, padding='VALID')
                .Conv2D('convlast', 1, strides=1, padding='SAME', activation=tf.identity)())
        return l

    def build_graph(self, input, output):
        input, output = input / 128.0 - 1, output / 128.0 - 1

        with argscope([Conv2D, Conv2DTranspose], 
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('gen'):
                fake_output = self.generator(input)
            with tf.variable_scope('discrim'):
                real_pred = self.discriminator(input, output)
                fake_pred = self.discriminator(input, fake_output)

        self.build_losses(real_pred, fake_pred)
        errL1 = tf.reduce_mean(tf.abs(fake_output - output), name='L1_loss')
        self.g_loss = tf.add(self.g_loss, LAMBDA * errL1, name='total_g_loss')
        add_moving_summary(errL1, self.g_loss)

        # # tensorboard visualization
        # if IN_CH == 1:
        #     input = tf.image.grayscale_to_rgb(input)
        # if OUT_CH == 1:
        #     output = tf.image.grayscale_to_rgb(output)
        #     fake_output = tf.image.grayscale_to_rgb(fake_output)

        visualize_tensors('input,output,fake', [input, output, fake_output], max_outputs=max(30, BATCH))

        self.collect_variables()

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=2e-4, trainable=False)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)


def get_data():
    datadir = args.data
    # imgs = glob.glob(os.path.join(datadir, '*.jpg'))
    # ds = ImageFromFile(imgs, channel=3, shuffle=True)

    # ds = MapData(ds, lambda dp: split_input(dp[0]))
    # augs = [imgaug.Resize(286), imgaug.RandomCrop(256)]
    # ds = AugmentImageComponents(ds, augs, (0, 1))
    # ds = BatchData(ds, BATCH)
    # ds = MultiProcessRunner(ds, 100, 1)
    ds_train = ImageMNISTFromFolder(folder=datadir, is_train=True)
    ds_valid = ImageMNISTFromFolder(folder=datadir, is_train=False)
    ds_train = BatchData(ds_train, BATCH)
    ds_valid = BatchData(ds_valid, 1)
    # ds_train = MultiProcessRunner(ds_train, 100, 1)
    return ds_train, ds_valid


def sample(datadir, model_path):
    pred = PredictConfig(
        session_init=SmartInit(model_path),
        model=Model(),
        input_names=['input', 'output'],
        output_names=['viz'])

    imgs = glob.glob(os.path.join(datadir, '*.jpg'))
    ds = ImageFromFile(imgs, channel=3, shuffle=True)
    ds = MapData(ds, lambda dp: split_input(dp[0]))
    ds = AugmentImageComponents(ds, [imgaug.Resize(256)], (0, 1))
    ds = BatchData(ds, 6)

    pred = SimpleDatasetPredictor(pred, ds)
    for o in pred.get_result():
        o = o[0][:, :, :, ::-1]
        stack_patches(o, nr_row=3, nr_col=2, viz=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', action='store_true', help='run sampling')
    parser.add_argument('--data', help='Image directory', required=True)
    parser.add_argument('--mode', choices=['AtoB', 'BtoA'], default='AtoB')
    parser.add_argument('-b', '--batch', type=int, default=32)
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    BATCH = args.batch

    if args.sample:
        assert args.load
        sample(args.data, args.load)
    else:
        logger.auto_set_dir()
        ds_train, ds_valid = get_data()

        # ds_train = QueueInput(ds_train)
        # ds_valid = QueueInput(ds_valid)
        ds_train = PrintData(ds_train)
        trainer = GANTrainer(ds_train, Model(), get_num_gpu())
        
        trainer.train_with_defaults(
            callbacks=[
                PeriodicTrigger(ModelSaver(), every_k_epochs=3),
                ScheduledHyperParamSetter('learning_rate', [(200, 1e-4)])
            ],
            steps_per_epoch=ds_train.size(),
            max_epoch=300,
            session_init=SmartInit(args.load)
        )