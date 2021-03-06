from ResNet import ResNet
import argparse
from utils import *

import horovod.tensorflow as hvd

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of ResNet"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--dataset', type=str, default='cifar10', help='[cifar10, cifar100, mnist, fashion-mnist, tiny, imagenet')
    parser.add_argument('--epoch', type=int, default=1, help='The number of epochs to run')
    parser.add_argument('--iteration', type=int, default=200, help='The number of interations in total')
    parser.add_argument('--batch_size', type=int, default=256, help='The size of batch per gpu')
    parser.add_argument('--res_n', type=int, default=18, help='18, 34, 50, 101, 152')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--amp', action='store_true', help='amp')
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)
    # --result_dir
    check_folder(args.log_dir)
    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')
    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""
def main():
    hvd.init()

    # parse arguments
    args = check_args(parse_args())
    if args is None:
      exit()

    cnn = ResNet(args)
    # build graph
    cnn.build_model()
    # show network architecture
    show_all_variables()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())


    hooks = [hvd.TimelineHook()]

    with tf.train.MonitoredTrainingSession(config=config, hooks=hooks) as sess:
        sess = hvd.TimelineSession(sess)

        if args.phase == 'train' :
            # launch the graph in a session
            cnn.train(sess)

            print(" [*] Training finished! \n")


if __name__ == '__main__':
    main()