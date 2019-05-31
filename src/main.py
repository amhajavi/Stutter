from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import keras
import numpy as np

sys.path.append('../tool')
import toolkits

# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--person', default='0', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--spec_len', default='', type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--data_path', default='/home/amh/Projects/Stutter/data/Clips', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--bottleneck_dim', default=512, type=int)
# set up learning rate, training loss and optimizer.
parser.add_argument('--epochs', default=56, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--warmup_ratio', default=0, type=float)
parser.add_argument('--loss', default='amsoftmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--optimizer', default='rmsprop', choices=['adam', 'sgd','rmsprop'], type=str)
parser.add_argument('--ohem_level', default=0, type=int,
                    help='pick hard samples from (ohem_level * batch_size) proposals, must be > 1')
global args
args = parser.parse_args()

def main():

    # gpu configuration
    toolkits.initialize_GPU(args)

    import model
    import generator

    # ==================================
    #       Get Train/Val.
    # ==================================
    trnlist, trnlb = toolkits.get_datalist(args, path='../meta/Stutter_Leave_One_Repetition/train_labels_{}.txt'.format(args.person))
    vallist, vallb = toolkits.get_datalist(args, path='../meta/Stutter_Leave_One_Repetition/validation_labels_{}.txt'.format(args.person))

    # construct the data generator.
    params = {'dim': (257, args.spec_len, 1),
              'mp_pooler': toolkits.set_mp(processes=12),
              'nfft': 512,
              'spec_len': args.spec_len,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 2,
              'sampling_rate': 16000,
              'batch_size': args.batch_size,
              'shuffle': True,
              'normalize': True,
              }

    # Datasets
    partition = {'train': trnlist.flatten(), 'val': vallist.flatten()}
    labels = {'train': trnlb.flatten(), 'val': vallb.flatten()}

    # Generators
    trn_gen = generator.DataGenerator(partition['train'], labels['train'], **params)
    vld_gen = generator.DataGenerator(partition['val'], labels['val'], **params)

    network = model.stutter_model(input_dim=params['dim'],
                                           num_class=params['n_classes'],
                                           mode='train', args=args)

    # ==> load pre-trained model ???
    mgpu = len(keras.backend.tensorflow_backend._get_available_gpus())
    if args.resume:
        if os.path.isfile(args.resume):
            if mgpu == 1: network.load_weights(os.path.join(args.resume), by_name=True)
            else: network.layers[mgpu + 1].load_weights(os.path.join(args.resume),  by_name=True)
            print('==> successfully loading model {}.'.format(args.resume))
        else:
            print("==> no checkpoint found at '{}'".format(args.resume))

    print(network.summary())

    print('==> gpu {} is, training {} images, classes: 0-{} '
          'loss: {}, ohemlevel: {}'.format(args.gpu, len(partition['train']), np.max(labels['train']),
                                                            args.loss, args.ohem_level))

    model_path, log_path = set_path(args)

    normal_lr = keras.callbacks.LearningRateScheduler(step_decay)

    tbcallbacks = keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True, write_images=False,
                                              update_freq=args.batch_size * 16)

    callbacks = [keras.callbacks.ModelCheckpoint(os.path.join(model_path, 'weights-{epoch:02d}-{val_acc:.3f}.h5'),
                                                 monitor='val_acc',
                                                 mode='max',
                                                 save_best_only=True),
                 normal_lr, tbcallbacks]

    if args.ohem_level > 1:     # online hard negative mining will be used
        candidate_steps = int(len(partition['train']) // args.batch_size)
        iters_per_epoch = int(len(partition['train']) // (args.ohem_level*args.batch_size))

        ohem_generator = generator.OHEM_generator(network,
                                                  trn_gen,
                                                  candidate_steps,
                                                  args.ohem_level,
                                                  args.batch_size,
                                                  params['dim'],
                                                  params['n_classes']
                                                  )


        A = ohem_generator.next()   # for some reason, I need to warm up the generator

        network.fit_generator(generator.OHEM_generator(network, trn_gen, iters_per_epoch,
                                                       args.ohem_level, args.batch_size,
                                                       params['dim'], params['n_classes']),
                              steps_per_epoch=iters_per_epoch,
                              epochs=args.epochs,
                              max_queue_size=10,
                              callbacks=callbacks,
                              validation_data=vld_gen,
                              use_multiprocessing=False,
                              workers=1,
                              verbose=1)

    else:
        network.fit_generator(trn_gen,
                              steps_per_epoch=int(len(partition['train'])//args.batch_size),
                              epochs=args.epochs,
                              max_queue_size=10,
                              callbacks=callbacks,
                              use_multiprocessing=False,
                              validation_data= vld_gen,
                              workers=1,
                              verbose=1)


def step_decay(epoch):
    '''
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every step epochs.
    '''
    half_epoch = args.epochs // 2
    stage1, stage2, stage3 = int(half_epoch * 0.5), int(half_epoch * 0.8), half_epoch
    stage4 = stage3 + stage1
    stage5 = stage4 + (stage2 - stage1)
    stage6 = args.epochs

    if args.warmup_ratio:
        milestone = [2, stage1, stage2, stage3, stage4, stage5, stage6]
        gamma = [args.warmup_ratio, 1.0, 0.1, 0.01, 1.0, 0.1, 0.01]
    else:
        milestone = [stage1, stage2, stage3, stage4, stage5, stage6]
        gamma = [1.0, 0.1, 0.01, 1.0, 0.1, 0.01]

    lr = 0.005
    init_lr = args.lr
    stage = len(milestone)
    for s in range(stage):
        if epoch < milestone[s]:
            lr = init_lr * gamma[s]
            break
    print('Learning rate for epoch {} is {}.'.format(epoch + 1, lr))
    return np.float(lr)


def set_path(args):
    import datetime
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    exp_path = os.path.join('repitition/person_{args.person}'.format(date, args=args))
    model_path = os.path.join('../model', exp_path)
    log_path = os.path.join('../log', exp_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    if not os.path.exists(log_path): os.makedirs(log_path)
    return model_path, log_path


def get_prediction():

    toolkits.initialize_GPU(args)

    import model
    import generator

    params = {'dim': (257, args.spec_len, 1),
    'mp_pooler': toolkits.set_mp(processes=12),
    'nfft': 512,
    'spec_len': args.spec_len,
    'win_length': 400,
    'hop_length': 160,
    'n_classes': 2,
    'sampling_rate': 16000,
    'batch_size': args.batch_size,
    'shuffle': False,
    'normalize': True,
    }

    network = model.stutter_model(input_dim=params['dim'],
                                       num_class=params['n_classes'],
                                       mode='train', args=args)

    personal_folder = '/home/amh/Projects/Stutter/model/Filler/person_{}/'.format(args.person)
    personal_files = os.listdir(personal_folder)[-1]
    best_model = os.path.join(personal_folder, personal_files)
    print(best_model)
    network.load_weights(best_model)
    vallist, vallb = toolkits.get_datalist(args, path='../meta/Stutter_Leave_One_Fillers/validation_labels_{}.txt'.format(args.person))
    vld_gen = generator.DataGenerator(vallist.flatten(), vallb.flatten(), **params)
    step = 0
    preds = []
    for i in range((len(vallist)//args.batch_size)+1):
        x_data, y_data = vld_gen.__getitem__(index=step+i)
        preds.extend(np.argmax(network.predict(x_data),axis=1))
    from sklearn.metrics import accuracy_score

    print(accuracy_score(preds, vallb))


if __name__ == "__main__":
    main()
    # get_prediction()
