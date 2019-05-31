from __future__ import print_function
from __future__ import absolute_import
import keras
import tensorflow as tf
import keras.backend as K

import backbone
weight_decay = 1e-4


class ModelMGPU(keras.Model):
    def __init__(self, ser_model, gpus):
        pmodel = keras.utils.multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


def amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


def stutter_model(input_dim=(257, 250, 1), num_class=8631, mode='train', args=None):
    net=args.net
    loss=args.loss
    bottleneck_dim=args.bottleneck_dim
    mgpu = len(keras.backend.tensorflow_backend._get_available_gpus())

    if net == 'resnet34s':
        inputs, x = backbone.resnet_2D_v1(input_dim=input_dim, mode=mode)
    else:
        inputs, x = backbone.resnet_2D_v2(input_dim=input_dim, mode=mode)


    # ===============================================
    #            Fully Connected Block 2
    # ===============================================
    x.add(keras.layers.Dense(bottleneck_dim, activation='relu',
                           kernel_initializer='orthogonal',
                           use_bias=True, trainable=True,
                           kernel_regularizer=keras.regularizers.l2(weight_decay),
                           bias_regularizer=keras.regularizers.l2(weight_decay),
                           name='fc6_deepid3'))

    # ===============================================
    #            Softmax Vs AMSoftmax
    # ===============================================
    x.add(keras.layers.Dense(num_class, activation='softmax',
                           kernel_initializer='orthogonal',
                           use_bias=False, trainable=True,
                           kernel_regularizer=keras.regularizers.l2(weight_decay),
                           bias_regularizer=keras.regularizers.l2(weight_decay),
                           name='prediction'))
    trnloss = 'categorical_crossentropy'

    model = x

    if mode == 'train':
        if mgpu > 1:
            model = ModelMGPU(model, gpus=mgpu)
        # set up optimizer.
        if args.optimizer == 'adam':  opt = keras.optimizers.Adam(lr=1e-3)
        elif args.optimizer =='sgd':  opt = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True)
        elif args.optimizer =='rmsprop':  opt = keras.optimizers.RMSprop(lr=0.1, rho=0.9, decay=0.0)
        else: raise IOError('==> unknown optimizer type')
        model.compile(optimizer=opt, loss=trnloss, metrics=['acc'])

    # model.summary()
    # exit()
    return model
