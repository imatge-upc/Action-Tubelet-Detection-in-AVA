from __future__ import print_function
import os
import sys
import math

from copy import deepcopy

CAFFE_PYTHON_PATH = os.path.join(os.path.dirname(__file__), "../python")
sys.path.insert(0, CAFFE_PYTHON_PATH)

import caffe

from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from caffe.model_libs import UnpackVariable, ConvBNLayer

from Dataset import GetDataset


def ACT_ConvBNLayer(net, initial_name, from_layer, out_layer, use_bn, use_relu, num_output,
    kernel_size, pad, stride, dilation=1, use_scale=True, lr_mult=1, conv_prefix='', 
    conv_postfix='', bn_prefix='', bn_postfix='_bn', scale_prefix='', scale_postfix='_scale', 
    bias_prefix='', bias_postfix='_bias', m = '', **bn_params):
    
    if use_bn:

        # parameters for convolution layer with batchnorm.
        kwargs = {
            'param': [dict(lr_mult=lr_mult, decay_mult=1, name = initial_name + m + '_w')],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_term': False,
            }

        eps = bn_params.get('eps', 0.001)
        moving_average_fraction = bn_params.get('moving_average_fraction', 0.999)
        use_global_stats = bn_params.get('use_global_stats', False)

        # parameters for batchnorm layer.
        bn_kwargs = {
            'param': [
                dict(lr_mult=0, decay_mult=0),
                dict(lr_mult=0, decay_mult=0),
                dict(lr_mult=0, decay_mult=0)],
            'eps': eps,
            'moving_average_fraction': moving_average_fraction,
            }

        bn_lr_mult = lr_mult

        if use_global_stats:
            # only specify if use_global_stats is explicitly provided;
            # otherwise, use_global_stats_ = this->phase_ == TEST;
            bn_kwargs = {
                'param': [
                dict(lr_mult=0, decay_mult=0),
                dict(lr_mult=0, decay_mult=0),
                dict(lr_mult=0, decay_mult=0)],
                'eps': eps,
                'use_global_stats': use_global_stats,
            }

            # not updating scale/bias parameters
            bn_lr_mult = 0

        # parameters for scale bias layer after batchnorm.
        if use_scale:
            sb_kwargs = {
                'bias_term': True,
                'param': [
                dict(lr_mult=bn_lr_mult, decay_mult=0),
                dict(lr_mult=bn_lr_mult, decay_mult=0)],
                'filler': dict(type='constant', value=1.0),
                'bias_filler': dict(type='constant', value=0.0),
                }
        else:
            bias_kwargs = {
            'param': [dict(lr_mult=bn_lr_mult, decay_mult=0)],
            'filler': dict(type='constant', value=0.0),
            }
    else:
        kwargs = {
            'param': [
            dict(lr_mult=lr_mult, decay_mult=1, name = initial_name + m + "_w"),
            dict(lr_mult=2 * lr_mult, decay_mult=0, name = initial_name + m + "_b")], 
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)
            }


    conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
    [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
    [pad_h, pad_w] = UnpackVariable(pad, 2)
    [stride_h, stride_w] = UnpackVariable(stride, 2)

    if kernel_h == kernel_w:
        net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
    else:
        net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
            kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
            stride_h=stride_h, stride_w=stride_w, **kwargs)

    if dilation > 1:
        net.update(conv_name, {'dilation': dilation})

    if use_bn:
        bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
        net[bn_name] = L.BatchNorm(net[conv_name], in_place=True, **bn_kwargs)    
        if use_scale:
            sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
            net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
        else:
            bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
            net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
    
    if use_relu:
        relu_name = '{}_relu'.format(conv_name)
        net[relu_name] = L.ReLU(net[conv_name], in_place=True)
        
    
def ACT_DataLayer(dname, K, batch_size, resize_height=300, resize_width=300, restart_iter=0, flow=False, ninput=1):
    
    kwargs = {'include': dict(phase=caffe_pb2.Phase.Value('TRAIN'))}
    ntop = K + 1

    return L.Python(name="data", ntop=ntop, python_param={
        'module': 'ACT_datalayer', 
        'layer': 'MultiframesLayer', 
        'param_str': 
            '{"dataset_name": "%s", "K": %d, "batch_size": %d, "resize_height": %d, "resize_width": %d, "restart_iter": %d, "flow": %s, "ninput": %d}'
            %(dname, K, batch_size, resize_height, resize_width, restart_iter, str(flow), ninput)
        }, **kwargs)

    
def ACT_VGGNetBody(net, from_layer, K=6, need_fc=True, fully_conv=False, reduced=False, 
    dilated=False, nopool=False, dropout=True, freeze_layers=[], dilate_pool4=False, lr_mult=1.0, m=""):
    """ Generate the body of VGGNet
        K is the length of the tubelet and of the input sequence
        m is the modality, if m == '', then the modality is RGB; if m == 'flow', the modality is flow
    """  
    assert K >= 1
    assert m in ['', 'flow']
    # from_layer --> input data
    for stream in xrange(K):
        assert all([from_layer + '_stream' + str(stream) + m in net.keys()])

    kwargs_initial = {
            'param': [dict(lr_mult=1*lr_mult, decay_mult=1), dict(lr_mult=2*lr_mult, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)
            }

    ################################################ conv 1 ######################################################
    for stream in xrange(K):
        kwargs = deepcopy(kwargs_initial)
        kwargs['param'] = [dict(lr_mult=1*lr_mult, decay_mult=1, name='conv1_1' + m + '_w'), dict(lr_mult=2*lr_mult, decay_mult=0, name='conv1_1' + m + '_b')]
        net['conv1_1_stream' + str(stream) + m ] = L.Convolution(net[from_layer + '_stream' + str(stream) + m], num_output=64, pad=1, kernel_size=3, **kwargs)
    
    for stream in xrange(K):
        kwargs = deepcopy(kwargs_initial)
        kwargs['param'] = [dict(lr_mult=1*lr_mult, decay_mult=1, name='conv1_2' + m + '_w'), dict(lr_mult=2*lr_mult, decay_mult=0, name='conv1_2' + m + '_b')]
        net['relu1_1_stream' + str(stream) + m] = L.ReLU(net['conv1_1_stream' + str(stream) + m], in_place=True)     
        net['conv1_2_stream' + str(stream) + m] = L.Convolution(net['relu1_1_stream' + str(stream) + m], num_output=64, pad=1, kernel_size=3, **kwargs)
        net['relu1_2_stream' + str(stream) + m] = L.ReLU(net['conv1_2_stream' + str(stream) + m], in_place=True) 

    for stream in xrange(K):
        if nopool:
            name = 'conv1_3_stream' + str(stream) + m            
            kwargs = deepcopy(kwargs_initial)
            kwargs['param'] = [dict(lr_mult=1*lr_mult, decay_mult=1, name='conv1_3' + m + '_w'), dict(lr_mult=2*lr_mult, decay_mult=0, name='conv1_3' + m + '_b')]
            net[name] = L.Convolution(net['relu1_2_stream' + str(stream) + m], num_output=64, pad=1, kernel_size=3, stride=2, **kwargs)
        else:            
            name = 'pool1_stream' + str(stream) + m
            net[name] = L.Pooling(net['relu1_2_stream' + str(stream) + m], pool=P.Pooling.MAX, kernel_size=2, stride=2)
    
    ################################################ conv 2 ######################################################

    for stream in xrange(K):
        kwargs = deepcopy(kwargs_initial)
        kwargs['param'] = [dict(lr_mult=1*lr_mult, decay_mult=1, name='conv2_1' + m + '_w'), dict(lr_mult=2*lr_mult, decay_mult=0, name='conv2_1' + m + '_b')]
        if nopool:
            name = 'conv1_3_stream' + str(stream) + m
        else:
            name = 'pool1_stream' + str(stream) + m       
        net['conv2_1_stream' + str(stream) + m] = L.Convolution(net[name], num_output=128, pad=1, kernel_size=3, **kwargs)
        net['relu2_1_stream' + str(stream) + m] = L.ReLU(net['conv2_1_stream' + str(stream) + m], in_place=True)
    
    for stream in xrange(K):
        kwargs = deepcopy(kwargs_initial)
        kwargs['param'] = [dict(lr_mult=1*lr_mult, decay_mult=1, name='conv2_2' + m + '_w'), dict(lr_mult=2*lr_mult, decay_mult=0, name='conv2_2' + m + '_b')]
        net['conv2_2_stream' + str(stream) + m] = L.Convolution(net['relu2_1_stream' + str(stream) + m], num_output=128, pad=1, kernel_size=3, **kwargs)
        net['relu2_2_stream' + str(stream) + m] = L.ReLU(net['conv2_2_stream' + str(stream) + m], in_place=True)

    for stream in xrange(K):
        if nopool:
            name = 'conv2_3_stream' + str(stream) + m            
            kwargs = deepcopy(kwargs_initial)
            kwargs['param'] = [dict(lr_mult=1*lr_mult, decay_mult=1, name='conv2_3' + m + '_w'), dict(lr_mult=2*lr_mult, decay_mult=0, name='conv2_3' + m + '_b')]
            net[name] = L.Convolution(net['relu2_2_stream' + str(stream) + m], num_output=128, pad=1, kernel_size=3, stride=2, **kwargs)
        else:            
            name = 'pool2_stream' + str(stream) + m
            net[name] = L.Pooling(net['relu2_2_stream' + str(stream) + m], pool=P.Pooling.MAX, kernel_size=2, stride=2) 

    ################################################ conv 3 ######################################################

    for stream in xrange(K):
        kwargs = deepcopy(kwargs_initial)
        kwargs['param'] = [dict(lr_mult=1*lr_mult, decay_mult=1, name='conv3_1' + m + '_w'), dict(lr_mult=2*lr_mult, decay_mult=0, name='conv3_1' + m + '_b')]
        if nopool:
            name = 'conv2_3_stream' + str(stream) + m
        else:
            name = 'pool2_stream' + str(stream) + m       
        net['conv3_1_stream' + str(stream) + m] = L.Convolution(net[name], num_output=256, pad=1, kernel_size=3, **kwargs)
        net['relu3_1_stream' + str(stream) + m] = L.ReLU(net['conv3_1_stream' + str(stream) + m], in_place=True)

    for stream in xrange(K):
        kwargs = deepcopy(kwargs_initial)
        kwargs['param'] = [dict(lr_mult=1*lr_mult, decay_mult=1, name='conv3_2' + m + '_w'), dict(lr_mult=2*lr_mult, decay_mult=0, name='conv3_2' + m + '_b')]
        net['conv3_2_stream' + str(stream) + m] = L.Convolution(net['relu3_1_stream' + str(stream) + m], num_output=256, pad=1, kernel_size=3, **kwargs)
        net['relu3_2_stream' + str(stream) + m] = L.ReLU(net['conv3_2_stream' + str(stream) + m], in_place=True) 

    for stream in xrange(K):
        kwargs = deepcopy(kwargs_initial)
        kwargs['param'] = [dict(lr_mult=1*lr_mult, decay_mult=1, name='conv3_3' + m + '_w'), dict(lr_mult=2*lr_mult, decay_mult=0, name='conv3_3' + m + '_b')]
        net['conv3_3_stream' + str(stream) + m] = L.Convolution(net['relu3_2_stream' + str(stream) + m], num_output=256, pad=1, kernel_size=3, **kwargs)
        net['relu3_3_stream' + str(stream) + m] = L.ReLU(net['conv3_3_stream' + str(stream) + m], in_place=True) 
    
    for stream in xrange(K):
        if nopool:
            name = 'conv3_4_stream' + str(stream) + m            
            kwargs = deepcopy(kwargs_initial)
            kwargs['param'] = [dict(lr_mult=1*lr_mult, decay_mult=1, name='conv3_4' + m + '_w'), dict(lr_mult=2*lr_mult, decay_mult=0, name='conv3_4' + m + '_b')]
            net[name] = L.Convolution(net['relu3_3_stream' + str(stream) + m], num_output=256, pad=1, kernel_size=3, stride=2, **kwargs)
        else:            
            name = 'pool3_stream' + str(stream) + m
            net[name] = L.Pooling(net['relu3_3_stream' + str(stream) + m], pool=P.Pooling.MAX, kernel_size=2, stride=2) 

    ################################################ conv 4 #####################################################
    for stream in xrange(K):
        kwargs = deepcopy(kwargs_initial)
        kwargs['param'] = [dict(lr_mult=1*lr_mult, decay_mult=1, name='conv4_1' + m + '_w'), dict(lr_mult=2*lr_mult, decay_mult=0, name='conv4_1' + m + '_b')]
        if nopool:
            name = 'conv3_4_stream' + str(stream) + m
        else:
            name = 'pool3_stream' + str(stream) + m       
        net['conv4_1_stream' + str(stream) + m] = L.Convolution(net[name], num_output=512, pad=1, kernel_size=3, **kwargs)
        net['relu4_1_stream' + str(stream) + m] = L.ReLU(net['conv4_1_stream' + str(stream) + m], in_place=True)

    for stream in xrange(K):
        kwargs = deepcopy(kwargs_initial)
        kwargs['param'] = [dict(lr_mult=1*lr_mult, decay_mult=1, name='conv4_2' + m + '_w'), dict(lr_mult=2*lr_mult, decay_mult=0, name='conv4_2' + m + '_b')]
        net['conv4_2_stream' + str(stream) + m] = L.Convolution(net['relu4_1_stream' + str(stream) + m], num_output=512, pad=1, kernel_size=3, **kwargs)
        net['relu4_2_stream' + str(stream) + m] = L.ReLU(net['conv4_2_stream' + str(stream) + m], in_place=True) 

    for stream in xrange(K):
        kwargs = deepcopy(kwargs_initial)
        kwargs['param'] = [dict(lr_mult=1*lr_mult, decay_mult=1, name='conv4_3' + m + '_w'), dict(lr_mult=2*lr_mult, decay_mult=0, name='conv4_3' + m + '_b')]
        net['conv4_3_stream' + str(stream) + m] = L.Convolution(net['relu4_2_stream' + str(stream) + m], num_output=512, pad=1, kernel_size=3, **kwargs)
        net['relu4_3_stream' + str(stream) + m] = L.ReLU(net['conv4_3_stream' + str(stream) + m], in_place=True) 

    for stream in xrange(K):
        if nopool:
            name = 'conv4_4_stream' + str(stream) + m            
            kwargs = deepcopy(kwargs_initial)
            kwargs['param'] = [dict(lr_mult=1*lr_mult, decay_mult=1, name='conv4_4' + m + '_w'), dict(lr_mult=2*lr_mult, decay_mult=0, name='conv4_4' + m + '_b')]
            net[name] = L.Convolution(net['relu4_3_stream' + str(stream) + m], num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
        else:            
            name = 'pool4_stream' + str(stream) + m
            if dilate_pool4:
                net[name] = L.Pooling(net['relu4_3_stream' + str(stream) + m], pool=P.Pooling.MAX, kernel_size=3, stride=1, pad=1)
                dilation = 2
            else:
                net[name] = L.Pooling(net['relu4_3_stream' + str(stream) + m], pool=P.Pooling.MAX, kernel_size=2, stride=2)
                dilation = 1

    kernel_size = 3
    pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2

    ################################################ conv 5 ######################################################
    for stream in xrange(K):
        kwargs = deepcopy(kwargs_initial)
        kwargs['param'] = [dict(lr_mult=1*lr_mult, decay_mult=1, name='conv5_1' + m + '_w'), dict(lr_mult=2*lr_mult, decay_mult=0, name='conv5_1' + m + '_b')]
        if nopool:
            name = 'conv4_4_stream' + str(stream) + m
        else:
            name = 'pool4_stream' + str(stream) + m       
        net['conv5_1_stream' + str(stream) + m] = L.Convolution(net[name], num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation, **kwargs)
        net['relu5_1_stream' + str(stream) + m] = L.ReLU(net['conv5_1_stream' + str(stream) + m], in_place=True)
    
    for stream in xrange(K):
        kwargs = deepcopy(kwargs_initial)
        kwargs['param'] = [dict(lr_mult=1*lr_mult, decay_mult=1, name='conv5_2' + m + '_w'), dict(lr_mult=2*lr_mult, decay_mult=0, name='conv5_2' + m + '_b')]
        net['conv5_2_stream' + str(stream) + m] = L.Convolution(net['relu5_1_stream' + str(stream) + m], num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation, **kwargs)
        net['relu5_2_stream' + str(stream) + m] = L.ReLU(net['conv5_2_stream' + str(stream) + m], in_place=True) 
    
    for stream in xrange(K):
        kwargs = deepcopy(kwargs_initial)
        kwargs['param'] = [dict(lr_mult=1*lr_mult, decay_mult=1, name='conv5_3' + m + '_w'), dict(lr_mult=2*lr_mult, decay_mult=0, name='conv5_3' + m + '_b')]
        net['conv5_3_stream' + str(stream) + m] = L.Convolution(net['relu5_2_stream' + str(stream) + m], num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation, **kwargs)
        net['relu5_3_stream' + str(stream) + m] = L.ReLU(net['conv5_3_stream' + str(stream) + m], in_place=True) 


    ################################################ FC ######################################################
    ######## fc layers ########
    if need_fc:
        # here starts if for K
        for stream in xrange(K):
            ################################################ conv 5_4 , pool5 ######################################################         
            ######### dilated #########
            if dilated:
                if nopool:
                    name = 'conv5_4_stream' + str(stream) + m    
                    kwargs = deepcopy(kwargs_initial)
                    kwargs['param'] = [dict(lr_mult=1*lr_mult, decay_mult=1, name='conv5_4' + m + '_w'), dict(lr_mult=2*lr_mult, decay_mult=0, name='conv5_4' + m + '_b')]
                    net[name] = L.Convolution(net['relu5_3_stream' + str(stream) + m], num_output=512, pad=1, kernel_size=3, stride=1, **kwargs)
                else:
                    name = 'pool5_stream' + str(stream) + m
                    net[name] = L.Pooling(net['relu5_3_stream' + str(stream) + m], pool=P.Pooling.MAX, kernel_size=2, stride=1)
            else:
                if nopool:
                    name = 'conv5_4_stream' + str(stream) + m    
                    kwargs = deepcopy(kwargs_initial)
                    kwargs['param'] = [dict(lr_mult=1*lr_mult, decay_mult=1, name='conv5_4' + m + '_w'), dict(lr_mult=2*lr_mult, decay_mult=0, name='conv5_4' + m + '_b')]
                    net[name] = L.Convolution(net['relu5_3_stream' + str(stream) + m], num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
                else:
                    name = 'pool5_stream' + str(stream) + m
                    net[name] = L.Pooling(net['relu5_3_stream' + str(stream) + m], pool=P.Pooling.MAX, kernel_size=2, stride=2)
            ###########################
            
            ################################################ fc6 ######################################################
            if fully_conv:
                ######### dilated #########
                if dilated:
                    if reduced:
                        dilation = dilation * 6
                        kernel_size = 3
                        num_output = 1024
                    else:
                        dilation = dilation * 2
                        kernel_size = 7
                        num_output = 4096
                else:
                    if reduced:
                        dilation = dilation * 3
                        kernel_size = 3
                        num_output = 1024
                    else:
                        kernel_size = 7
                        num_output = 4096               
                ###########################

                pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2

                ######### fc6 + relu6 #########
                kwargs = deepcopy(kwargs_initial)
                kwargs['param'] = [dict(lr_mult=1*lr_mult, decay_mult=1, name='fc6' + m + '_w'), dict(lr_mult=2*lr_mult, decay_mult=0, name='fc6' + m + '_b')]
                net['fc6_stream' + str(stream) + m  ] = L.Convolution(net[name], num_output=num_output,  pad=pad, kernel_size=kernel_size, dilation=dilation, **kwargs)
                net['relu6_stream' + str(stream) + m] = L.ReLU(net['fc6_stream' + str(stream) + m], in_place=True) 
                ###########################

                if dropout:
                    # net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)
                    net['drop6_stream' + str(stream) + m] = L.Dropout(net['relu6_stream' + str(stream) + m], dropout_ratio=0.5, in_place=True)

                ################################################ fc7 ######################################################
                kwargs = deepcopy(kwargs_initial)
                kwargs['param'] = [dict(lr_mult=1*lr_mult, decay_mult=1, name='fc7' + m + '_w'), dict(lr_mult=2*lr_mult, decay_mult=0, name='fc7' + m + '_b')]
                if reduced:
                    net['fc7_stream' + str(stream) + m  ]  = L.Convolution(net['relu6_stream' + str(stream) + m], num_output=1024, kernel_size=1, **kwargs)
                else:
                    net['fc7_stream' + str(stream) + m  ]  = L.Convolution(net['relu6_stream' + str(stream) + m], num_output=4096, kernel_size=1, **kwargs)

                # net.relu7 = L.ReLU(net.fc7, in_place=True)
                net['relu7_stream' + str(stream) + m] = L.ReLU(net['fc7_stream' + str(stream) + m], in_place=True) 
                if dropout:
                    # net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)
                    net['drop7_stream' + str(stream) + m] = L.Dropout(net['relu7_stream' + str(stream) + m], dropout_ratio=0.5, in_place=True)

            else:
                ################################################ fc6 ######################################################
                net['fc6_stream' + str(stream) + m  ] = L.InnerProduct(net['pool5_stream' + str(stream) + m], num_output=4096)
                net['relu6_stream' + str(stream) + m] = L.ReLU(net['fc6_stream' + str(stream) + m], in_place=True) 
                if dropout:
                    net['drop6_stream' + str(stream) + m] = L.Dropout(net['relu6_stream' + str(stream) + m], dropout_ratio=0.5, in_place=True)
                ################################################ fc7 ######################################################
                net['fc7_stream' + str(stream) + m  ] = L.InnerProduct(net['relu6_stream' + str(stream) + m], num_output=4096)
                net['relu7_stream' + str(stream) + m] = L.ReLU(net['fc7_stream' + str(stream) + m], in_place=True) 
                if dropout:
                    net['drop7_stream' + str(stream) + m] = L.Dropout(net['relu7_stream' + str(stream) + m], dropout_ratio=0.5, in_place=True)
    ###########################
    
    ################################################ Update freeze layers ###################################################### 
    kwargs_initial['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    for freeze_layer in freeze_layers:
        kwargs = deepcopy(kwargs_initial)
        kwargs['param'] = [dict(lr_mult=0, decay_mult=0, name=freeze_layer + m + '_w'), dict(lr_mult=0, decay_mult=0, name=freeze_layer + m + '_b')]
        for stream in xrange(K):
            if freeze_layer + '_stream' + str(stream) + m in layers:
                net.update(freeze_layer + '_stream' + str(stream) + m, kwargs)

    
    return net
    

def ACT_AddExtraLayers300(net, K=6, use_batchnorm=True, lr_mult=1.0, m=''):
    """Add extra layers on top of a "base" network (e.g. VGGNet or Inception).
        K is the length of the tubelet and of the input sequence
        m is the modality, if m == '', then the modality is RGB; if m == 'flow', the modality is flow
    """  

    use_relu = True
    # Add additional convolutional layers.
    # 19 x 19
    from_layer_initial = net.keys()[-1]
    from_layer_initial = from_layer_initial.split('_stream')[0]

    for stream in xrange(K):
        from_layer = from_layer_initial
        initial_name = 'conv6_1'
        out_layer = initial_name +'_stream' + str(stream) + m
        ACT_ConvBNLayer(net, initial_name, from_layer + '_stream' + str(stream) + m, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1, lr_mult=lr_mult, m=m)

        from_layer = out_layer
        initial_name = 'conv6_2'
        out_layer = initial_name +'_stream' + str(stream) + m
        ACT_ConvBNLayer(net, initial_name, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2, lr_mult=lr_mult, m=m)

        # 5 x 5
        from_layer = out_layer
        initial_name = 'conv7_1'
        out_layer = initial_name +'_stream' + str(stream) + m
        ACT_ConvBNLayer(net, initial_name, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1, lr_mult=lr_mult, m=m)

        from_layer = out_layer
        initial_name = 'conv7_2'
        out_layer = initial_name +'_stream' + str(stream) + m
        ACT_ConvBNLayer(net, initial_name,from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2, lr_mult=lr_mult, m=m)

        # 3 x 3
        from_layer = out_layer
        initial_name = 'conv8_1'
        out_layer = initial_name +'_stream' + str(stream) + m
        ACT_ConvBNLayer(net, initial_name, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1, lr_mult=lr_mult, m=m)

        from_layer = out_layer
        initial_name = 'conv8_2'
        out_layer = initial_name +'_stream' + str(stream) + m
        ACT_ConvBNLayer(net, initial_name, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1, lr_mult=lr_mult, m=m)

        # 1 x 1
        from_layer = out_layer
        initial_name = 'conv9_1'
        out_layer = initial_name +'_stream' + str(stream) + m
        ACT_ConvBNLayer(net, initial_name,from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1, lr_mult=lr_mult, m=m)

        from_layer = out_layer
        initial_name = 'conv9_2'
        out_layer = initial_name +'_stream' + str(stream) + m
        ACT_ConvBNLayer(net, initial_name, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1, lr_mult=lr_mult, m=m)

    return net    
    
    
def ACT_CreateCuboidHead(net, K=6, data_layer="data", num_classes=[], from_layers=[], 
    normalizations=[], use_batchnorm=True, lr_mult=1, use_scale=True, min_sizes=[], 
    max_sizes=[], prior_variance = [0.1], aspect_ratios=[], steps=[], img_height=0, 
    img_width=0, share_location=True, flip=True, clip=True, offset=0.5, kernel_size=1, pad=0,
    conf_postfix='', loc_postfix='', m='', fusion="concat", **bn_param):
    
            
    ##################### 3 change it!!! #######################################
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    
    if aspect_ratios:
        assert len(from_layers) == len(aspect_ratios), "from_layers and aspect_ratios should have same length"
    
    if steps:
        assert len(from_layers) == len(steps), "from_layers and steps should have same length"
    
    net_layers = net.keys()
    
    assert data_layer in net_layers, "data_layer is not in net's layers"


    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []

    for i in range(0, num):
        from_layer = from_layers[i]

        # Get the normalize value.
        if normalizations:
            if normalizations[i] != -1:
                for stream in xrange(K):
                    norm_name = "{}_norm_stream{}{}".format(from_layer, stream, m)
                    net[norm_name] = L.Normalize(net[from_layer + '_stream' + str(stream) + m], scale_filler=dict(type="constant", value=normalizations[i]),
                        across_spatial=False, channel_shared=False)
                from_layer = "{}_norm".format(from_layer)

        # ACT: add a concatenation layer across streams
        if fusion == "concat":
            net[from_layer + '_concat'] = L.Concat( bottom=[from_layer + '_stream' + str(stream) + m for stream in xrange(K)], axis=1)
            from_layer += '_concat'
        else:
            assert fusion == "sum"
            net[from_layer + '_sum'] = L.EltWise( bottom=[from_layer + '_stream' + str(stream) + m for stream in xrange(K)])
            from_layer += '_sum'

        # Estimate number of priors per location given provided parameters.
        min_size = min_sizes[i]
        if type(min_size) is not list:
            min_size = [min_size]
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        max_size = []
        if len(max_sizes) > i:
            max_size = max_sizes[i]
            if type(max_size) is not list:
                max_size = [max_size]
            if max_size:
                assert len(max_size) == len(min_size), "max_size and min_size should have same length."
        if max_size:
            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)
        else:
            num_priors_per_location = (1 + len(aspect_ratio)) * len(min_size)
        if flip:
            num_priors_per_location += len(aspect_ratio) * len(min_size)
        step = []
        if len(steps) > i:
            step = steps[i]

        # ACT-detector: location prediction layer 
        # location prediction for K different frames
        name = "{}_mbox_loc{}".format(from_layer, loc_postfix)
        num_loc_output = num_priors_per_location * 4 * K
        if not share_location:
            num_loc_output *= num_classes
                    
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_loc_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])

        # ACT-detector: confidence prediction layer
        # joint prediction of all frames
        name = "{}_mbox_conf{}".format(from_layer, conf_postfix)
        num_conf_output = num_priors_per_location * num_classes;
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_conf_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layers.append(net[flatten_name])

        # Create prior generation layer.
        name = "{}_mbox_priorbox".format(from_layer)
        net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_size,
                clip=clip, variance=prior_variance, offset=offset)
        if max_size:
            net.update(name, {'max_size': max_size})
        if aspect_ratio:
            net.update(name, {'aspect_ratio': aspect_ratio, 'flip': flip})
        if step:
            net.update(name, {'step': step})
        if img_height != 0 and img_width != 0:
            if img_height == img_width:
                net.update(name, {'img_size': img_height})
            else:
                net.update(name, {'img_h': img_height, 'img_w': img_width})
        priorbox_layers.append(net[name])

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_loc"
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])

    name = "mbox_conf"
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])

    name = "mbox_priorbox"
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])


    return mbox_layers
    
    
def ACT_generate_prototxt(dname, K=6, flow=False):
    """ Generates the train, test, deploy and solver prototxts for the datasets used in ACT-detector. 
        dname: 'UCFSports', 'JHMDB', 'JHMDB2', 'JHMDB3', 'UCF101', 'UCF101v2'
        K: length of the tubelet and input sequence. In ACT-detector K=6
        flow: if true, then use modality = FLOW5; if false, then modality = RGB
    """

    ######################### Frame PARAMS #########################
    IMGSIZE = 300

    ######################### General PARAMS #########################
    modality_str = 'flow' if flow else ''
    mode_str = 'FLOW5' if flow else 'RGB'

    ######################### Dataset PARAMS #########################
    dd = GetDataset(dname)
    num_classes = dd.nlabels + 1 # +1 for background
    if dname=='UCFSports':
        niter = 60000
        lr_steps = [40000, 55000]
    elif dname in ['JHMDB', 'JHMDB2', 'JHMDB3']:
        niter = 240000
        lr_steps = [160000, 220000]
    elif dname in ['UCF101', 'UCF101v2']:
        niter = 600000
        lr_steps = [400000, 550000]
    elif dname=='AVA':
        niter = 240000
        lr_steps = [160000, 220000]
    else:
        raise Exception("Unknown dataset " + dname)
    
    ######################### Model PATHS #########################
    #dirname = os.path.join(os.path.dirname(__file__), "..", "models", "ACT-detector", 'generated_' + dd.NAME)
    dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "ACT-detector", 'generated_' + dd.NAME))
    if not os.path.isdir(dirname): 
        os.system('mkdir -p ' + dirname)
    deploy_net_file = "{}/deploy_{}.prototxt".format(dirname, mode_str)
    train_net_file = "{}/train_{}.prototxt".format(dirname, mode_str)
    solver_file = "{}/solver_{}.prototxt".format(dirname, mode_str)
    model_name = "ACTdetector_{}_{}".format(dname, mode_str)
    # The pretrained model. 
    pretrain_model_file = os.path.join(dirname, "..", 'initialization_VGG_ILSVRC16_K{}_{}.caffemodel'.format(K, mode_str))

    ############ BATCH NORM PARAMS ######################
    # If true, use batch norm for all newly added layers.
    # Currently only the non batch norm version has been tested.
    use_batchnorm = False
    lr_mult = 1
    # Use different initial learning rate.
    if use_batchnorm:
        base_lr = 0.0004
    else:
        # A learning rate for batch_size = 1, num_gpus = 1.
        base_lr = 0.00004

    ############ MultiBoxLoss PARAMS ######################
    share_location = True
    background_label_id=0
    train_on_diff_gt = True
    normalization_mode = P.Loss.VALID
    code_type = P.PriorBox.CENTER_SIZE
    ignore_cross_boundary_bbox = False
    mining_type = P.MultiBoxLoss.MAX_NEGATIVE
    neg_pos_ratio = 3.
    loc_weight = (neg_pos_ratio + 1.) / 4.
    multibox_loss_param = {
        'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
        'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
        'loc_weight': loc_weight /float(K),
        'num_classes': num_classes,
        'share_location': share_location,
        'match_type': P.MultiBoxLoss.PER_PREDICTION,
        'overlap_threshold': 0.5,
        'use_prior_for_matching': True,
        'background_label_id': background_label_id,
        'use_difficult_gt': train_on_diff_gt,
        'neg_pos_ratio': neg_pos_ratio,
        'neg_overlap': 0.5,
        'code_type': code_type,
        }
    act_cuboid_loss_param = {
        'sequence_length': K,
    }
    multibox_loss_param['ignore_cross_boundary_bbox'] = ignore_cross_boundary_bbox
    multibox_loss_param['mining_type'] = mining_type
    loss_param = {
        'normalization': normalization_mode,
    }

    ############ PARAMS for generating PRIORS ######################
    # minimum dimension of input image
    min_dim = IMGSIZE
    mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
    # in percent %
    min_ratio = 20
    max_ratio = 90
    step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in xrange(min_ratio, max_ratio + 1, step):
        min_sizes.append(min_dim * ratio / 100.) 
        max_sizes.append(min_dim * (ratio + step) / 100.)
    min_sizes = ([min_dim * 10 / 100.] + min_sizes)
    max_sizes = ([min_dim * 20 / 100.] + max_sizes)
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    normalizations = [20, -1, -1, -1, -1, -1]
    steps = [8, 16, 32, 64, 100, 300] 
    # variance used to encode/decode prior bboxes.
    if code_type == P.PriorBox.CENTER_SIZE:
        prior_variance = [0.1, 0.1, 0.2, 0.2]
    else:
        prior_variance = [0.1]
    flip = True
    clip = False

    ############# GPU & SOLVER PARAMS ######################
    # Defining which GPUs to use.
    gpulist=[0]
    num_gpus = len(gpulist)

    # Divide the mini-batch to different GPUs.=
    batch_size = int(32 / K)
    accum_batch_size = batch_size
    iter_size = accum_batch_size / batch_size
    solver_mode = P.Solver.CPU
    device_id = 0
    batch_size_per_device = batch_size
    if num_gpus > 0:
        batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))
        iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
        solver_mode = P.Solver.GPU
        device_id = int(gpulist[0])

    if normalization_mode == P.Loss.NONE:
        base_lr /= batch_size_per_device
    elif normalization_mode == P.Loss.VALID:
        base_lr *= 25. / loc_weight
    elif normalization_mode == P.Loss.FULL:
        # Roughly there are 2000 prior bboxes per image.
        # TODO(weiliu89): Estimate the exact # of priors.
        base_lr *= 2000.

    # Which layers to freeze (no backward) during training.
    freeze_layers = []

    solver_param = {
        # Train parameters
        'base_lr': 0.0001,
        'weight_decay': 0.0005,
        'lr_policy': "multistep",
        'stepvalue': lr_steps,
        'gamma': 0.1,
        'momentum': 0.9,
        'max_iter': niter,
        'snapshot': 10000,
        'display': 10,
        'average_loss': 10,
        'type': "SGD",
        'solver_mode': solver_mode,
        'device_id': device_id,
        'debug_info': False,
        'snapshot_after_train': True,
        'iter_size': 1,
        }

    # parameters for generating detection output.
    det_out_param = {
        'num_classes': num_classes,
        'share_location': share_location,
        'background_label_id': background_label_id,
        'nms_param': {'nms_threshold': 0.45, 'top_k': 400},
        'keep_top_k': 200,
        'confidence_threshold': 0.01,
        'code_type': code_type,
        }


    ######################### TRAIN PROTOTXT #########################
    net = caffe.NetSpec()

    top_datalayer = ACT_DataLayer(dname, K, batch_size, resize_height=IMGSIZE, resize_width=IMGSIZE, restart_iter=0, flow=flow, ninput=5 if flow else 1)
    assert len(top_datalayer) == K + 1

    for i in range(K):
        net['data_stream' + str(i) + modality_str] = top_datalayer[i]
    net['label'] = top_datalayer[K]

    ACT_VGGNetBody(net, from_layer='data', K=K, fully_conv=True, reduced=True, dilated=True,
        dropout=False, freeze_layers=freeze_layers, m=modality_str, lr_mult=1.0/float(K))

    ACT_AddExtraLayers300(net, K, use_batchnorm, m=modality_str, lr_mult=lr_mult/float(K))
    mbox_layers = ACT_CreateCuboidHead(net, K, data_layer='data_stream0' + modality_str, from_layers=mbox_source_layers,
            use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
            aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
            num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
            prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=lr_mult, m=modality_str)
    name = "mbox_loss"
    mbox_layers.append(net.label)

    # CUBOID loss 
    net[name] = L.ACTCuboidLoss(*mbox_layers, multibox_loss_param=multibox_loss_param,
            act_cuboid_loss_param=act_cuboid_loss_param,
            loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
            propagate_down=[True, True, False, False])

    # Saving ..
    with open(train_net_file, 'w') as f:
        print('name: "{}_train"'.format(model_name), file=f)
        print(net.to_proto(), file=f)

    ######################### DEPLOY PROTOTXT #########################
    net = caffe.NetSpec()

    # Fake data layer that we delete later, just to have the output existing as top
    top_datalayer = ACT_DataLayer(dname, K, batch_size, resize_height=IMGSIZE, resize_width=IMGSIZE, restart_iter=0, flow=flow, ninput=5 if flow else 1)
    assert len(top_datalayer) == K + 1
    
    for i in range(K):
        net['data_stream' + str(i) + modality_str] = top_datalayer[i]
    
    ACT_VGGNetBody(net, from_layer='data', K=K, fully_conv=True, reduced=True, dilated=True,
        dropout=False, freeze_layers=freeze_layers, m=modality_str, lr_mult=1.0/float(K))

    ACT_AddExtraLayers300(net, K, use_batchnorm, m=modality_str, lr_mult=lr_mult/float(K))
    mbox_layers = ACT_CreateCuboidHead(net, K, data_layer='data_stream0'+modality_str, from_layers=mbox_source_layers,
        use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
        aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
        num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
        prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=lr_mult, m=modality_str)

    # net and mbox_layers
    conf_name = "mbox_conf"
    if multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.SOFTMAX:
        reshape_name = "{}_reshape".format(conf_name)
        net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, num_classes]))
        softmax_name = "{}_softmax".format(conf_name)
        net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
        flatten_name = "{}_flatten".format(conf_name)
        net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
        mbox_layers[1] = net[flatten_name]
    elif multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.LOGISTIC:
        sigmoid_name = "{}_sigmoid".format(conf_name)
        net[sigmoid_name] = L.Sigmoid(net[conf_name])
        mbox_layers[1] = net[sigmoid_name]

    # Detection output layer:
    # Saving detections for ACT-detector
    # -- The RGB stream saves boxes after per-class nms at 0.45 and thresholding scores
    # -- The flow stream saves all the regressed cuboids (with their scores
    if modality_str == "":
        net.detection_out = L.ACTDetectionOutput(*mbox_layers,
            detection_output_param=det_out_param,
            act_detection_output_param={'sequence_length': K},
            include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    else:        
        net.detection_out_full = L.ACTDetectionOutput(*mbox_layers,
            detection_output_param=det_out_param,
            act_detection_output_param={'sequence_length': K, 'save_full': True},
            include=dict(phase=caffe_pb2.Phase.Value('TEST')))
          
    net_param = net.to_proto()
    del net_param.layer[0]
    net_param.name = '{}_deploy'.format(model_name)
    for stream in xrange(K):
        net_param.input.extend(['data_stream' + str(stream) + modality_str])
        net_param.input_shape.extend([
            caffe_pb2.BlobShape(dim=[1, 3 * (5 if flow else 1), IMGSIZE, IMGSIZE])])
            
    # Saving .. 
    with open(deploy_net_file, 'w') as f:
        print(net_param, file=f)

    ######################### SOLVER PROTOTXT #########################
    solver = caffe_pb2.SolverParameter(
        train_net=train_net_file, snapshot_prefix=dirname, **solver_param)

    # Saving ..
    with open(solver_file, 'w') as f:
        print(solver, file=f)

        
if __name__=="__main__":
    dname = sys.argv[1]
    K = 6
    flow = bool(eval(sys.argv[2]))
    print(flow)
    ACT_generate_prototxt(dname, K=K, flow=flow)