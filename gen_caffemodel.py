'''
Transforming network definition from mxnet.params to caffe.caffemodel
'''
import mxnet
import sys
sys.path.insert(0, '/home/huangxuankun/caffe/caffe/python')
import caffe
import numpy as np

caffe_model_def = sys.argv[1]
caffe_model_params = sys.argv[2]

mxnet_model_prefix = sys.argv[3]
mxnet_model_epoch = sys.argv[4]

caffemodel = caffe.Net(caffe_model_def, caffe_model_params, caffe.TEST)
mxnetmodel = mxnet.model.FeedForward.load(mxnet_model_prefix, int(mxnet_model_epoch))
arg = mxnetmodel.arg_params
aux = mxnetmodel.aux_params

for key in caffemodel.params.keys():
	if key.endswith('conv2d'): 
		assert caffemodel.params[key][0].data.shape == arg[key+'_weight'].asnumpy().shape
		caffemodel.params[key][0].data[...] = arg[key+'_weight'].asnumpy()
		# no_bias: True
	elif key.startswith('fc'):
		assert caffemodel.params[key][0].data.shape == arg[key+'_weight'].asnumpy().shape
		caffemodel.params[key][0].data[...] = arg[key+'_weight'].asnumpy()
		# no_bias: False
		assert caffemodel.params[key][1].data.shape == arg[key+'_bias'].asnumpy().shape
		caffemodel.params[key][1].data[...] = arg[key+'_bias'].asnumpy()
	elif key.endswith('batchnorm'):
		print 'ops?...'
		assert caffemodel.params[key][0].data.shape == aux[key+'_moving_mean'].asnumpy().shape
		caffemodel.params[key][0].data[...] = aux[key+'_moving_mean'].asnumpy()
		assert caffemodel.params[key][1].data.shape == aux[key+'_moving_var'].asnumpy().shape
		caffemodel.params[key][1].data[...] = aux[key+'_moving_var'].asnumpy()
	elif key.endswith('scale'):
		assert caffemodel.params[key][0].data.shape == arg[key[:-6]+'_batchnorm_gamma'].asnumpy().shape
		mean = aux[key[:-6]+'_batchnorm_moving_mean'].asnumpy()
		var = aux[key[:-6]+'_batchnorm_moving_var'].asnumpy()
		gamma = arg[key[:-6]+'_batchnorm_gamma'].asnumpy()
		beta = arg[key[:-6]+'_batchnorm_beta'].asnumpy()
		eps = 1e-10
		caffemodel.params[key][0].data[...] = gamma / np.power(var + eps, 0.5)
		caffemodel.params[key][1].data[...] = beta - gamma * mean / np.power(var + eps, 0.5)
		# caffemodel.params[key][0].data[...] == arg[key[:-6]+'_batchnorm_gamma'].asnumpy()
		# assert caffemodel.params[key][1].data.shape == arg[key[:-6]+'_batchnorm_beta'].asnumpy().shape
		# caffemodel.params[key][1].data[...] == arg[key[:-6]+'_batchnorm_beta'].asnumpy()
	else:
		print '[ERROR] Unknown layer:', key

caffemodel.save('converted.caffemodel')


