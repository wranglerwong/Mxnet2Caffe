'''
Transforming network definition from mxnet.json to caffe.prototxt
'''
import sys
import json
import sys
sys.path.insert(0, '/home/huangxuankun/caffe/caffe/python')
import caffe
from caffe import layers as L
from caffe import params as P

# model_def: Inception-7-symbol.json
model_def = sys.argv[1]

N = caffe.NetSpec()
N.data, N.label = L.Data(batch_size=64,
                         backend=P.Data.LMDB, source='/home/tmp_data_dir/huangxuankun/batch/imagenet2015/CLS/val_lmdb',
                         transform_param=dict(scale=1./255, crop_size=299), include=dict(phase=True), ntop=2)

def gen_conv(name, bottom, kernel_size, num_output, stride, pad, group=1):
	if kernel_size[0] == kernel_size[1]:
		exec(name+'=L.Convolution('+bottom+', kernel_size='+str(kernel_size[0])+', num_output='+str(num_output)+', stride='+str(stride)+', pad='+str(pad[0])+', group='+str(group)+', bias_term=False, weight_filler=dict(type="gaussian", std=0.01))')
	else:
		exec(name+'=L.Convolution('+bottom+', kernel_h='+str(kernel_size[0])+', kernel_w='+str(kernel_size[1])+', num_output='+str(num_output)+', stride='+str(stride)+', pad_h='+str(pad[0])+', pad_w='+str(pad[1])+', group='+str(group)+', bias_term=False, weight_filler=dict(type="gaussian", std=0.01))')

def gen_pooling(name, bottom, kernel_size, stride, pad, pool_type):
	if pool_type == 'max':
		exec(name+'=L.Pooling('+bottom+', kernel_size='+str(kernel_size[0])+', stride='+str(stride)+', pad='+str(pad[0])+', pool=P.Pooling.MAX)')
	elif pool_type == 'avg':
		exec(name+'=L.Pooling('+bottom+', kernel_size='+str(kernel_size[0])+', stride='+str(stride)+', pad='+str(pad[0])+', pool=P.Pooling.AVE)')

def gen_fc(name, bottom, num_output):
	exec(name+'=L.InnerProduct('+bottom+', num_output='+str(num_output)+', weight_filler=dict(type="xavier"))')

def gen_relu(name, bottom):
	exec(name+'=L.ReLU('+bottom+', in_place=True)')

def gen_bn(name, bottom):
	exec(name+'=L.BatchNorm('+bottom+', use_global_stats=True, in_place=True, eps=0.001, moving_average_fraction=0.9997)')

def gen_scale(name, bottom):
	exec(name+'=L.Scale('+bottom+', bias_term=True, in_place=True)')

def gen_concat(name, bottom):
	exec(name+'=L.Concat('+bottom+', axis=1)')

def gen_flatten(name, bottom):
	exec(name+'=L.Flatten('+bottom+')')

def gen_softmax(name, bottom):
	exec(name+'=L.Softmax('+bottom+')')

def parse_mx_json(model_def):
	f = open(model_def, 'r')
	model_json = json.load(f)
	nodes = model_json['nodes']
	arg_nodes = model_json['arg_nodes']
	heads = model_json['heads']
	for i in xrange(len(nodes)):
		if i in arg_nodes:
			continue
		layer_type = nodes[i]['op']
		layer_inputs = nodes[i]['inputs']
		layer_name = nodes[i]['name']
		inputs_arr = []
		for j in xrange(len(layer_inputs)):
			tmp_input = layer_inputs[j][0]
			if tmp_input in arg_nodes:
				continue
			inputs_arr.append(nodes[tmp_input]['name'])

		if layer_type == 'Convolution':
			name = 'N.{}'.format(layer_name)
			inputs_str = ''
			if len(inputs_arr) == 0: # first convolution
				inputs_str = 'N.data'
			else:
				inputs_str = ', '.join('N.{}'.format(x) for x in inputs_arr)

			kernel_size = map(lambda x: int(x.strip()), nodes[i]['param']['kernel'].strip('(,)').split(','))
			num_output = nodes[i]['param']['num_filter']
			stride = map(lambda x: int(x.strip()), nodes[i]['param']['stride'].strip('(,)').split(','))[0]
			pad = map(lambda x: int(x.strip()), nodes[i]['param']['pad'].strip('(,)').split(','))
			group = nodes[i]['param']['num_group']
			if inputs_str.endswith('_relu'):
				inputs_str = inputs_str[:-5]+'_conv2d'
			gen_conv(name, inputs_str, kernel_size, num_output, stride, pad, group)
			print inputs_str
		elif layer_type == 'Pooling':
			name = 'N.{}'.format(layer_name)
			inputs_str = ', '.join('N.{}'.format(x) for x in inputs_arr)
			kernel_size = map(lambda x: int(x.strip()), nodes[i]['param']['kernel'].strip('(,)').split(','))
			stride = map(lambda x: int(x.strip()), nodes[i]['param']['stride'].strip('(,)').split(','))[0]
			pad = map(lambda x: int(x.strip()), nodes[i]['param']['pad'].strip('(,)').split(','))
			pool_type = nodes[i]['param']['pool_type']
			gen_pooling(name, inputs_str, kernel_size, stride, pad, pool_type)
		elif layer_type == 'FullyConnected':
			name = 'N.{}'.format(layer_name)
			inputs_str = ', '.join('N.{}'.format(x) for x in inputs_arr)
			num_output = nodes[i]['param']['num_hidden']
			gen_fc(name, inputs_str, num_output)
		elif layer_type == 'Activation':
			name = 'N.{}'.format(layer_name)
			inputs_str = ', '.join('N.{}'.format(x) for x in inputs_arr)
			gen_relu(name, inputs_str[:-10]+'_conv2d')
		elif layer_type == 'BatchNorm':
			name = 'N.{}'.format(layer_name)
			inputs_str = ', '.join('N.{}'.format(x) for x in inputs_arr)
			# gen_bn(name, inputs_str)
			gen_scale(name[:-10]+'_scale', inputs_str)
		elif layer_type == 'Concat':
			name = 'N.{}'.format(layer_name)
			inputs_str = ', '.join('N.{}'.format(x) for x in inputs_arr)
			gen_concat(name, inputs_str)
		elif layer_type == 'Flatten':
			name = 'N.{}'.format(layer_name)
			inputs_str = ', '.join('N.{}'.format(x) for x in inputs_arr)
			gen_flatten(name, inputs_str)
		elif layer_type == 'SoftmaxOutput':
			name = 'N.{}'.format(layer_name)
			inputs_str = ', '.join('N.{}'.format(x) for x in inputs_arr)
			gen_softmax(name, inputs_str)
		else:
			print '[ERROR] Unknown layer type: {}\n'.format(layer_type)

def main():
	parse_mx_json(model_def)
	proto = N.to_proto()
	with open('converted.prototxt', 'w') as fp:
		fp.write(str(proto))

if __name__ == '__main__':
	main()

