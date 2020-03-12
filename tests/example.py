import tflite2onnx as t2o

basedir = '/home/scratch.zhenhuaw_sw/onnx/shrub//misc/models/tfbuilder/'
f = 'elementwise/abs.int32.tflite'

t2o.convert(basedir + f, 'm.onnx')
