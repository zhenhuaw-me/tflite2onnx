import copy
import os
import logging

import shrub
import tflite2onnx as t2o

shrub.util.formatLogging(logging.DEBUG)


def end2end_test(model_name, use_layout, atol, rtol):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    tflm_dir = os.path.abspath(cur_dir + '/../assets/tests')
    tflm_name = model_name + '.tflite'
    onnx_name = model_name + '.onnx'
    tflm_path = os.path.join(tflm_dir, tflm_name)
    t2o.convert(tflm_path, onnx_name)

    m = shrub.tflite.parse(tflm_path)
    m.genInput()

    # TFLite model is supposed to be end to end quantized
    tflite_ret = shrub.tflite.run(tflm_path, m.inputs)
    oquant = shrub.tflite.parseQuantParam(tflm_path, False)[0]
    foutputs = list()
    for f in tflite_ret:
        foutput = copy.deepcopy(f)
        foutput.quant = oquant
        foutput.dequantize()
        foutputs.append(foutput)

    # ONNX model is supposed to be only several operators quantized
    iquant = shrub.tflite.parseQuantParam(tflm_path, True)[0]
    finputs = list()
    for q in m.inputs:
        finput = copy.deepcopy(q)
        finput.quant = iquant
        finput.dequantize()
        finputs.append(finput)
    onnx_ret = shrub.onnx.run(onnx_name, finputs, use_layout)

    assert(shrub.network.cmpTensors(foutputs, onnx_ret, atol=atol, rtol=rtol, useLayout=use_layout))


def test_quantized_ops():
    OP_LIST = (
        'conv.uint8',
        'conv-relu.uint8',
        'depthwise-conv.uint8',
    )

    for op in OP_LIST:
        end2end_test(op, 'NCHW', 1e-7, 1e-5)


def test_quantized_networks():
    NETWORK_LIST = (
        'mobilenet_v1_0.25_128_quant',
    )

    for net in NETWORK_LIST:
        # relax precision for end to end network
        end2end_test(net, 'NCHW', 1e-1, 1e-5)


if __name__ == '__main__':
    test_quantized_ops()
    test_quantized_networks()
