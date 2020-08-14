import copy
import os
import logging

import shrub
import tflite2onnx as t2o

shrub.util.formatLogging(logging.DEBUG)


def end2end_test(model_name, use_layout):
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

    # ONNX model is supposed to be only several operators quantized
    iquant = shrub.tflite.parseQuantParam(tflm_path, True)[0]
    oquant = shrub.tflite.parseQuantParam(tflm_path, False)[0]
    finputs = list()
    for q in m.inputs:
        finput = copy.deepcopy(q)
        finput.quant = iquant
        finput.dequantize()
        finputs.append(finput)
    onnx_ret = shrub.onnx.run(onnx_name, finputs, use_layout)
    qoutputs = list()
    for f in onnx_ret:
        qoutput = copy.deepcopy(f)
        qoutput.quant = oquant
        qoutput.quant.quantized = False
        qoutput.quantize()
        qoutputs.append(qoutput)

    assert(shrub.network.cmpTensors(qoutputs, tflite_ret, atol=1.0))


def test_quantized_ops():
    OP_LIST = (
        'conv.uint8',
        # 'conv-relu.uint8',
        # 'depthwise-conv.uint8',
    )

    for op in OP_LIST:
        end2end_test(op, 'NCHW')


# def test_quantized_networks():
#     NETWORK_LIST = (
#         'mobilenet_v1_0.25_128_quant',
#     )

#     for net in NETWORK_LIST:
#         end2end_test(net, 'NCHW')


if __name__ == '__main__':
    test_quantized_ops()
    # test_quantized_networks()
