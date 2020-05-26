import os
import subprocess


def cmd_convert(model_name):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    tflm_dir = os.path.abspath(cur_dir + '/../assets/tests')
    tflm_name = model_name + '.tflite'
    onnx_name = model_name + '.onnx'
    tflm_path = os.path.join(tflm_dir, tflm_name)

    cmd = "tflite2onnx %s %s" % (tflm_path, onnx_name)
    cmd = cmd.split(' ')

    process = subprocess.run(cmd)
    assert(process.returncode == 0)


def test_cmd_convert():
    MODEL_LIST = (
        'abs.float32',
    )

    for m in MODEL_LIST:
        cmd_convert(m)


if __name__ == '__main__':
    test_cmd_convert()
