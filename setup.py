import setuptools
import tflite2onnx

setuptools.setup(
    name=tflite2onnx.NAME,
    version=tflite2onnx.VERSION,
    description=tflite2onnx.DESCRIPTION,
    license=tflite2onnx.LICENSE,
    packages=setuptools.find_packages(),
    python_requires='>=3.5.*, <4',
    install_requires=['numpy', 'onnx', 'tflite'],

    entry_points={
        'console_scripts': [
            "tflite2onnx = tflite2onnx.convert:cmd_convert"
        ],
    },

    author='王振华(Zhenhua WANG)',
    author_email='i@jackwish.net',
    url="https://jackwish.net/tflite2onnx",

    project_urls={
        'Bug Reports': 'https://github.com/jackwish/tflite2onnx/issues',
        'Source': 'https://github.com/jackwish/tflite2onnx',
    },
)
