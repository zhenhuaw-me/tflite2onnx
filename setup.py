# coding: utf-8
import setuptools, os

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='tflite2onnx',
    version='0.0.1',
    description="Converting TensorFlow Lite models to ONNX models",
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 1 - Planning',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=['tflite', 'tensorflow', 'onnx'],
    python_requires='>=3.5.*, <4',
    install_requires=['numpy'],

    author='王振华(Zhenhua WANG)',
    author_email='i@jackwish.net',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://jackwish.net/tflite2onnx",

    project_urls={
        'Bug Reports': 'https://github.com/jackwish/tflite2onnx/issues',
        'Source': 'https://github.com/jackwish/tflite2onnx',
    },
)
