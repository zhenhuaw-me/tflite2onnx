#!/bin/bash

if [ "$(uname -s)" == "Darwin" ]; then
  root_dir=$(dirname $(dirname $(greadlink -f $0})))
else
  root_dir=$(dirname $(dirname $(readlink -f $0})))
fi

${root_dir}/tools/build-wheel.sh

read -p "Will upload to test.pypi.org, for real publishment type \"Release\": " input_str
if [ -z "${input_str}" -o ${input_str} != "Release" ]; then
  python3 -m twine upload \
    --repository-url https://test.pypi.org/legacy/ \
    ${root_dir}/assets/dist/tflite2onnx-*
else
  read -p "Will publish the package, are you sure to continue [Y|N] ? " input_str
  if [ -n "${input_str}" -a ${input_str} = "Y" ]; then
    echo "Uploading..."
    python3 -m twine upload ${root_dir}/assets/dist/tflite2onnx-*
  fi
fi
