import logging
import tflite
import numpy as np

from tflite2onnx.layout import Layout
from tflite2onnx import mapping
from tflite2onnx.op.common import Operator

logger = logging.getLogger('tflite2onnx')


class Resize(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR: 'Resize',
        tflite.BuiltinOperator.RESIZE_BILINEAR: 'Resize',
        # No RESIZE_BICUBIC in BuiltinOperator
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)
        # Four choices:
        # half_pixel, pytorch_half_pixel, align_corners, asymmetric, tf_crop_and_resize
        self.attrs['coordinate_transformation_mode'] = 'half_pixel'
        # This attribute is valid only if "mode" is "cubic".
        # The coefficient 'a' used in cubic interpolation.
        # Two common choice are -0.5 (in some cases of TensorFlow) and -0.75 (in PyTorch).
        self.attrs['cubic_coeff_a'] = -0.75
        self.attrs['exclude_outside'] = 0
        self.attrs['extrapolation_value'] = 0.0
        # Three interpolation modes: nearest (default), linear and cubic.
        # The "linear" mode includes linear interpolation for 1D tensor
        # and N-linear interpolation for N-D tensor
        # (for example, bilinear interpolation for 2D tensor).
        # The "cubic" mode includes cubic interpolation for 1D tensor
        # and N-cubic interpolation for N-D tensor
        # (for example, bicubic interpolation for 2D tensor).
        self.attrs['mode'] = 'nearest'
        # Four modes: round_prefer_floor (default, as known as round half down),
        # round_prefer_ceil (as known as round half up), floor, ceil.
        # Only used by nearest interpolation.
        # It indicates how to get "nearest" pixel in input tensor from x_original,
        # so this attribute is valid only if "mode" is "nearest".
        self.attrs['nearest_mode'] = 'round_prefer_floor'

        self.setInited()

    @property
    def type(self):
        return 'Resize'

    @property
    def isRESIZE_BILINEAR(self):
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        return opcode is tflite.BuiltinOperator.RESIZE_BILINEAR

    @property
    def isRESIZE_NEAREST_NEIGHBOR(self):
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        return opcode is tflite.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert (opcode in self.TypeMapping)
        
        assert(op.InputsLength() == 2), "Only first two arguments: image and size, are compulsory"
        assert(op.OutputsLength() == 1)

        # image
        # TFLite requires its `Resize` to be NHWC
        ilayout = Layout('NHWC', 'NCHW')
        im = self.parseInput(0, ilayout)

        # Fill 'ROI' empty temporarily
        # because 'tf_crop_and_resize' was not found in BuiltinOptions
        # of ResizeBilinear and ResizeNearestNeighbor
        roi = self.TFactory.createVector(im, np.array([]))
        roi.addConsumer(self)
        self.inputs.append(roi)

        # Fill empty to scales temporarily
        sc = self.TFactory.createVector(im, np.array([]))
        self.inputs.append(sc)

        # output size expected
        sz = self.parseInput(1)

        # In case Resize, the number of elements of both the arguments
        # 'sizes' and 'scales' are required to be the same as the rank of input 'X'.
        # The 'X' usually has a (N,C,H,W) layout after transforming
        # while the 'sizes' and 'scales' only have (H_new,W_new).
        # Thus we copy the N and C to achieve (N, C, H_new,W_new) for these two arguments.
        assert len(sz.data) == 2
        assert len(im.shape) == 4
        sz.data = np.concatenate((np.array([im.shape[0], im.shape[-1]]), sz.data))
        sz.shape = [len(im.shape)]
        sz.dtype = mapping.DTYPE_NAME2ONNX['int64']

        # output
        olayout = Layout('NHWC', 'NCHW')
        self.parseOutput(0, olayout)

        # options
        op_opt = op.BuiltinOptions()
        option = tflite.ResizeBilinearOptions() if self.isRESIZE_BILINEAR \
            else tflite.ResizeNearestNeighborOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)

        if self.isRESIZE_BILINEAR:
            self.attrs['mode'] = 'linear'
        elif self.isRESIZE_NEAREST_NEIGHBOR:
            self.attrs['mode'] = 'nearest'

        if option.AlignCorners():
            self.attrs['coordinate_transformation_mode'] = 'align_corners'
        elif option.HalfPixelCenters():
            self.attrs['coordinate_transformation_mode'] = 'half_pixel'

        self.setParsed()

    def propagatableTensors(self):
        return list()

    def transform(self):
        pass
