import logging
import tflite
import numpy as np
from tflite2onnx.layout import Layout
from tflite2onnx.op.activation import handleFusedActivation
from tflite2onnx.op.common import Operator
from tflite2onnx.op.padding import PaddingMapping
from tflite2onnx import mapping
from tflite2onnx.op.padding import computePaddingSize

logger = logging.getLogger('tflite2onnx')

"""
See about ConvTranspose.
https://github.com/onnx/onnx/blob/master/docs/Operators.md#ConvTranspose
"""
class ConvTranspose(Operator):
    TypeMapping = {
    
        tflite.BuiltinOperator.TRANSPOSE_CONV: 'ConvTranspose',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)

        self.attrs['dilations'] = []
        self.attrs['group'] = -1
        self.attrs['output_shape'] = []
        self.attrs['kernel_shape'] = []
        self.attrs['strides'] = []
        self.attrs['auto_pad'] = 'SAME_UPPER'  # See ComputePaddingHeightWidth() of TFLite
                       
        self.setInited()

    @property
    def type(self):
        return 'ConvTranspose'

    
    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        
        assert(opcode is tflite.BuiltinOperator.TRANSPOSE_CONV)
        assert(op.InputsLength() == 3)
        assert(op.OutputsLength() == 1)
           
        
        # X
        ilayout = Layout('NHWC', 'NCHW')
        self.parseInput(2, ilayout)
  
        # weight
        wlayout = Layout('CHWM', 'MCHW') 
        W = self.parseInput(1, wlayout)
    
        # output
        olayout = Layout('NHWC', 'NCHW')
        O = self.parseOutput(0, olayout)
        os = O.shape
        os_olayout = olayout.transform(os)
                
        # options
        op_opt = op.BuiltinOptions()
        option = tflite.TransposeConvOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)        
     
        
        # set attributes                
        self.attrs['output_shape'] = os_olayout
        self.attrs['dilations'] = [1, 1]
        self.attrs['group'] = 1
        self.attrs['auto_pad'] = PaddingMapping[option.Padding()]
        self.attrs['kernel_shape'] = W.shape[1:3]  
        self.attrs['strides'] = [option.StrideH(), option.StrideW()]
        
        self.setParsed()

    def propagatableTensors(self):
        return list()

    def transform(self):
        pass
