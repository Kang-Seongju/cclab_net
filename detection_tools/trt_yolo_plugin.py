import ctypes
import sys
import os
import time
import argparse
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
from .utils import load_class_names, rescale_boxes
from utils.utils import non_max_suppression
import torch
from detection_tools.utils import post_processing
import pycuda.autoinit

# Simple helper data class that's a little nicer to use than a 2-tuple.
def GiB(val):
    return val * 1 << 30

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        del self.device
        del self.host

def allocate_buffers(engine, batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:

        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dims = engine.get_binding_shape(binding)

        # in case batch dimension is -1 (dynamic)
        if dims[0] < 0:
            size *= -1
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


class Trt_yolo(object):
    def __init__(self, engine_path, num_classes, img_size, half):
        self.img_size = img_size
        self.half = half
        self.engine = engine_path
        self.num_classes = num_classes
        #self.letter_box = letter_box
        self.IN_IMAGE_H,self.IN_IMAGE_W = img_size
        # 추후에 multi-batch를 지원하려면 나중에 확장하기
        self.inference_fn = do_inference
        self.trt_logger = trt.Logger()
        trt.init_libnvinfer_plugins(self.trt_logger, '')
        self.engine = self.get_engine()

        # self.batch = 1 # explicit batch ->1 , if use multil batch , fix it
        # self.no = num_classes + 5
        # self.oi =[0, 1, 2, 3] + list(range(5, self.no)) # objectness score + class score


        #self.input_shape = get_input_shape(self.engine)
        try:
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine, 1)
            self.context.set_binding_shape(0, (1, 3, 256, self.IN_IMAGE_W))

        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e

    def __del__(self):
        # Cuda memory free
        del self.outputs
        del self.inputs
        del self.stream

    '''input 점검하기 '''

    def detect(self, image_src, conf_thresh=0.4, nms_thresh=0.6):

        # im0 = image_src.unsqueeze(0)
        im0 = image_src.shape #(480, 854, 3)
        img = letterbox(image_src, new_shape= self.img_size[0])[0] #
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0  shape: 3 x 256 x 416
        self.inputs[0].host =img # 네트워크 입력에 맞게 resize된 입력
        pred = do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        onnx_score = pred[0].reshape(-1,4)
        box_score = pred[1].reshape(-1,4)
        pred = np.concatenate((box_score,onnx_score),1)
        pred = pred.reshape(1,-1,8)

        return pred, img, im0

    def get_engine(self):
        print("Reading engine from file {}".format(self.engine))
        with open(self.engine, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())


def letterbox(img, new_shape=(416, 416), color=(128, 128, 128),
              auto=True, scaleFill=False, scaleup=True, interp=cv2.INTER_AREA):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = max(new_shape) / max(shape)
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=interp)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



