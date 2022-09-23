import sys
import os
import argparse
import numpy as np
import cv2
import onnxruntime
import torch
from utils.datasets import *
from utils.utils import *
from utils.torch_utils import select_device
from models import *  # set ONNX_EXPORT in models.py
from utils.general import check_requirements

def export_onnx(opt, file):
    #  ONNX export
    # opt.opset, opt.train, opt.dynamic, opt.simplify, opt.workspace
    print(file)
    opset =opt.opset
    train =opt.train
    dynamic = opt.dynamic
    simplify = opt.simplify
    workspace = opt.workspace
    trt_save_name = file.split('.')[0] + '.engine'
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    img_size = (256, 416)


    # Initialize model
    model = Darknet(opt.model, img_size)  # img_size -> 416, 416
    weights = opt.weights

    # attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Eval mode
    model.to(device).eval()
    im = torch.zeros((1,3)+img_size)

    if opt.half:
        im, model = im.half(), model.half()
    try:
        import onnx
        print('start onnx exporting..')
        torch.onnx.export(
            model,
            im,
            file,
            verbose=False,
            opset_version=opset,
            do_constant_folding=not train,
            input_names = ['input'],
            output_names = ['outputs']
        )


        # Checks
        model = onnx.load(file)  # load onnx model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph

        # Simplify
        if simplify:
            try:
                check_requirements(('onnx-simplifier',))
                import onnxsim

                print("simplyfying with onnx simplify...")
                model_onnx, check = onnxsim.simplify(
                    file,
                    dynamic_input_shape=dynamic,
                    input_shapes={'images': list(im.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, file)
            except Exception as e:
                print('simplify failure')
                print(e)

    except Exception as e:
        print('onnx_export_fail')
        print(e)
        return

    ONNX = 'trtexec --onnx=%s' % file
    BATCH = '--explicitBatch'
    ENGINE = '--saveEngine=%s' % trt_save_name
    WORKSPACE = '--workspace=%d' % workspace
    FP = '--fp16 '
    command = ' '.join([ONNX, BATCH, ENGINE, WORKSPACE, FP])
    os.system(command)
    return



@torch.no_grad()
def run(opt):
    # Load PyTorch model
    weights = opt.weights
    device = select_device(opt.device)
    onnx_file_name = opt.model.split('/')[-1].split('.')[0] +'.onnx'

    assert not (device.type == 'cpu' and opt.half), '--half only compatible with GPU export, i.e. use --device 0'


    # Input
    im =  torch.zeros(opt.batch_size, 3, *opt.imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection


    # Exports
    export_onnx(opt, onnx_file_name)


    # Finish
    print('Exprot completed !')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='set the cfg file path ')
    parser.add_argument('--workspace', type=int, default=4096, help='set the workspace size for TRT trasnformation')
    parser.add_argument('--weights', type=str, required=True, help='weights path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[256, 416], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv3 Detect() inplace=True')
    parser.add_argument('--train', action='store_true', help='model.train() mode')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TF: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=11, help='ONNX: opset version')

    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    print("Converting to onnx and running demo ...")
    opt = parse_opt()
    run(opt)
