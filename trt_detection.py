import os
import time
import argparse
import cv2
import pycuda.autoinit

from detection_tools.display_utils import open_window, set_display, show_fps
from detection_tools.detect_uilts import add_camera_args, Detect
from detection_tools.utils import *
from detection_tools.trt_yolo_plugin import Trt_yolo
from utils.utils import *
# real time object detection with TensorRT optimized YOLO engine

WINDOW_NAME = 'Trt_object_detection'


def parse_args():
    desc = ('Display real-time object detection in video file with TensorRT optimized YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)

    parser.add_argument(
        '-c', '--name_file', type=str, default='data/coco.names',
        help='path of class name file')

    parser.add_argument(
        '-t', '--conf_thresh', type=float, default=0.1,
        help='set the detection confidence threshold')
    parser.add_argument(
        '-n', '--nms_thresh', type=float, default=0.4,
        help='set the detection confidence threshold')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='set the path of engine file ex: ./yolo.engine')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')

    args = parser.parse_args()
    return args

def loop_and_detect(cam, trt_yolo, conf_th, nms_th, class_list):
    # cam: the camera instance (video source)
    # trt_yolo : trt_yolo object detetor instance
    # vis : for visualization

    full_screen = False
    fps = 0.0
    tic = time.time()
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(class_list))]

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        # img shape -> (480, 854, 3)
        pred, rsz_img, orisz_img = trt_yolo.detect(img,conf_th, nms_th) # trt engine 기반 추론
        pred = non_max_suppression(torch.from_numpy(pred), conf_th, nms_th)
        # rsz_img.shape
        for i, det in enumerate(pred):
            s, im0 = '', img
            if det is not None and len(det):
                det[:,:4] = scale_coords(rsz_img.shape[1:], det[:, :4], orisz_img).round()

                # print result
                for c in det[:,-1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, class_list[int(c)])  # add to string

                # write results
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (class_list[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

        im0 = show_fps(im0, fps) # fps 표시
        cv2.imshow(WINDOW_NAME, im0)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc


        key = cv2.waitKey(1)
        if key == 27: # Esc키로 quit
            break
        elif key == ord('F') or key == ord('f'):
            full_screen = not full_screen
            set_display(WINDOW_NAME, full_screen)



def main():
    args = parse_args()
    if not os.path.isfile('%s' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.engine) not found!' % args.model)

    cam = Detect(args) # video and camera instance module
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    # input size of network
    img_size = (416, 416)
    cls_list =load_class_names(args.name_file)

    open_window(
        WINDOW_NAME, 'TensorRT object detecion',
        cam.img_width, cam.img_height)

    trt_yolo = Trt_yolo(args.model, len(cls_list), img_size, args.half)
    loop_and_detect(cam, trt_yolo, args.conf_thresh, args.nms_thresh, cls_list)
    cam.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()