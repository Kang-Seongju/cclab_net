import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Open Image Dataset Downloader  & train')

    ####################################
    # settings for training
    ####################################
    parser.add_argument("--custom", type=bool, default=True, help="custom train or not")
    parser.add_argument("--classes", type=int, default=80, help="Number of classes for training")
    parser.add_argument("--cfg", type=str, default="config/trial1.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
    parser.add_argument('--epochs', type=int, default=273)  # 500200 batches at bs 16, 117263 images = 273 epochs
    parser.add_argument('--accumulate', type=int, default=2, help='batches to accumulate before optimizing')
    parser.add_argument('--weights', type=str, default=None, help='initial weights')
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument('--batch-size', type=int, default=2)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument("--conf_thres", type=float, default=0.3, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="Evaluation: IOU threshold for non-maximum suppression")
    parser.add_argument("--logdir", type=str, default="logs", help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--transfer', action='store_true', help='transfer learning')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--img-weights', action='store_true', help='select training images by weight')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--arc', type=str, default='default', help='yolo architecture')  # defaultpw, uCE, uBCE
    parser.add_argument('--prebias', action='store_true', help='transfer-learn yolo biases prior to training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--var', type=float, help='debug variable')


    ####################################
    # OpenImage dataset download settings
    ####################################
    parser.add_argument('--limit', required=False, type=int, default=None,
                        metavar="integer number",
                        help='Optional limit on number of images to download')
    parser.add_argument('--dm_list', required=False, default='domains.txt',nargs='+',
                        metavar="list of classes",
                        help="Sequence of 'strings' of the wanted classes")
    parser.add_argument('--noLabels', required=False, action='store_true',
                        help='No labels creations')
    parser.add_argument('--n_threads', required=False, metavar="[default 20]", default=120,
                        help='Num of the threads for download dataset')


    # Not essential options below
    parser.add_argument('-y', '--yes', required=False, action='store_true',
                        # metavar="Yes to download missing files",
                        help='ans Yes to possible download of missing files')
    parser.add_argument('--sub', required=False, choices=['h', 'm'],
                        metavar="Subset of human verified images or machine generated (h or m)",
                        help='Download from the human verified dataset or from the machine generated one.')
    parser.add_argument('--image_IsOccluded', required=False, choices=['0', '1'],
                        metavar="1 or 0",
                        help='Optional characteristic of the images. Indicates that the object is occluded by another object in the image.')
    parser.add_argument('--image_IsTruncated', required=False, choices=['0', '1'],
                        metavar="1 or 0",
                        help='Optional characteristic of the images. Indicates that the object extends beyond the boundary of the image.')
    parser.add_argument('--image_IsGroupOf', required=False, choices=['0', '1'],
                        metavar="1 or 0",
                        help='Optional characteristic of the images. Indicates that the box spans a group of objects (min 5).')
    parser.add_argument('--image_IsDepiction', required=False, choices=['0', '1'],
                        metavar="1 or 0",
                        help='Optional characteristic of the images. Indicates that the object is a depiction.')
    parser.add_argument('--image_IsInside', required=False, choices=['0', '1'],
                        metavar="1 or 0",
                        help='Optional characteristic of the images. Indicates a picture taken from the inside of the object.')
    parser.add_argument('--OID_CSV', required=False,
                        metavar="/path/to/custom/csv/",
                        help='Directory of the OID dataset folder')
    return parser.parse_args()



