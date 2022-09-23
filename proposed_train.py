from random import shuffle
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.data import DataLoader
from utils.augmentations import AUGMENTATION_TRANSFORMS
from utils.transforms import DEFAULT_TRANSFORMS

from utils.parse_config import parse_data_cfg
from utils.utils import to_cpu, load_classes, print_environment_info, provide_determinism, worker_seed_set

import test  # import test.py to get mAP after each epoch
from proposed_model import *
from utils.datasets import *
from utils.utils import *


hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/416 if img_size != 416)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.225,  # iou training threshold
       'lr0': 0.00579,  # initial learning rate (SGD=1E-3, Adam=9E-5)
       'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 0.5,  # focal loss gamma
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98,  # image rotation (+/- deg)
       'translate': 0.05,  # image translation (+/- fraction)
       'scale': 0.05,  # image scale (+/- gain)
       'shear': 0.641}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt (optional)
f = glob.glob('hyp*.txt')
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v

prebias = 'store_true'
notest = 'store_true'
def _create_data_loader(anno, img_path, batch_size, img_size, multiscale_training, cls_path):
    dataset = ListDataset(
        anno,
        img_path,
        AUGMENTATION_TRANSFORMS,
        img_size,
        multiscale_training,
        cls_path
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set
    )

    return dataloader

def _create_validation_data_loader(anno, img_path, batch_size, img_size, cls_path):
    dataset = ListDataset(
        anno, 
        img_path, 
        DEFAULT_TRANSFORMS,
        img_size, 
        False, 
        cls_path
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    return dataloader

def train_p(cfg, wt, data_path, out_path, epochs = 100, tb_writer = None):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = cfg
    weight_path = wt
    last = out_path+'/last.pt'
    # Initialize
    init_seeds()
    results_file = './result_log/res_log.txt'
    # Configure run
    train_anno_path = data_path+'annotations/instances_train2017.json'
    valid_anno_path = data_path+'annotations/instances_val2017.json'
    train_path = data_path+'train2017/'
    valid_path = data_path+'val2017/'
    test_path = data_path+'test2017/'
    cls_path = data_path+'coco.names'


    print(train_anno_path, valid_anno_path)
    print(train_path, valid_path, cls_path)

    cls = read_class(cls_path)
    nc = len(cls)  # number of classes

    # Initialize model
    model = CCLAB(model_path, arc = 'default').to(device)

    img_size = 416
    batch_size = 8

    # Optimizer
    pg0, pg1 = [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if 'Conv2d.weight' in k:
            pg1 += [v]  # parameter group 1 (apply weight_decay)
        else:
            pg0 += [v]  # parameter group 0

    optimizer = optim.Adam(pg0, lr=hyp['lr0'])
    # optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    del pg0, pg1

    # optimizer = torch_utils.Lookahead(optimizer, k=5, alpha=0.5)

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_fitness = float('inf')

    if weight_path != None:  # pytorch format
        # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        chkpt = torch.load(weight_path, map_location=device)

        # load model
        try:
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(chkpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (args.weights, args.cfg, args.weights)
            raise KeyError(s) from e

        # load optimizer
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']

        # load results
        if chkpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt

        start_epoch = chkpt['epoch'] + 1
        del chkpt
    else :
        model.apply(weights_init_normal)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    # Initialize distributed training
    if device != 'cpu' and torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    #transform
    train_dataloader = _create_data_loader(
        train_anno_path,
        train_path,
        batch_size,
        img_size,
        True,
        cls_path
    )

    # Load validation dataloader
    validation_dataloader = _create_validation_data_loader(
        valid_anno_path,
        valid_path,
        batch_size,
        img_size,
        cls_path
    )

    img_sz_min = round(img_size / 32 / 1.5)
    img_sz_max = round(img_size / 32 * 1.5)

    # Start training
    nb = len(train_dataloader)
    model.nc = nc
    model.arc = 'default'
    model.hyp = hyp
    

    maps = np.zeros(nc)  # mAP per class
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    torch_utils.model_info(model, report='summary')  # 'full' or 'summary'

    print('Starting model training for %g epochs...', epochs)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))

        # Freeze backbone at epoch 0, unfreeze at epoch 1 (optional)
        freeze_backbone = False
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        mloss = torch.zeros(4).to(device)  # mean losses
        pbar = tqdm(enumerate(train_dataloader), total=nb)  # progress bar
        for i, (paths, imgs, targets) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0

            targets = targets.to(device)

            # Multi-Scale training
            
            if ni / 4 % 10 == 0:  #  adjust (67% - 150%) every 10 batches
                img_size = random.randrange(img_sz_min, img_sz_max + 1) * 32
            sf = img_size / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Run model
            pred = model(imgs) # imgs shape -> [8,3,416,416]

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Scale loss by nominal batch_size of 64
            loss *= batch_size / 64

            # Compute gradient
            # if mixed_precision:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            #     loss.backward()
            loss.backward()

            # Accumulate gradient for x batches before optimizing
            if ni % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, len(targets), img_size)
            pbar.set_description(s)

            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        scheduler.step()

        # Process epoch results
        final_epoch = epoch + 1 == epochs
        
        results, maps = test.test(
            cfg,
            valid_path,
            batch_size=batch_size,
            img_size=img_size,
            model=model,
            conf_thres=0.001 if final_epoch else 0.1,  # 0.1 for speed
            save_json=final_epoch,
            dataloader=validation_dataloader,
            class_list = cls
        )

        # Write epoch results
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        # Write Tensorboard results
        if tb_writer:
            x = list(mloss) + list(results)
            titles = ['GIoU', 'Objectness', 'Classification', 'Train loss',
                      'Precision', 'Recall', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification']
            for xi, title in zip(x, titles):
                tb_writer.add_scalar(title, xi, epoch)

        # Update best mAP
        fitness = sum(results[4:])  # total loss
        if fitness < best_fitness:
            best_fitness = fitness

        # Save training results

        with open(results_file, 'r') as f:
            # Create checkpoint
            chkpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': f.read(),
                        'model': model.module.state_dict() if type(
                            model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                        'optimizer': None if final_epoch else optimizer.state_dict()}

        # Save last checkpoint
        ckp_path = './weights/proposed_'+str(epoch) + '.pth'
        torch.save(model.state_dict(), ckp_path)
            
        # Delete checkpoint
        del chkpt

        # end epoch ----------------------------------------------------------------------------------------------------

    plot_results()  # save as results.png
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()

    return results

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)