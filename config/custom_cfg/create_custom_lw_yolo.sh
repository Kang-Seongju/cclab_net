#!/bin/bash

NUM_CLASSES=$1
MODEL_NAME=$2
DOMAIN_NAME=$3
echo "
[net]
# Testing
# batch=1
# subdivisions=1
# Training
batch=8
subdivisions=2
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=400
max_batches = $(expr $NUM_CLASSES \* 4000)
policy=steps
steps= $(expr $NUM_CLASSES \* 1600) ,$(expr $NUM_CLASSES \* 1800)
scales=.1,.1

#0
[convolutional]
batch_normalize=1
filters=16
size=3
stride=1
pad=1
activation=leaky
# 416

#1
[maxpool]
size=2
stride=2
#208

#2
[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky
# 208
### Dual-residual

#3
[maxpool]
size = 2
stride = 2

#4
[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

#5
[upsample]
stride = 2

#6
[route]
layers = -4

#7
[upsample]
stride = 2

#8
[convolutional]
batch_normalize=1
filters=16
size=1
stride=1
pad=1
activation=leaky

#9
[maxpool]
size = 2
stride = 2

#10
[route]
layers = -1, 6

#11
[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

#12
[shortcut]
from = -10

## Dual residual end

#13
[maxpool]
size=2
stride=2
#104

#14
[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky
# 104
### Dual-residual

#15
[maxpool]
size = 2
stride = 2

#16
[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

#17
[upsample]
stride = 2

#18
[route]
layers = -4

#19
[upsample]
stride = 2

#20
[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

#21
[maxpool]
size = 2
stride = 2

#22
[route]
layers = -1, 14

#23
[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

#24
[shortcut]
from = -10

### end

#25
[maxpool]
size = 2
stride = 2
#52

#26
[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky
# 52
### Dual-residual

#27
[maxpool]
size = 2
stride = 2
#26

#28
[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

#29
[upsample]
stride = 2

#30
[route]
layers = -4

#31
[upsample]
stride = 2

#32
[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

#33
[maxpool]
size = 2
stride = 2

#34
[route]
layers = -1, 26

#35
[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

#36
[shortcut]
from = -10

[maxpool]
size = 2
stride = 2
#26

#26
[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky
# 52
### Dual-residual

#27
[maxpool]
size = 2
stride = 2
#26

#28
[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

#29
[upsample]
stride = 2

#30
[route]
layers = -4

#31
[upsample]
stride = 2

#32
[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

#33
[maxpool]
size = 2
stride = 2

#34
[route]
layers = -1, 38

#35
[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

#36
[shortcut]
from = -10


### Branch A ####

#17
[convolutional]
batch_normalize=1
filters=512
size=3
stride=2
pad=1
activation=leaky
# 13

#18
[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky
# 13

#19
[convolutional]
size=1
stride=1
pad=1
filters=$(expr 3 \* $(expr $NUM_CLASSES \+ 5))
activation=linear


#20
[yolo]
mask = 6,7,8
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=$NUM_CLASSES
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1

#12
[route]
layers = -3

#13
[upsample]
stride = 2

### Branch B ###

#14
[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

#15
[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

#16
[convolutional]
size=1
stride=1
pad=1
filters=$(expr 3 \* $(expr $NUM_CLASSES \+ 5))
activation=linear


#17
[yolo]
mask = 3,4,5
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=$NUM_CLASSES
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1

#18
[route]
layers = -3

#19
[upsample]
stride = 2

#20
[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

#21
[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

#22
[convolutional]
size=1
stride=1
pad=1
filters=$(expr 3 \* $(expr $NUM_CLASSES \+ 5))
activation=linear

#23
[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=$NUM_CLASSES
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
" >> "${DOMAIN_NAME}_${MODEL_NAME}_${NUM_CLASSES}.cfg"
