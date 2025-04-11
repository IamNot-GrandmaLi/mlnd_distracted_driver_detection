# 生成对混合模型的输入
import os
import shutil

from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *


from keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_v3_preprocess_input
from keras.applications.xception import Xception, preprocess_input as xception_preprocess_input
# from keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocess_input


import h5py
import math

dir = r"F:\JupyterWorkSpace\BBBBBBS"

resnet50_weight_file = "resnet50-imagenet-finetune152.h5"
xception_weight_file = "xception-imagenet-finetune116.h5"
inceptionV3_weight_file = "inceptionV3-imagenet-finetune172.h5"

def write_gap(tag, MODEL, weight_file, image_size, lambda_func=None, featurewise_std_normalization=True):
    input_tensor = Input((*image_size, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
    base_model = MODEL(input_tensor=x, weights=None, include_top=False)

    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    model.load_weights(os.path.join("models", weight_file), by_name=True)

    print(MODEL.__name__)
    train_gen = ImageDataGenerator(
        featurewise_std_normalization=featurewise_std_normalization,
        samplewise_std_normalization=False,
        rotation_range=10.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.1,
        zoom_range=0.1,
    )
    gen = ImageDataGenerator(
        featurewise_std_normalization=featurewise_std_normalization,
        samplewise_std_normalization=False,
    )

    batch_size = 64
    train_generator = train_gen.flow_from_directory(os.path.join(dir, 'train'), target_size=image_size, shuffle=False, batch_size=batch_size)
    print("subdior to train type {}".format(train_generator.class_indices))
    valid_generator = gen.flow_from_directory(os.path.join(dir, 'valid'), target_size=image_size, shuffle=False, batch_size=batch_size)
    print("subdior to valid type {}".format(valid_generator.class_indices))

    print("predict_generator train {}".format(math.ceil(train_generator.samples // batch_size + 1)))
    train = model.predict(train_generator, steps=math.ceil(train_generator.samples // batch_size + 1))
    print("train: {}".format(train.shape))
    print("predict_generator valid {}".format(math.ceil(valid_generator.samples // batch_size + 1)))
    valid = model.predict(valid_generator, steps=math.ceil(valid_generator.samples // batch_size + 1))
    print("valid: {}".format(valid.shape))

    print("begin create database {}".format(MODEL.__name__))
    with h5py.File(os.path.join("models", tag, "bottleneck_{}.h5".format(MODEL.__name__)), 'w') as h:
        h.create_dataset("train", data=train)
        h.create_dataset("valid", data=valid)
        h.create_dataset("label", data=train_generator.classes)
        h.create_dataset("valid_label", data=valid_generator.classes)
    print("Data saved to bottleneck_{}.h5 successfully.".format(MODEL.__name__))


def write_gap_test(tag, MODEL, weight_file, image_size, lambda_func=None, featurewise_std_normalization=True):
    # 输入张量
    input_tensor = Input((*image_size, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)  # 使用lambda函数进行额外的预处理
    # 加载基本模型，不包括顶部的全连接层
    base_model = MODEL(input_tensor=x, weights=None, include_top=False)
    # 构建最终的模型，使用全局平均池化
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    # 加载权重文件
    model.load_weights("models/" + weight_file, by_name=True)

    print(MODEL.__name__)
    
    # 定义测试数据的生成器
    gen = ImageDataGenerator(
        featurewise_std_normalization=featurewise_std_normalization,
        samplewise_std_normalization=False,
    )
    
    batch_size = 64
    # 加载测试数据
    test_generator = gen.flow_from_directory(
        os.path.join(dir, 'test'),
        target_size=image_size,
        shuffle=False,
        batch_size=batch_size,
        class_mode=None  # 不使用标签
    )
    
    print("predict_generator test {}".format(math.ceil(test_generator.samples // batch_size + 1)))
    
    # 预测测试数据
    test = model.predict(test_generator, steps=math.ceil(test_generator.samples // batch_size + 1))
    print("test: {}".format(test.shape))

    # 将预测结果保存到文件
    print("begin create database {}".format(MODEL.__name__))
    with h5py.File(os.path.join("models", tag, "bottleneck_{}_test.h5".format(MODEL.__name__)),'w') as h:
        h.create_dataset("test", data=test)
    
    print("write_gap_test {} succeeded".format(MODEL.__name__))

def normal_preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2
    return x

###
### subdir = noscale
###

print("===== Train & Valid =====")
tag = "myt00"
# write_gap(tag, ResNet50, resnet50_weight_file, (240, 320))
# write_gap(tag, Xception, xception_weight_file, (320, 480), xception_preprocess_input)
write_gap(tag, InceptionV3, inceptionV3_weight_file, (320, 480), inception_v3_preprocess_input)

print("===== Test =====")
tag = "myt00"
write_gap_test(tag, ResNet50, resnet50_weight_file, (240, 320))
write_gap_test(tag, Xception, xception_weight_file, (320, 480), xception_preprocess_input)
write_gap_test(tag, InceptionV3, inceptionV3_weight_file, (320, 480), inception_v3_preprocess_input)