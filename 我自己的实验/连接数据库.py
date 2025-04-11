import pymysql
from datetime import datetime

import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50, Xception, InceptionV3
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xcep_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as incep_preprocess
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import threading

# 初始化模型和权重，其中权重为none
image_size = 224
resnet = ResNet50(weights=None, include_top=False, pooling='avg', input_shape=(image_size, image_size, 3))
xception = Xception(weights=None, include_top=False, pooling='avg', input_shape=(image_size, image_size, 3))
inception = InceptionV3(weights=None, include_top=False, pooling='avg', input_shape=(image_size, image_size, 3))
# 将模型的权重加载到模型中
resnet.load_weights(r"F:\JupyterWorkSpace\BBBBBBS\models\resnet50-imagenet-finetune152.h5", by_name=True)
xception.load_weights(r"F:\JupyterWorkSpace\BBBBBBS\models\xception-imagenet-finetune116.h5", by_name=True)
inception.load_weights(r"F:\JupyterWorkSpace\BBBBBBS\models\inceptionV3-imagenet-finetune172.h5", by_name=True)
# 加载混合分类器
model_mix = load_model(r"F:\JupyterWorkSpace\BBBBBBS\models\mixed-model.h5")

# 分类标签
classes = ['c0安全驾驶', 'c1右手打字', 'c2右手接电话','c3左手打字','c4左手接电话',
           'c5调收音机','c6喝饮料','c7拿后面的东西','c8整理头发和化妆','c9和其他乘客说话']


def insert_prediction_to_db(driverID, class_name, prediction=None, driver_name=None):
    """
    将预测结果写入数据库（两种表）
    参数:
        driverID: 司机ID
        class_name: 预测类别名称
        prediction: 所有类别的概率分布数组(可选)
        driver_name: 司机姓名(可选)
    """
    try:
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='czy123',
            database='driver_act',
            charset='utf8mb4'
        )
        with conn.cursor() as cursor:
            # 1. 写入简单预测记录
            sql_simple = """
                INSERT INTO driver_prediction_stats 
                (driver_id, prediction_time, predicted_class) 
                VALUES (%s, %s, %s)
            """
            cursor.execute(sql_simple, (driverID, datetime.now(), class_name))
            
            # 2. 如果有详细预测数据，写入详细统计表
            if prediction is not None and len(prediction) == 10:
                # 将预测概率数组转为字典
                stats = {
                    'c0': int(prediction[0]),
                    'c1': int(prediction[1]),
                    'c2': int(prediction[2]),
                    'c3': int(prediction[3]),
                    'c4': int(prediction[4]),
                    'c5': int(prediction[5]),
                    'c6': int(prediction[6]),
                    'c7': int(prediction[7]),
                    'c8': int(prediction[8]),
                    'c9': int(prediction[9])
                }
                
                # 如果未提供driver_name，尝试从数据库查询
                if driver_name is None:
                    cursor.execute("SELECT driver_name FROM drivers WHERE driver_id = %s", (driverID,))
                    result = cursor.fetchone()
                    driver_name = result[0] if result else "未知司机"
                
                sql_detail = """
                    INSERT INTO prediction_stats (
                        prediction_time, driver_id, driver_name,
                        c0_safe_driving, c1_texting_right, c2_phone_right,
                        c3_texting_left, c4_phone_left, c5_adjust_radio,
                        c6_drinking, c7_reaching_back, c8_hair_makeup, c9_talking_to_passenger
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = (
                    datetime.now(), driverID, driver_name,
                    stats['c0'], stats['c1'], stats['c2'],
                    stats['c3'], stats['c4'], stats['c5'],
                    stats['c6'], stats['c7'], stats['c8'], stats['c9']
                )
                cursor.execute(sql_detail, values)
                
        conn.commit()
        print(f"✅ 数据已写入数据库：{class_name}")
    except Exception as e:
        print(f"❌ 数据库写入失败：{e}")
    finally:
        if 'conn' in locals() and conn.open:
            conn.close()

# 修改后的预测函数
def predict_single_image(driverID, image_path):
    # 加载图像和预测代码保持不变...
        # 加载图像
    img = load_img(image_path, target_size=(image_size, image_size))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # 提取特征
    feat_resnet = resnet.predict(resnet_preprocess(np.copy(img_array)))
    feat_xcep = xception.predict(xcep_preprocess(np.copy(img_array)))
    feat_incep = inception.predict(incep_preprocess(np.copy(img_array)))

    # 拼接特征
    bottleneck_feature = np.concatenate([feat_resnet, feat_xcep, feat_incep], axis=1)

    # 分类预测
    prediction = model_mix.predict(bottleneck_feature)
    class_idx = np.argmax(prediction[0])
    class_name = classes[class_idx]

    print("预测概率分布：", prediction[0])
    print("预测类别索引：", class_idx)
    print("预测概率：",prediction[0][class_idx])
    print("预测类别名称：", class_name)
    # 在预测后调用更新后的插入函数
    insert_prediction_to_db(
        driverID=driverID,
        class_name=class_name,
        prediction=prediction[0],  # 传递整个概率分布数组
        driver_name=None  # 可以留空，函数会尝试从数据库查询
    )