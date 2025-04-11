import tensorflow as tf
print("TF版本:", tf.__version__)
print("GPU可用:", tf.test.is_gpu_available())
print("设备列表:", tf.config.list_physical_devices())
import tensorflow.keras
print(dir(tensorflow.keras))  # 查看是否有 `models` 模块