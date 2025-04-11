drop table prediction_driver_stats;
drop table prediction_stats;
# 状态表(预测总表)，每次预测都记录在这张表中
# 暂时好像用不到
CREATE TABLE prediction_stats (
    id INT AUTO_INCREMENT PRIMARY KEY,
    driver_id INT NOT NULL,
    driver_name VARCHAR(50) NOT NULL,
    prediction_time DATETIME NOT NULL,
    c0_safe_driving INT DEFAULT 0,
    c1_texting_right INT DEFAULT 0,
    c2_phone_right INT DEFAULT 0,
    c3_texting_left INT DEFAULT 0,
    c4_phone_left INT DEFAULT 0,
    c5_adjust_radio INT DEFAULT 0,
    c6_drinking INT DEFAULT 0,
    c7_reaching_back INT DEFAULT 0,
    c8_hair_makeup INT DEFAULT 0,
    c9_talking_to_passenger INT DEFAULT 0
);
# 司机预测表，包含司机id，预测时间，预测状态
CREATE TABLE driver_prediction_stats (
    id INT PRIMARY KEY AUTO_INCREMENT,
    driver_id INT NOT NULL,
    prediction_time DATETIME NOT NULL,
    predicted_class VARCHAR(50) NOT NULL
);
# 司机表，记录司机的登录信息
CREATE TABLE drivers (
    id INT PRIMARY KEY AUTO_INCREMENT,
    driver_id INT NOT NULL,
    driver_name VARCHAR(50) NOT NULL,
    username VARCHAR(50) UNIQUE NULL,
    pass_word VARCHAR(50) NOT NULL
);
INSERT INTO drivers (driver_id, driver_name, username, pass_word)
VALUES (1001, '张三', 'zhangsan', 'password123'),
    (1002, '李四', 'lisi', 'safe456'),
    (1003, '王五', 'wangwu', 'secure789'),
    (1004, '赵六', 'zhaoliu', 'driver123'),
    (1005, '钱七', 'qianqi', 'pass789');