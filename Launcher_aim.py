# import cv2
# import numpy as np
# import serial
# import time
# import sys
#
# # 硬件配置
# DEFAULT_PORT = '/dev/ttyS1'  # 常见USB串口设备
# # DEFAULT_PORT = '/dev/ttyACM0' # 适用于某些Arduino设备
# BAUD_RATE = 115200
#
# # 视觉参数
# FRAME_WIDTH = 1920
# FRAME_HEIGHT = 1080
# MIN_AREA_RATIO = 0.0002  # 目标最小面积比例
# CIRCULARITY_THRESH = 0.6  # 圆形度阈值
# MAX_DISPLACEMENT = 200  # 最大允许帧间位移（像素），可根据实际情况调整
#
# # 优化颜色阈值 (HSV)
# lower_green = np.array([50, 0, 100])  # 降低亮度和饱和度下限
# upper_green = np.array([200, 255, 255])  # 缩小色相范围，降低亮度上限
#
#
# def setup_camera():
#     cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
#     if not cap.isOpened():
#         raise RuntimeError("摄像头初始化失败，请检查：\n1. 摄像头连接\n2. 用户权限（video组）\n3. 其他正在使用摄像头的程序")
#     cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
#     # 固定摄像头参数
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
#     cap.set(cv2.CAP_PROP_FPS,30)  # 设置帧率为30
#     cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 手动曝光模式
#     cap.set(cv2.CAP_PROP_EXPOSURE, 17)
#     cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # 关闭自动白平衡
#
#     return cap
#
#
# def detect_green_dot(frame, prev_center,last_counters):
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, lower_green, upper_green)
#     ret, binary = cv2.threshold(mask, 160, 255, cv2.THRESH_BINARY)
#     counters, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#
#     #cv2.imshow('binary', binary)
#     #cv2.imshow('frame',frame)
#     #cv2.waitKey(0)
#
#     def cal_circle_rate(contour):
#         if len(contour) < 5:
#             return 0
#         (x, y), radius = cv2.minEnclosingCircle(contour)
#         area = cv2.contourArea(contour)
#         circle_area = 3.1415926 * radius * radius
#         return area / circle_area
#
#
#     if prev_center == (2000, 999):
#         return (0,999),counters
#     max1 = 0
#     max_index = -1
#     min_x=-1
#     min_y=-1
#     min_distance=2000
#     if prev_center == (0, 999):
#         for i in range(len(counters)):
#             contour = counters[i]
#             circle_rate = cal_circle_rate(contour)
#             (x, y), radius = cv2.minEnclosingCircle(contour)
#             if circle_rate <CIRCULARITY_THRESH or y>500 or y<150:
#                 continue
#             current_center = (int(x), int(y))
#             for j in range(len(last_counters)):
#                 last_contour = last_counters[j]
#                 circle_rate = cal_circle_rate(last_contour)
#                 (cx, cy), last_radius = cv2.minEnclosingCircle(last_contour)
#                 if circle_rate <CIRCULARITY_THRESH or cy>500 or cy<150:
#                   continue
#                 dx = current_center[0] - cx
#                 dy = current_center[1] - cy
#                 distance = np.sqrt(dx ** 2 + dy ** 2)
#                 if distance < min_distance and cy<500 and cy>150:
#                     min_distance = distance
#                     min_x=int(cx)
#                     min_y=int(cy)
#         if min_x==-1 and min_y==-1:
#           return (2000,999),counters
#         return (min_x, min_y),counters
#     for i in range(len(counters)):
#         contour = counters[i]
#         (x, y), radius = cv2.minEnclosingCircle(contour)
#         current_center = (int(x), int(y))
#
#         # 帧间位移检查（仅当上一帧有有效中心时）
#         if prev_center != (2000, 999):
#             dx = current_center[0] - prev_center[0]
#             dy = current_center[1] - prev_center[1]
#             distance = np.sqrt(dx ** 2 + dy ** 2)
#             if distance > MAX_DISPLACEMENT:
#                 continue  # 位移过大，跳过该轮廓
#
#         circle_rate = cal_circle_rate(contour)
#         if circle_rate > max1 and y<500 and y>150:
#             max1 = circle_rate
#             max_index = i
#     if max_index != -1:
#         (x, y), radius = cv2.minEnclosingCircle(counters[max_index])
#         return (int(x), int(y)),counters
#     else:
#         return (2000, 999),counters  # 无效坐标表示未检测到有效目标
#
#
# def setup_serial(port):
#     try:
#         ser = serial.Serial(port, BAUD_RATE)
#         time.sleep(2)  # 等待串口稳定
#         print(f"成功连接到 {port}")
#         return ser
#     except Exception as e:
#         print(f"连接失败: {str(e)}\n请检查：\n1. 设备是否存在\n2. 用户权限(dialout组)\n3. 波特率设置")
#         return None
#
#
# def format_coordinates(cx):
#     """统一坐标格式化（4位宽度）"""
#     return f"{cx:04d}"
#
#
# def main():
#     print("Linux版绿点追踪系统启动")
#     ser = setup_serial(DEFAULT_PORT)
#     cap = setup_camera()
#     prev_center = (2000, 999)  # 初始化上一帧中心为无效坐标
#     last_counters = []
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("视频流中断")
#                 break
#
#             current_center,last_counters= detect_green_dot(frame, prev_center,last_counters)
#
#
#             cx, cy = current_center
#             formatted_cx = format_coordinates(cx)
#             formatted_cy = format_coordinates(cy)
#             data = f"{formatted_cx},{formatted_cy}\n"
#
#             if ser and ser.is_open:
#                 try:
#                     ser.write(data.encode())
#                     print(f"坐标发送: {data.strip()}")
#                 except serial.SerialException as e:
#                     print(f"串口错误: {e}")
#                     ser.close()
#                     ser = None
#             else:
#                 print(f"模拟发送: {data.strip()}")
#
#             prev_center = current_center  # 更新有效中心
#
#
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#     finally:
#         cap.release()
#         if ser and ser.is_open:
#             ser.close()
#         cv2.destroyAllWindows()
#         print("系统已安全关闭")
#
#
# if __name__ == "__main__":
#     main()
import cv2
import numpy as np
import serial
import time
import sys
import logging
from collections import deque

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 硬件配置
DEFAULT_PORT = '/dev/ttyS1'
BAUD_RATE = 115200

# 视觉参数
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
MIN_AREA_RATIO = 0.0002
CIRCULARITY_THRESH = 0.7
MAX_DISPLACEMENT = 200
MIN_RADIUS = 10
MAX_RADIUS = 100

# 优化颜色阈值（HSV）
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])

# 预处理参数
MORPH_KERNEL_SIZE = 5  # 形态学操作核大小
GAUSSIAN_BLUR_SIZE = (5, 5)  # 高斯模糊核大小
MEDIAN_BLUR_SIZE = 5  # 中值滤波大小
CLOSE_ITERATIONS = 2  # 闭运算迭代次数
OPEN_ITERATIONS = 1  # 开运算迭代次数

# 跟踪参数
TRACKING_HISTORY = 5  # 跟踪历史长度
SMOOTHING_FACTOR = 0.7  # 平滑因子

# 无效坐标常量
INVALID_COORD = (2000, 999)


class GreenDotTracker:
    def __init__(self, port=DEFAULT_PORT, baud_rate=BAUD_RATE):
        self.port = port
        self.baud_rate = baud_rate
        self.ser = None
        self.cap = None
        self.prev_center = INVALID_COORD
        self.last_counters = []

        # 跟踪历史
        self.track_history = deque(maxlen=TRACKING_HISTORY)
        self.smoothed_center = INVALID_COORD

        # 性能统计
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()

    #图像预处理流程
    def preprocess_frame(self, frame):

        # 1. 降噪处理
        # 高斯滤波去除高斯噪声
        blurred = cv2.GaussianBlur(frame, GAUSSIAN_BLUR_SIZE, 0)
        # 中值滤波去除椒盐噪声
        denoised = cv2.medianBlur(blurred, MEDIAN_BLUR_SIZE)

        # 2. 对比度增强（CLAHE）
        # 转换为LAB色彩空间
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # 合并通道并转回BGR
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # 3. 锐化处理
        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]]) / 9
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)

        return sharpened
        #改进的颜色分割
    def color_segmentation(self, frame):

        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 创建颜色掩码
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # 形态学操作：去除噪声和填充空洞
        # 创建形态学核
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))

        # 先开运算去除小噪声
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=OPEN_ITERATIONS)

        # 再闭运算填充空洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERATIONS)

        return mask
        #增强二值图像
    def enhance_binary(self, binary):

        # 距离变换，增强圆形特征
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # 归一化距离变换
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

        # 自适应阈值增强圆形区域
        enhanced = (dist_transform > 0.5).astype(np.uint8) * 255

        return enhanced

        #计算圆形度
    def calculate_circularity(self, contour):

        if len(contour) < 5:
            return 0

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            return 0


        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return circularity

       #检查是否为有效目标
    def is_valid_target(self, contour, center_y):

        # 计算面积
        area = cv2.contourArea(contour)
        if area < FRAME_WIDTH * FRAME_HEIGHT * MIN_AREA_RATIO:
            return False

        # 计算圆形度
        circularity = self.calculate_circularity(contour)
        if circularity < CIRCULARITY_THRESH:
            return False

        # 检查半径范围
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < MIN_RADIUS or radius > MAX_RADIUS:
            return False

        # 检查Y坐标范围
        if center_y < 150 or center_y > 500:
            return False

        return True

        #根据面积过滤轮廓
    def filter_by_area(self, contours):

        min_area = FRAME_WIDTH * FRAME_HEIGHT * MIN_AREA_RATIO
        return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        #平滑坐标输出
    def smooth_coordinates(self, new_center):

        if new_center == INVALID_COORD:
            self.track_history.clear()
            return INVALID_COORD

        # 添加到历史记录
        self.track_history.append(new_center)

        if len(self.track_history) < 2:
            return new_center

        # 加权平均平滑
        weights = np.exp(np.linspace(-1, 0, len(self.track_history)))
        weights /= weights.sum()

        smooth_x = sum(w * c[0] for w, c in zip(weights, self.track_history))
        smooth_y = sum(w * c[1] for w, c in zip(weights, self.track_history))

        # 指数移动平均
        if self.smoothed_center != INVALID_COORD:
            smooth_x = SMOOTHING_FACTOR * smooth_x + (1 - SMOOTHING_FACTOR) * self.smoothed_center[0]
            smooth_y = SMOOTHING_FACTOR * smooth_y + (1 - SMOOTHING_FACTOR) * self.smoothed_center[1]

        self.smoothed_center = (int(smooth_x), int(smooth_y))
        return self.smoothed_center

    #检测绿色目标（带预处理）
    def detect_green_dot(self, frame):

        # 1. 图像预处理
        preprocessed = self.preprocess_frame(frame)

        # 2. 颜色分割
        mask = self.color_segmentation(preprocessed)

        # 3. 增强二值图像
        binary = self.enhance_binary(mask)

        # 4. 查找轮廓
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5. 过滤小面积轮廓
        contours = self.filter_by_area(contours)

        if not contours:
            return INVALID_COORD, contours

        # 6. 选择最佳目标
        best_contour = None
        best_score = -1

        for contour in contours:
            # 计算最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center_y = int(y)

            # 有效性检查
            if not self.is_valid_target(contour, center_y):
                continue

            current_center = (int(x), int(y))

            # 帧间位移检查
            if self.prev_center != INVALID_COORD:
                dx = current_center[0] - self.prev_center[0]
                dy = current_center[1] - self.prev_center[1]
                distance = np.sqrt(dx ** 2 + dy ** 2)
                if distance > MAX_DISPLACEMENT:
                    continue

            # 计算综合得分（圆形度 + 面积归一化）
            circularity = self.calculate_circularity(contour)
            area = cv2.contourArea(contour)
            area_score = area / (FRAME_WIDTH * FRAME_HEIGHT)
            score = circularity * 0.7 + area_score * 0.3

            if score > best_score:
                best_score = score
                best_contour = contour

        if best_contour is not None:
            (x, y), radius = cv2.minEnclosingCircle(best_contour)
            center = (int(x), int(y))

            # 坐标平滑
            smoothed_center = self.smooth_coordinates(center)
            return smoothed_center, contours

        # 没有找到有效目标，清空历史
        self.track_history.clear()
        self.smoothed_center = INVALID_COORD
        return INVALID_COORD, contours

        #初始化摄像头
    def setup_camera(self):

        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                raise RuntimeError("摄像头初始化失败")

            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, 17)
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)

            # 等待摄像头稳定
            time.sleep(2)

            logging.info(f"摄像头初始化成功: {FRAME_WIDTH}x{FRAME_HEIGHT}")
            return True
        except Exception as e:
            logging.error(f"摄像头初始化失败: {e}")
            return False

        #初始化串口
    def setup_serial(self):

        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)
            logging.info(f"串口连接成功: {self.port}")
            return True
        except Exception as e:
            logging.error(f"串口连接失败: {e}")
            return False

        #计算FPS
    def calculate_fps(self):

        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_time

        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = current_time
            logging.debug(f"当前FPS: {self.fps:.2f}")

        return self.fps

        #绘制可视化信息
    def draw_visualization(self, frame, center, contours):

        # 绘制所有检测到的轮廓
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        # 绘制目标中心点
        if center != INVALID_COORD:
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.circle(frame, center, 15, (0, 0, 255), 2)

            # 显示坐标
            cv2.putText(frame, f"({center[0]}, {center[1]})",
                        (center[0] + 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 显示FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示状态
        status = "Target Found" if center != INVALID_COORD else "No Target"
        color = (0, 255, 0) if center != INVALID_COORD else (0, 0, 255)
        cv2.putText(frame, status, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame

        #统一坐标格式化
    def format_coordinates(self, cx, cy):

        return f"{cx:04d},{cy:04d}\n"

        #主循环
    def run(self):

        # 初始化硬件
        if not self.setup_camera():
            return

        if not self.setup_serial():
            logging.warning("串口初始化失败，将运行在模拟模式")

        try:
            while True:
                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    logging.error("视频流中断")
                    break

                # 计算FPS
                self.calculate_fps()

                # 检测目标
                current_center, contours = self.detect_green_dot(frame)

                # 更新上一帧中心
                if current_center != INVALID_COORD:
                    self.prev_center = current_center
                else:
                    self.prev_center = INVALID_COORD

                # 发送坐标
                if current_center != INVALID_COORD:
                    cx, cy = current_center
                    data = self.format_coordinates(cx, cy)

                    if self.ser and self.ser.is_open:
                        try:
                            self.ser.write(data.encode())
                            logging.debug(f"坐标发送: {data.strip()}")
                        except serial.SerialException as e:
                            logging.error(f"串口错误: {e}")
                            self.ser.close()
                            self.ser = None
                    else:
                        logging.info(f"模拟发送: {data.strip()}")

                # 可视化
                vis_frame = self.draw_visualization(frame.copy(), current_center, contours)
                cv2.imshow('Green Dot Tracker', vis_frame)

                # 按键退出
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):  # 按's'保存当前帧
                    cv2.imwrite(f'screenshot_{time.time()}.jpg', frame)
                    logging.info("截图已保存")

        except KeyboardInterrupt:
            logging.info("程序被用户中断")
        finally:
            self.cleanup()

        #清理资源
    def cleanup(self):

        if self.cap:
            self.cap.release()
        if self.ser and self.ser.is_open:
            self.ser.close()
        cv2.destroyAllWindows()
        logging.info("系统已安全关闭")

    #主函数
def main():

    print("Linux版绿点追踪系统启动（带预处理）")
    print("控制说明:")
    print("  - 'q' 键: 退出程序")
    print("  - 's' 键: 保存当前帧截图")

    tracker = GreenDotTracker()
    tracker.run()


if __name__ == "__main__":
    main()