import argparse
import time
import sys
import numpy as np
import cv2
import mediapipe as mp
import pyrealsense2 as rs
from PyQt5 import QtWidgets
import pyqtgraph.opengl as gl
from pythonosc import udp_client

OSC_ADDRESS = "/mediapipe/pose"
VISIBILITY_THRESHOLD = 0.5

def send_pose(client, landmark_list):
    if landmark_list is None:
        client.send_message(OSC_ADDRESS, 0)
        return

    # 创建消息并发送
    builder = udp_client.OscMessageBuilder(address=OSC_ADDRESS)
    builder.add_arg(1)
    for landmark in landmark_list.landmark:
        builder.add_arg(landmark.x)
        builder.add_arg(landmark.y)
        builder.add_arg(landmark.z)
        builder.add_arg(landmark.visibility)
    msg = builder.build()
    client.send(msg)

def draw_dashed_line(img, pt1, pt2, color, thickness=1, gap=5):
    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
    if dist == 0:
        return
    dash_len = 10
    num_dashes = int(dist / (dash_len + gap))
    for i in range(num_dashes):
        start_frac = (i * (dash_len + gap)) / dist
        end_frac = ((i * (dash_len + gap)) + dash_len) / dist
        start = (int(pt1[0] + (pt2[0] - pt1[0]) * start_frac), int(pt1[1] + (pt2[1] - pt1[1]) * start_frac))
        end = (int(pt1[0] + (pt2[0] - pt1[0]) * end_frac), int(pt1[1] + (pt2[1] - pt1[1]) * end_frac))
        cv2.line(img, start, end, color, thickness)

def process_landmarks(results, width, height):
    landmark_points_2d = []
    landmark_visibility = []
    for landmark in results.pose_landmarks.landmark:
        x_px = int(landmark.x * width)
        y_px = int(landmark.y * height)
        landmark_points_2d.append((x_px, y_px))
        landmark_visibility.append(landmark.visibility)
    return landmark_points_2d, landmark_visibility

def draw_landmarks(image, landmark_points_2d, landmark_visibility, mp_pose):
    # 绘制关键点和连接
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        pt1 = landmark_points_2d[start_idx]
        pt2 = landmark_points_2d[end_idx]
        vis1 = landmark_visibility[start_idx]
        vis2 = landmark_visibility[end_idx]

        if vis1 >= VISIBILITY_THRESHOLD and vis2 >= VISIBILITY_THRESHOLD:
            # 两个点都可见，使用正常颜色绘制
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        else:
            # 使用红色虚线绘制
            draw_dashed_line(image, pt1, pt2, (0, 0, 255), 2)

    # 绘制关键点
    for idx, (pt, vis) in enumerate(zip(landmark_points_2d, landmark_visibility)):
        if vis >= VISIBILITY_THRESHOLD:
            cv2.circle(image, pt, 5, (0, 255, 0), -1)
        else:
            cv2.circle(image, pt, 5, (0, 0, 255), -1)

def transform_coordinates(point_3d):
    # 原始坐标
    x, y, z = point_3d[0], point_3d[1], point_3d[2]
    # 坐标变换
    x_new = x
    y_new = -y
    z_new = -z
    return [x_new, y_new, z_new]

def update_3d_plot(landmark_points, last_landmark_points, results, scatter_plot, connections, mp_pose):
    # 移除 None 值，准备绘制
    valid_points = [pt for pt in landmark_points if pt is not None]
    if valid_points:
        # 更新 3D 散点图
        scatter_plot.setData(pos=np.array(valid_points), size=5, color=(1, 0, 0, 1))

        # 更新骨架连接
        for i, connection in enumerate(mp_pose.POSE_CONNECTIONS):
            start_idx, end_idx = connection
            pt1 = landmark_points[start_idx]
            pt2 = landmark_points[end_idx]

            if pt1 is not None and pt2 is not None:
                # 判断连接的两个点是否在当前帧可见
                vis1 = results.pose_landmarks.landmark[start_idx].visibility
                vis2 = results.pose_landmarks.landmark[end_idx].visibility

                if vis1 >= VISIBILITY_THRESHOLD and vis2 >= VISIBILITY_THRESHOLD:
                    # 使用正常颜色绘制
                    color = (0, 1, 0, 1)  # 绿色
                else:
                    # 使用红色绘制
                    color = (1, 0, 0, 1)  # 红色

                connection_points = np.array([pt1, pt2])
                connections[i].setData(pos=connection_points, color=color)
            else:
                # 保持上一帧的连接
                pass
    else:
        # 没有有效的点，不更新绘制
        pass

def main():
    # 读取参数
    parser = argparse.ArgumentParser()
    rs_group = parser.add_argument_group("RealSense")
    rs_group.add_argument("--resolution", default=[640, 480], type=int, nargs=2, metavar=('width', 'height'),
                          help="Resolution of the realsense stream.")
    rs_group.add_argument("--fps", default=30, type=int, help="Framerate of the realsense stream.")

    mp_group = parser.add_argument_group("MediaPipe")
    mp_group.add_argument("--model-complexity", default=1, type=int, help="Set model complexity (0=Light, 1=Full, 2=Heavy).")
    mp_group.add_argument("--no-smooth-landmarks", action="store_false", help="Disable landmark smoothing.")
    mp_group.add_argument("--static-image-mode", action="store_true", help="Enables static image mode.")
    mp_group.add_argument("-mdc", "--min-detection-confidence", type=float, default=0.5, help="Minimum detection confidence.")
    mp_group.add_argument("-mtc", "--min-tracking-confidence", type=float, default=0.5, help="Minimum tracking confidence.")

    nw_group = parser.add_argument_group("Network")
    nw_group.add_argument("--ip", default="127.0.0.1", help="The IP of the OSC server")
    nw_group.add_argument("--port", type=int, default=7400, help="The port the OSC server is listening on")

    args = parser.parse_args()

    # 创建OSC客户端
    client = udp_client.SimpleUDPClient(args.ip, args.port)

    # 设置MediaPipe
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        smooth_landmarks=args.no_smooth_landmarks,
        static_image_mode=args.static_image_mode,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence)

    # 创建RealSense管道
    pipeline = rs.pipeline()

    width, height = args.resolution

    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, args.fps)

    profile = pipeline.start(config)

    # 获取相机内参
    intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    # 对齐深度帧到彩色帧
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 初始化 PyQt 应用程序
    app = QtWidgets.QApplication(sys.argv)

    # 创建一个 OpenGL 的窗口
    window = gl.GLViewWidget()
    window.show()
    window.setWindowTitle('3D Pose Visualization')

    # 设置视角，使画面放大（拉近）
    window.setCameraPosition(distance=3.0, elevation=20, azimuth=90)  # 根据需要调整参数

    # 添加坐标轴
    axes = gl.GLAxisItem()
    axes.setSize(x=1, y=1, z=1)  # 设置坐标轴的长度
    window.addItem(axes)

    # 创建用于存储关键点的 3D 散点图
    scatter_plot = gl.GLScatterPlotItem()
    window.addItem(scatter_plot)

    # 创建用于存储骨架连接的线条
    connections = []
    for _ in mp_pose.POSE_CONNECTIONS:
        line = gl.GLLinePlotItem(width=2)
        window.addItem(line)
        connections.append(line)

    prev_frame_time = 0

    # 初始化上一次的关键点位置
    last_landmark_points = [None] * 33

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # 转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 深度图可视化
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # 处理彩色图像
            image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            # 发送姿态数据
            send_pose(client, results.pose_landmarks)

            # 准备在图像上绘制
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 自定义绘制函数
            if results.pose_landmarks:
                landmark_points_2d, landmark_visibility = process_landmarks(results, width, height)
                draw_landmarks(image, landmark_points_2d, landmark_visibility, mp_pose)
            else:
                # 如果没有检测到人体，可以选择在图像上显示提示
                cv2.putText(image, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

            # 获取3D坐标点并更新3D显示
            if results.pose_landmarks:
                landmark_points = []

                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    visibility = landmark.visibility

                    if visibility >= VISIBILITY_THRESHOLD:
                        # 将归一化坐标转换为像素坐标
                        x_px = int(landmark.x * width)
                        y_px = int(landmark.y * height)

                        # 确保像素坐标在图像范围内
                        x_px = min(max(x_px, 0), width - 1)
                        y_px = min(max(y_px, 0), height - 1)

                        # 获取深度值
                        depth = depth_frame.get_distance(x_px, y_px)

                        # 过滤无效深度值
                        if depth <= 0:
                            # 使用上一帧的坐标
                            landmark_points.append(last_landmark_points[idx])
                            continue

                        # 将像素坐标和深度值转换为三维坐标
                        point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x_px, y_px], depth)

                        # 坐标变换
                        transformed_point = transform_coordinates(point_3d)
                        landmark_points.append(transformed_point)

                        # 更新最后的关键点位置
                        last_landmark_points[idx] = transformed_point
                    else:
                        # 不可见，使用上一帧的坐标
                        landmark_points.append(last_landmark_points[idx])

                # 更新3D绘图
                update_3d_plot(landmark_points, last_landmark_points, results, scatter_plot, connections, mp_pose)
            else:
                # 没有检测到人体，保持上一帧的显示
                pass

            # 处理 PyQt 的事件循环
            QtWidgets.QApplication.processEvents()

            # 计算FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_frame_time + 1e-5)  # 防止除零
            prev_frame_time = current_time

            # 在图像上显示FPS
            cv2.putText(image, f"FPS: {fps:.0f}", (7, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)

            # 显示彩色图像（带有自定义绘制的骨骼）
            cv2.imshow('RealSense Pose Detector', image)

            # 显示深度图像
            cv2.imshow('Depth Map', depth_colormap)

            # 按下ESC键退出
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        pose.close()
        pipeline.stop()

if __name__ == "__main__":
    main()
