# !/usr/bin/env python
"""
此代码主要用于收集数据示意（简易版）
对应博文的详细教程在 https://www.cnblogs.com/kin-zhang/p/16057173.html

0. 此示意引用参考全部总结与博客之中 再次不做赘述
1. 记得修改一下保存数据的绝对路径
2. 检查CARLA PORT是否和设置的是一致的 免得连接不上
3. 相关注释基本都没咋写 具体可以看看博客
"""
import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import numpy as np
import cv2
from queue import Queue, Empty
import copy
import random

random.seed(0)

# args
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
parser.add_argument('--port', '-p', default=2000, type=int, help='TCP port to listen to (default: 2000)')
parser.add_argument('--tm_port', default=8000, type=int, help='Traffic Manager Port (default: 8000)')
parser.add_argument('--ego-spawn', type=list, default=None, help='[x,y] in world coordinate')
parser.add_argument('--top-view', default=True, help='Setting spectator to top view on ego car')
parser.add_argument('--map', default='Town01', help='Town Map')
parser.add_argument('--sync', default=True, help='Synchronous mode execution')
parser.add_argument('--sensor-h', default=2.4, help='Sensor Height')
# 给绝对路径 记得改位置哦！
parser.add_argument('--save-path', default='/home/cpsgroup/trust_aware_hrc/sensor_data/', help='Synchronous mode execution')
args = parser.parse_args()

# 图片大小可自行修改
IM_WIDTH = 256
IM_HEIGHT = 256

actor_list, sensor_list = [], []
sensor_type = ['rgb', 'lidar', 'imu', 'gnss']


def main(args):
    # We start creating the client
    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)

    # world = client.get_world()
    world = client.load_world('Town01')
    blueprint_library = world.get_blueprint_library()
    try:
        original_settings = world.get_settings()
        settings = world.get_settings()

        # We set CARLA syncronous mode
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)
        spectator = world.get_spectator()

        # 手动规定
        # transform_vehicle = carla.Transform(carla.Location(0, 10, 0), carla.Rotation(0, 0, 0))
        # 自动选择
        transform_vehicle = random.choice(world.get_map().get_spawn_points())
        ego_vehicle = world.spawn_actor(random.choice(blueprint_library.filter("model3")), transform_vehicle)
        actor_list.append(ego_vehicle)

        # -------------------------- 进入传感器部分 --------------------------#
        sensor_queue = Queue()
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        imu_bp = blueprint_library.find('sensor.other.imu')
        gnss_bp = blueprint_library.find('sensor.other.gnss')

        # 可以设置一些参数 set the attribute of camera
        cam_bp.set_attribute("image_size_x", "{}".format(IM_WIDTH))
        cam_bp.set_attribute("image_size_y", "{}".format(IM_HEIGHT))
        cam_bp.set_attribute("fov", "60")
        # cam_bp.set_attribute('sensor_tick', '0.1')

        cam01 = world.spawn_actor(cam_bp, carla.Transform(carla.Location(z=args.sensor_h), carla.Rotation(yaw=0)),
                                  attach_to=ego_vehicle)
        cam01.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_front"))
        sensor_list.append(cam01)

        cam02 = world.spawn_actor(cam_bp, carla.Transform(carla.Location(z=args.sensor_h), carla.Rotation(yaw=60)),
                                  attach_to=ego_vehicle)
        cam02.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_right"))
        sensor_list.append(cam02)

        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('points_per_second', '200000')
        lidar_bp.set_attribute('range', '32')
        lidar_bp.set_attribute('rotation_frequency', str(int(1 / settings.fixed_delta_seconds)))  #

        lidar01 = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(z=args.sensor_h)), attach_to=ego_vehicle)
        lidar01.listen(lambda data: sensor_callback(data, sensor_queue, "lidar"))
        sensor_list.append(lidar01)

        imu01 = world.spawn_actor(imu_bp, carla.Transform(carla.Location(z=args.sensor_h)), attach_to=ego_vehicle)
        imu01.listen(lambda data: sensor_callback(data, sensor_queue, "imu"))
        sensor_list.append(imu01)

        gnss01 = world.spawn_actor(gnss_bp, carla.Transform(carla.Location(z=args.sensor_h)), attach_to=ego_vehicle)
        gnss01.listen(lambda data: sensor_callback(data, sensor_queue, "gnss"))
        sensor_list.append(gnss01)
        # -------------------------- 传感器设置完毕 --------------------------#

        # 设置traffic manager
        tm = client.get_trafficmanager(args.tm_port)
        tm.set_synchronous_mode(True)
        # 是否忽略红绿灯
        # tm.ignore_lights_percentage(ego_vehicle, 100)
        # 如果限速30km/h -> 30*(1-10%)=27km/h
        tm.global_percentage_speed_difference(10.0)
        ego_vehicle.set_autopilot(True, tm.get_port())

        while True:
            # Tick the server
            world.tick()

            # 将CARLA界面摄像头跟随车动
            loc = ego_vehicle.get_transform().location
            spectator.set_transform(
                carla.Transform(carla.Location(x=loc.x, y=loc.y, z=35), carla.Rotation(yaw=0, pitch=-90, roll=0)))

            w_frame = world.get_snapshot().frame
            print("\nWorld's frame: %d" % w_frame)
            try:
                rgbs = []

                for i in range(0, len(sensor_list)):
                    s_frame, s_name, s_data = sensor_queue.get(True, 1.0)
                    print("    Frame: %d   Sensor: %s" % (s_frame, s_name))
                    sensor_type = s_name.split('_')[0]
                    if sensor_type == 'rgb':
                        rgbs.append(_parse_image_cb(s_data))
                    elif sensor_type == 'lidar':
                        lidar = _parse_lidar_cb(s_data)
                    elif sensor_type == 'imu':
                        imu_yaw = s_data.compass
                    elif sensor_type == 'gnss':
                        gnss = s_data

                # 仅用来可视化 可注释
                rgb = np.concatenate(rgbs, axis=1)[..., :3]
                cv2.imshow('vizs', visualize_data(rgb, lidar, imu_yaw, gnss))
                cv2.waitKey(100)
                if rgb is None or args.save_path is not None:
                    # 检查是否有各自传感器的文件夹
                    mkdir_folder(args.save_path)

                    filename = args.save_path + 'rgb/' + str(w_frame) + '.png'
                    cv2.imwrite(filename, np.array(rgb[..., ::-1]))
                    filename = args.save_path + 'lidar/' + str(w_frame) + '.npy'
                    np.save(filename, lidar)
                    ###"""

            except Empty:
                print("    Some of the sensor information is missed")

    finally:
        world.apply_settings(original_settings)
        tm.set_synchronous_mode(False)
        for sensor in sensor_list:
            sensor.destroy()
        for actor in actor_list:
            actor.destroy()
        print("All cleaned up!")


def mkdir_folder(path):
    for s_type in sensor_type:
        if not os.path.isdir(os.path.join(path, s_type)):
            os.makedirs(os.path.join(path, s_type))
    return True


def sensor_callback(sensor_data, sensor_queue, sensor_name):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue
    sensor_queue.put((sensor_data.frame, sensor_name, sensor_data))


# modify from world on rail code
def visualize_data(rgb, lidar, imu_yaw, gnss, text_args=(cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)):
    canvas = np.array(rgb[..., ::-1])

    if lidar is not None:
        lidar_viz = lidar_to_bev(lidar).astype(np.uint8)
        lidar_viz = cv2.cvtColor(lidar_viz, cv2.COLOR_GRAY2RGB)
        canvas = np.concatenate([canvas, cv2.resize(lidar_viz.astype(np.uint8), (canvas.shape[0], canvas.shape[0]))],
                                axis=1)

    # cv2.putText(canvas, f'yaw angle: {imu_yaw:.3f}', (4, 10), *text_args)
    # cv2.putText(canvas, f'log: {gnss[0]:.3f} alt: {gnss[1]:.3f} brake: {gnss[2]:.3f}', (4, 20), *text_args)

    return canvas


# modify from world on rail code
def lidar_to_bev(lidar, min_x=-24, max_x=24, min_y=-16, max_y=16, pixels_per_meter=4, hist_max_per_pixel=10):
    xbins = np.linspace(
        min_x, max_x + 1,
               (max_x - min_x) * pixels_per_meter + 1,
    )
    ybins = np.linspace(
        min_y, max_y + 1,
               (max_y - min_y) * pixels_per_meter + 1,
    )
    # Compute histogram of x and y coordinates of points.
    hist = np.histogramdd(lidar[..., :2], bins=(xbins, ybins))[0]
    # Clip histogram
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel
    # Normalize histogram by the maximum number of points in a bin we care about.
    overhead_splat = hist / hist_max_per_pixel * 255.
    # Return splat in X x Y orientation, with X parallel to car axis, Y perp, both parallel to ground.
    return overhead_splat[::-1, :]


# modify from manual control
def _parse_image_cb(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


# modify from leaderboard
def _parse_lidar_cb(lidar_data):
    points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
    points = copy.deepcopy(points)
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    return points


if __name__ == "__main__":
    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')