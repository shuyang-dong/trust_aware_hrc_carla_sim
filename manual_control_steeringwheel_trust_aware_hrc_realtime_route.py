#!/usr/bin/env python

# Copyright (c) 2019 Intel Labs
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control with steering wheel Logitech G29.

To drive start by preshing the brake pedal.
Change your wheel_config.ini according to your steering wheel.

To find out the values of your steering wheel use jstest-gtk in Ubuntu.

"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

import pandas as pd

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc
from carla.agents.navigation.global_route_planner import GlobalRoutePlanner
from carla.agents.navigation.controller import VehiclePIDController
from carla.agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from carla.agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error


import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_z
    from pygame.locals import K_x
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, actor_filter, route_id, ego_spawn_point_index, dynamic_instance_df,static_instance_df):
        self.world = carla_world
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self.other_veh_list = []
        self.dynamic_instance_list = []
        self.route_id = route_id
        self.ego_spawn_point_index = ego_spawn_point_index
        self.dynamic_instance_df = dynamic_instance_df
        self.static_instance_df = static_instance_df
        self.spawn_points_list = self.world.get_map().get_spawn_points()
        print(self.dynamic_instance_df)
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.trust_level = 4
        self.trust_level_list = []

    def spawn_walker(self, spawn_point_walker, pedestrian_speed=1.0, pedestrian_heading=70):
        #spawn_point_walker = carla.Transform(location=location, rotation=rotation)
        ped_blueprints = self.world.get_blueprint_library().filter("walker.*")
        player = self.world.spawn_actor(random.choice(ped_blueprints), spawn_point_walker)
        player_control = carla.WalkerControl()
        player_control.speed = pedestrian_speed
        player_rotation = carla.Rotation(0, pedestrian_heading, 0)
        player_control.direction = player_rotation.get_forward_vector()
        player.apply_control(player_control)
        return player

    def spawn_vehicle(self, veh_blueprint_str, spawn_point_vehicle, if_static=False):
        veh_blueprint = self.world.get_blueprint_library().filter(veh_blueprint_str)[0]
        veh = self.world.spawn_actor(veh_blueprint, spawn_point_vehicle)
        #self.other_veh_list.append(veh)
        #veh_control = carla.VehicleControl()
        #veh.apply_control(veh_control)
        if not if_static:
            veh.set_autopilot(True)
        return veh

    def other_veh_control(self, veh, dest_waypoint):
        custom_controller = VehiclePIDController(veh, args_lateral={'K_P': 1, 'K_D': 0.0, 'K_I': 0},
                                                 args_longitudinal={'K_P': 1, 'K_D': 0.0, 'K_I': 0.0})
        target_waypoint = self.world.get_map().get_waypoint(dest_waypoint.location)
        control_signal = custom_controller.run_step(1, target_waypoint)
        veh.apply_control(control_signal)

    def spawn_static_instance_in_sim(self, blueprint, spawnpoint_index):
        veh = self.world.try_spawn_actor(blueprint, self.spawn_points_list[spawnpoint_index])
        if veh != None:
            self.other_veh_list.append(veh)
            print('spawn veh: ', veh)

    def spawn_static_instance_before_sim(self):
        #static_instance_df = pd.read_csv(static_instance_file)
        for item, row in self.static_instance_df.iterrows():
            spawnpoint_index = int(row['spawnpoint_index'])
            print(self.world.get_blueprint_library().filter(row['blueprint']))
            blueprint = self.world.get_blueprint_library().filter(row['blueprint'])[0]
            veh = self.world.try_spawn_actor(blueprint, self.spawn_points_list[spawnpoint_index])
            if veh != None:
                self.other_veh_list.append(veh)
                print('spawn veh: ', row['blueprint'])

    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        # vehicle_blueprint_lib = self.world.get_blueprint_library().filter(self._actor_filter)
        # blueprint = random.choice(vehicle_blueprint_lib)
        # assign vehicle blueprint
        blueprint = self.world.get_blueprint_library().filter('vehicle.chevrolet.impala')[0]
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_points = self.world.get_map().get_spawn_points()
            #spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            spawn_point = spawn_points[int(self.ego_spawn_point_index)]
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            # spawn other veh in the simulation
            # self.spawn_vehicle(spawn_points[226])

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        # new list to save trust level
        self.trust_level = 4
        self.trust_level_list = []

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)
        self.trust_level_list.append(self.trust_level)
        # apply control for other vehicles in the simulation
        # spawn_points = self.world.get_map().get_spawn_points()
        # for veh in self.other_veh_list:
        #     dest = spawn_points[228]
        #     self.other_veh_control(veh, dest)

        # generate dynamic instance triggered by ego vehicle's location
        ego_waypoint = self.world.get_map().get_waypoint(self.player.get_location())
        ego_road_id = ego_waypoint.road_id
        ego_lane_id = ego_waypoint.lane_id
        # dynamic_instance_this_time = self.dynamic_instance_df[(self.dynamic_instance_df['ego_veh_road_id']==ego_road_id) &
        #                                                       (self.dynamic_instance_df['ego_veh_lane_id']==ego_lane_id)]
        dynamic_instance_this_time = self.dynamic_instance_df[(self.dynamic_instance_df['ego_veh_road_id'] == ego_road_id)] #when the ego veh enter the road with this instance
        if (len(dynamic_instance_this_time) != 0):
            dynamic_bp = dynamic_instance_this_time['blueprint'].to_list()[0]
            if dynamic_bp.split('.')[0] == 'walker':
                if (dynamic_instance_this_time['generate_flag'].to_list()[0] == 0):  # only works for 1 instance on 1 road segment
                    dynamic_spawnpoint_index = int(dynamic_instance_this_time['spawnpoint_index'])
                    dynamic_spawnpoint = self.spawn_points_list[dynamic_spawnpoint_index]
                    dynamic_instance_this_time_index = dynamic_instance_this_time.index.values[0]
                    pedestrian_speed = dynamic_instance_this_time['speed'].to_list()[0]
                    pedestrian_heading = dynamic_instance_this_time['heading'].to_list()[0]
                    dynamic_spawnpoint.location.y += dynamic_instance_this_time['delta_y'].to_list()[0]
                    dynamic_spawnpoint.location.x += dynamic_instance_this_time['delta_x'].to_list()[0]
                    dynamic_instance = self.spawn_walker(dynamic_spawnpoint, pedestrian_speed=pedestrian_speed,
                                                         pedestrian_heading=pedestrian_heading)
                    print('generate dynamic instance: ', dynamic_instance)
                    self.dynamic_instance_list.append({ego_road_id: dynamic_instance.id})
                    self.dynamic_instance_df.loc[dynamic_instance_this_time_index, 'generate_flag'] = 1
            else:
                dynamic_instance_this_time = self.dynamic_instance_df[
                    (self.dynamic_instance_df['ego_veh_road_id'] == ego_road_id) &
                    (self.dynamic_instance_df['ego_veh_lane_id']==-ego_lane_id)]
                dynamic_spawnpoint_index = int(dynamic_instance_this_time['spawnpoint_index'])
                dynamic_spawnpoint = self.spawn_points_list[dynamic_spawnpoint_index]
                dynamic_instance_this_time_index = dynamic_instance_this_time.index.values[0]
                if (dynamic_instance_this_time['generate_flag'].to_list()[0] == 0):  # only works for 1 instance on 1 road segment
                    dynamic_instance = self.spawn_vehicle(dynamic_bp, dynamic_spawnpoint, if_static=False)
                    print('generate dynamic instance: ', dynamic_instance)
                    self.dynamic_instance_list.append({ego_road_id: dynamic_instance.id})
                    self.dynamic_instance_df.loc[dynamic_instance_this_time_index, 'generate_flag'] = 1

            # #print('dynamic_instance_this_time_generate_flag: ', dynamic_instance_this_time['generate_flag'].to_list())
            # if (dynamic_instance_this_time['generate_flag'].to_list()[0]==0): # only works for 1 instance on 1 road segment
            #     #print(dynamic_instance_this_time, dynamic_instance_this_time['generate_flag'].to_list()[0])
            #     dynamic_instance_this_time_index = dynamic_instance_this_time.index.values[0]
            #     #print(dynamic_instance_this_time_index)
            #     dynamic_bp = dynamic_instance_this_time['blueprint'].to_list()[0]
            #     dynamic_spawnpoint_index = int(dynamic_instance_this_time['spawnpoint_index'])
            #     dynamic_spawnpoint = self.spawn_points_list[dynamic_spawnpoint_index]
            #     #dynamic_instance = self.world.spawn_actor(dynamic_bp, dynamic_spawnpoint)
            #     #print('dynamic_bp: ', dynamic_bp.split('.'), dynamic_bp)
            #     if dynamic_bp.split('.')[0] == 'walker':
            #         pedestrian_speed = dynamic_instance_this_time['speed'].to_list()[0]
            #         pedestrian_heading = dynamic_instance_this_time['heading'].to_list()[0]
            #         dynamic_spawnpoint.location.y += dynamic_instance_this_time['delta_y'].to_list()[0]
            #         dynamic_spawnpoint.location.x += dynamic_instance_this_time['delta_x'].to_list()[0]
            #         dynamic_instance = self.spawn_walker(dynamic_spawnpoint,  pedestrian_speed=pedestrian_speed, pedestrian_heading=pedestrian_heading)
            #     else:
            #         dynamic_instance = self.spawn_vehicle(dynamic_bp, dynamic_spawnpoint, if_static=False)
            #     print('generate dynamic instance: ', dynamic_instance)
            #     self.dynamic_instance_list.append({ego_road_id:dynamic_instance.id})
            #     self.dynamic_instance_df.loc[dynamic_instance_this_time_index, 'generate_flag'] = 1
            #     dynamic_instance_this_time = self.dynamic_instance_df[
            #         (self.dynamic_instance_df['ego_veh_road_id'] == ego_road_id) &
            #         (self.dynamic_instance_df['ego_veh_lane_id'] == ego_lane_id)]
            # else:
            #     pass
            #
        print(self.dynamic_instance_list)
        # destory generated dynamic instance when ego vehicle is not on corresponding lane
        for item in self.dynamic_instance_list:
            print(item)
            print(list(item.values()))
            if not (ego_road_id in item):
                actor_list = self.world.get_actors()
                actor = actor_list.find(list(item.values())[0])
                print('destory dynamic instance: ', actor)
                if isinstance(actor, carla.Vehicle) or isinstance(actor, carla.Walker):
                    actor.destroy()

        # # generate static instance before ego veh reach its road
        # static_instance_this_time = self.static_instance_df[
        #     (self.static_instance_df['ego_veh_road_id'] == ego_road_id) & (self.static_instance_df['ego_veh_lane_id']==ego_lane_id)]
        # if (len(static_instance_this_time) != 0):
        #     static_bp = static_instance_this_time['blueprint'].to_list()[0]
        #     spawnpoint_index = int(static_instance_this_time['spawnpoint_index'])
        #     self.spawn_static_instance_in_sim(static_bp, spawnpoint_index)





    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()

        #actor_list = self.world.get_actors()
        if len(self.other_veh_list) != 0:
            for actor in self.other_veh_list:
                if isinstance(actor, carla.Vehicle) or isinstance(actor, carla.Walker):
                    print('destory: ', actor)
                    actor.destroy()
        trust_level_file_path = '/home/cpsgroup/trust_aware_hrc/trust_level_record/trust_level_record_{route_id}.csv'.format(route_id=self.route_id)
        trust_level_df = pd.DataFrame(columns=['trust_level'])
        trust_level_df['trust_level'] = self.trust_level_list
        trust_level_df.to_csv(trust_level_file_path)

# ==============================================================================
# -- DualControl -----------------------------------------------------------
# ==============================================================================


class DualControl(object):
    def __init__(self, world, route_point_index_list, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self.agent = BasicAgent(world.player.vehicle)
            #world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

        # initialize steering wheel
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            raise ValueError("Please Connect Just One Joystick")

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        self._parser = ConfigParser()
        self._parser.read('/home/cpsgroup/trust_aware_hrc/wheel_config.ini')
        print(self._parser.sections())
        self._steer_idx = int(
            self._parser.get('G29 Racing Wheel', 'steering_wheel'))
        self._throttle_idx = int(
            self._parser.get('G29 Racing Wheel', 'throttle'))
        self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
        self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
        self._handbrake_idx = int(
            self._parser.get('G29 Racing Wheel', 'handbrake'))

    def parse_events(self, world, clock, sync_mode):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    world.restart()
                elif event.button == 1:
                    world.hud.toggle_info()
                elif event.button == 2:
                    world.camera_manager.toggle_camera()
                elif event.button == 3:
                    world.next_weather()
                elif event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.button == 23:
                    world.camera_manager.next_sensor()
                elif event.button == 10: # switch between autopilot and manual using button R3
                    self._autopilot_enabled = not self._autopilot_enabled
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification('Agent %s' % ('On' if self._autopilot_enabled else 'Off'))
                elif event.button == 7:  # for trust level+1
                    if world.trust_level + 1 <= 7:
                        world.trust_level += 1
                    else:
                        world.trust_level = 7
                elif event.button == 11:  # for trust level-1
                    if world.trust_level - 1 >= 0:
                        world.trust_level -= 1
                    else:
                        world.trust_level = 0

            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r:
                    world.camera_manager.toggle_recording()
                elif event.key == K_z: # for trust level+1
                    if world.trust_level+1<=7:
                        world.trust_level += 1
                    else:
                        world.trust_level =7
                elif event.key == K_x:# for trust level-1
                    if world.trust_level-1>=0:
                        world.trust_level -= 1
                    else:
                        world.trust_level = 0

                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p:
                        if not self._autopilot_enabled and not sync_mode:
                            print("WARNING: You are currently in asynchronous mode and could "
                                  "experience some issues with the traffic simulation")
                        self._autopilot_enabled = not self._autopilot_enabled
                        #world.player.set_autopilot(self._autopilot_enabled)
                        #world.hud.notification('Agent control %s' % ('On' if self._autopilot_enabled else 'Off'))
                        if self._autopilot_enabled:
                            world.hud.notification('Agent control %s' % ('On' if self._autopilot_enabled else 'Off'))
                            # apply control by agent instead
                            if self.agent.done():
                                # check on the next destination
                                print("The target has been reached, stopping the simulation")
                                break
                            control = self.agent.run_step()
                            control.manual_gear_shift = False
                            world.player.apply_control(control)
                        else:
                            world.hud.notification('Manual control %s' % ('Off' if self._autopilot_enabled else 'On'))
                            if isinstance(self._control, carla.VehicleControl):
                                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                                self._parse_vehicle_wheel()
                                self._control.reverse = self._control.gear < 0
                            elif isinstance(self._control, carla.WalkerControl):
                                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
                            world.player.apply_control(self._control)

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._parse_vehicle_wheel()
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_vehicle_wheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        # print (jsInputs)
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 = 1.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd

        #toggle = jsButtons[self._reverse_idx]

        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.world.get_map().name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            #'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            #u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            #'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            #'Height:  % 18.0f m' % t.location.z,
            'Trust level: %s'%world.trust_level,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))




    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7))]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '50')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None \
            else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data) # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        route_folder = '/home/cpsgroup/trust_aware_hrc/route'
        route_file = '{route_folder}/route_{id}.csv'.format(route_folder=route_folder, id=args.route_id)
        route_df = pd.read_csv(route_file)
        print(route_df)
        route_point_index_list = route_df['route_point_index'].to_list()
        ego_spawn_point_index = int(route_df['spawn_point_index'].to_list()[0])
        #ego_spawn_point_index = 175
        ego_dest_point_index = int(route_df['destination_point_index'].to_list()[0])
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        sim_world = client.load_world('Town01')
        map = sim_world.get_map()
        dynamic_instance_file = '/home/cpsgroup/trust_aware_hrc/instance_info/dynamic_instance_info.csv'
        dynamic_instance_df = pd.read_csv(dynamic_instance_file)
        static_instance_file = '/home/cpsgroup/trust_aware_hrc/instance_info/static_instance_info.csv'
        static_instance_df = pd.read_csv(static_instance_file)
        world = World(sim_world, hud, args.filter, args.route_id, ego_spawn_point_index, dynamic_instance_df, static_instance_df)
        #
        #static_instance_file = '/home/cpsgroup/trust_aware_hrc/instance_info/static_instance_info.csv'
        #static_instance_df = pd.read_csv(static_instance_file)
        world.spawn_static_instance_before_sim()
        # #world = World(client.get_world(), hud, args.filter)
        # #agent = BehaviorAgent(world.player, behavior='cautious')
        # original_settings = sim_world.get_settings()
        # settings = sim_world.get_settings()
        # if not settings.synchronous_mode:
        #     settings.synchronous_mode = True
        #     settings.fixed_delta_seconds = 0.05
        # sim_world.apply_settings(settings)
        #
        # traffic_manager = client.get_trafficmanager()
        # traffic_manager.set_synchronous_mode(True)
        # traffic_manager.global_percentage_speed_difference(50)

        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.global_percentage_speed_difference(50)
        ##

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                  "experience some issues with the traffic simulation")

        controller = DualControl(world, traffic_manager, route_point_index_list, args.autopilot)

        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()
        while True:
            if args.sync:
                sim_world.tick()
            clock.tick_busy_loop(60)
            if controller.parse_events(world, clock, args.sync):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            # if ego vehicle near destination, destory world and save file
            # ego_dest_waypoint = map.get_waypoint(world.spawn_points_list[ego_dest_point_index].location)
            # ego_waypoint = map.get_waypoint(world.player.get_location())
            # if (ego_waypoint.lane_id==ego_dest_waypoint.lane_id) and (ego_waypoint.road_id==ego_dest_waypoint.road_id):
            #     distance =  math.sqrt((ego_waypoint.location.x - ego_dest_waypoint.location.x)**2 + (ego_waypoint.location.y - ego_dest_waypoint.location.y)**2)
            #     if distance<=5.0:
            #         world.destroy()

        # clock = pygame.time.Clock()
        # while True:
        #     clock.tick_busy_loop(60)
        #     if controller.parse_events(world, clock):
        #         return
        #     world.tick(clock)
        #     world.render(display)
        #     pygame.display.flip()



    finally:

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    argparser.add_argument(
        '--route_id',
        default='0',
        help='Id of preset route file')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
