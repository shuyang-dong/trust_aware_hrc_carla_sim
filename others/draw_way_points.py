import carla
import sys
import glob
import os

import pandas as pd
import matplotlib.pyplot as plt

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
#sys.path.append('../')
from carla.agents.navigation.global_route_planner import GlobalRoutePlanner
# from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

# # draw route recorded in trust level file
# file = '/home/cpsgroup/trust_aware_hrc/trust_level_record/trust_level_record_4.csv'
# route_df = pd.read_csv(file)
# x_list = []
# y_list = []
# for item, row in route_df.iterrows():
#     location = row['location']
#     location = location.strip('(')
#     location = location.strip(')')
#     location = location.split(',')
#     x_list.append(float(location[0]))
#     y_list.append(float(location[1]))
# #plt.gca().invert_xaxis()
# plt.gca().invert_yaxis()
# plt.plot(x_list, y_list, 'o')
# plt.show()

#
client = carla.Client("localhost", 2000)
client.set_timeout(10)
world = client.load_world('Town01')
spectator = world.get_spectator()
amap = world.get_map()
sampling_resolution = 2

# dao = GlobalRoutePlannerDAO(amap, sampling_resolution)
grp = GlobalRoutePlanner(amap, sampling_resolution)
# grp.setup()
spawn_points = world.get_map().get_spawn_points()
for i in range(len(spawn_points)):
    print(i, spawn_points[i])
#

print(spawn_points)
###
a = carla.Location(spawn_points[75].location)
b = carla.Location(spawn_points[16].location)
c = amap.get_waypoint(spawn_points[241].location)
file = amap.to_opendrive()
print(type(file))
client.generate_opendrive_world(file)
print(c.id, c.lane_id, c.road_id)
w1 = grp.trace_route(a, b) # there are other funcations can be used to generate a route in GlobalRoutePlanner.
#world.debug.draw_arrow(a, b, thickness=0.1, arrow_size=0.1, color=carla.Color(255, 0, 0), life_time=0)


i = 0
for w in w1:
    wp = w[0]
    print(w, wp)
    wp_junc = wp.get_junction()
    print('wp_junc: ', wp_junc, type(wp_junc))
    # print(wp.id, wp.lane_id, wp.road_id, wp.lane_type, type(wp.lane_type))
    # if wp_junc != None:
    #     print(wp_junc, wp_junc.id, wp_junc.bounding_box)
    #     waypoints_junc = wp_junc.get_waypoints(carla.LaneType.Any)
    #     print(waypoints_junc)
    #     print('\n')
    # if i % 10 == 0:
    #     world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
    #     color=carla.Color(r=255, g=0, b=0), life_time=120.0,
    #     persistent_lines=True)
    # else:
    #     world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
    #     color = carla.Color(r=0, g=0, b=255), life_time=1000.0,
    #     persistent_lines=True)
    # i += 1