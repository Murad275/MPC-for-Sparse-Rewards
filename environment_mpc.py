#!/usr/bin/env python


import math
from math import pi
import numpy as np
import rospy
from geometry_msgs.msg import Point, Pose, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
# from tf.transformations import euler_from_quaternion, quaternion_from_euler
import copy
from mpc_obs import mpc

from respawnGoal import Respawn

import copy

target_not_movable = False

class Env():
    def __init__(self, action_dim=2):
        self.yaw = 0
        self.ref_state = [0, 0, 0]
        self.curr_state = [0, 0, 0]
        self.scan = []
        self.obst_dist = [0,0]
        self.goal_x = 0
        self.goal_y = 0
        self.goal_angle = 0
        self.heading = 0
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.action_agent = [0,0]
        self.action_mpc = [0,0]
        # self.vis_pub = rospy.Publisher('visualization_marker', Marker)
        
        self.respawn_goal = Respawn()
        self.past_distance = 0.
        self.stopped = 0
        self.action_dim = action_dim
        self.mpc = mpc()
        
        self.get_old = False
                
        #Keys CTRL + c will stop script
        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        #you can stop turtlebot by publishing an empty Twist
        #message
        rospy.loginfo("Stopping TurtleBot")
        self.pub_cmd_vel.publish(Twist())
        rospy.sleep(1)

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        self.past_distance = goal_distance

        return goal_distance


    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians


    def getOdometry(self, odom):
        self.past_position = copy.deepcopy(self.position)
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        # orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = self.euler_from_quaternion(orientation.x, orientation.y, orientation.z, orientation.w)

        self.yaw = yaw
        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)
        self.goal_angle = goal_angle
        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)
#######################################################
    def state_global(self):

        return self.position.x, self.position.y, self.yaw
#######################################################        
    def getState(self, scan):
        scan_range = []
        # scan_mins = []
        heading = self.heading
        min_range = 0.13
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf') or scan.ranges[i] == float('inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]) or scan.ranges[i] == float('nan'):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        min_scan = round(min(scan_range), 2)  ## distance to nearest obstacle
        
        
        self.curr_state_np = np.array([round(self.position.x,2), round(self.position.y,2), round(self.yaw, 2)])

        self.ref_state[0] = self.goal_x
        self.ref_state[1] = self.goal_y
        self.ref_state[2] = self.goal_angle

        self.ref_state_np = np.array([round(self.goal_x, 3), round(self.goal_y, 3), round(self.goal_angle, 3), 0, 0])
        
        id = scan_range.index(min(scan_range))
        
        rad = self.yaw + self.scan.angle_increment * id
        self.obst_dist[0] = self.position.x + min_scan* math.cos(rad)
        self.obst_dist[1] = self.position.y + min_scan* math.sin(rad)   

        if min_range > min(scan_range) > 0:
            done = True

        noise = np.random.normal(0, 0.05)

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2) + noise
        if current_distance <= 0.12:
            self.get_goalbox = True

        return scan_range + [heading, current_distance, min_scan, rad ], done

    #***************************************************************
    def setReward(self, state, done):
        current_distance = state[-3]
        # heading = state[-4]

       
        reward =  0
        self.past_distance = current_distance
        a, b, c, d = float('{0:.3f}'.format(self.position.x)), float('{0:.3f}'.format(self.past_position.x)), float('{0:.3f}'.format(self.position.y)), float('{0:.3f}'.format(self.past_position.y))
        if a == b and c == d:
            self.stopped += 1
            if self.stopped == 20:
                self.stopped = 0
                done = True
        else:
            self.stopped = 0

        if done:
            reward = -100.
            self.pub_cmd_vel.publish(Twist())

        if self.get_goalbox:
            reward = 100
            self.pub_cmd_vel.publish(Twist())
            if world:
                self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True, running=True)
                if target_not_movable:
                    self.reset()
            else:
                self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward, done
#****************************** Call MPC action here 
    def step(self, action, get_mpc=False):
        linear_vel = action[0]
        ang_vel = action[1]
        vel_cmd = Twist()
        cont_mpc = self.mpc.run_mpc(self.ref_state_np, self.curr_state_np,
                                    self.obst_dist[0], self.obst_dist[1])
        
        self.action_agent = [linear_vel, ang_vel]
        self.action_mpc = [cont_mpc[0], cont_mpc[1]]

        if get_mpc:
        
            vel_cmd.linear.x= cont_mpc[0]
            vel_cmd.angular.z= cont_mpc[1]
           
        else:
            vel_cmd.linear.x = linear_vel
            vel_cmd.angular.z = ang_vel

  #******************************
        ret_act=[round(vel_cmd.linear.x,2), round(vel_cmd.angular.z,2)]
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data)
        reward, done = self.setReward(state, done)

        return np.asarray(state), reward, done, ret_act

    def reset(self):
        #print('aqui2_____________---')
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
                self.scan = data
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False
        else:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)

        self.goal_distance = self.getGoalDistace()
        state, _ = self.getState(data)

        return np.asarray(state)
