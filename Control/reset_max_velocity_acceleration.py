#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Time
from iiwa_msgs.srv import SetPathParameters
from iiwa_msgs.msg import JointPosition

import numpy as np
import sys

if __name__ == "__main__":
    argv = sys.argv
    joint_relative_velocity = 0.03 # 0.01 # 0.03
    joint_relative_acceleration = 0.03 # 0.01 # 0.03
    if len(argv) >= 2:
        joint_relative_velocity = float(argv[1])
    if len(argv) >= 3:
        joint_relative_acceleration = float(argv[2])

    # define ros node
    rospy.init_node('reset_max_velocity_acceleration', anonymous=True)

    # get current eef pose
    msg = rospy.wait_for_message("/iiwa/state/CartesianPose", PoseStamped, timeout=10)
    current_eef_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, 
                                          msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
    current_eef_pose[3:] /= np.linalg.norm(current_eef_pose[3:])

    # target eef pose
    target_eef_pose = np.array([0.228822, 0.0459129, 0.749912, 0.128973, 0.979442, -0.0205517, 0.153741])
    target_eef_pose[3:] /= np.linalg.norm(target_eef_pose[3:])

    print(f"current_eef_pose: {current_eef_pose.round(decimals=3)}")
    print(f"target_eef_pose : {target_eef_pose.round(decimals=3)}")

    # CartesianPose publisher
    pub = rospy.Publisher("/iiwa/command/CartesianPose", PoseStamped)
    rospy.sleep(1)

    # move to target eef pose
    out_msg = PoseStamped()
    out_msg.pose.position.x = target_eef_pose[0]
    out_msg.pose.position.y = target_eef_pose[1]
    out_msg.pose.position.z = target_eef_pose[2]
    out_msg.pose.orientation.x = target_eef_pose[3]
    out_msg.pose.orientation.y = target_eef_pose[4]
    out_msg.pose.orientation.z = target_eef_pose[5]
    out_msg.pose.orientation.w = target_eef_pose[6]
    print(f"\nMoving to eef pose {target_eef_pose}...")
    pub.publish(out_msg)
    rospy.wait_for_message("/iiwa/state/DestinationReached", Time, timeout=10)
    print(f"Reached eef pose {target_eef_pose}!")

    # path parameters service
    rospy.wait_for_service("/iiwa/configuration/pathParameters", timeout=5)
    print(f"Setting joint_relative_velocity to {joint_relative_velocity} & joint_relative_acceleration to {joint_relative_acceleration}...")
    service = rospy.ServiceProxy("/iiwa/configuration/pathParameters", SetPathParameters)
    resp = service(joint_relative_velocity=joint_relative_velocity, joint_relative_acceleration=joint_relative_acceleration)
    print(f"Success={resp.success} (error={resp.error})!\n\n")

    # move back to home pose
    home_joint_position = np.array([0, 0, 0, -90*np.pi/180, 0, 90*np.pi/180, 15*np.pi/180])
    out_msg = JointPosition()
    out_msg.position.a1 = home_joint_position[0]
    out_msg.position.a2 = home_joint_position[1]
    out_msg.position.a3 = home_joint_position[2]
    out_msg.position.a4 = home_joint_position[3]
    out_msg.position.a5 = home_joint_position[4]
    out_msg.position.a6 = home_joint_position[5]
    out_msg.position.a7 = home_joint_position[6]

    # JointPosition publisher
    pub = rospy.Publisher("/iiwa/command/JointPosition", JointPosition)
    rospy.sleep(1)

    print(f"\nMoving back home joint pos {home_joint_position}...")
    pub.publish(out_msg)
    rospy.wait_for_message("/iiwa/state/DestinationReached", Time, timeout=10)
    print(f"Reached home joint pos {home_joint_position}!")
