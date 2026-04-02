# Copyright (c) 2024 ICHIRO ITS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import socket
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    hostname = socket.gethostname()
    ninshiki_config_path = os.path.expanduser(f'~/ichiro-ws/configuration/{hostname}/detection/')
    shisen_config_path = os.path.expanduser(f'~/ichiro-ws/configuration/{hostname}/camera/')

    gpu_arg = DeclareLaunchArgument('gpu', default_value='0', description='Enable GPU (0 or 1)')
    myriad_arg = DeclareLaunchArgument('myriad', default_value='0', description='Enable Myriad/NCS2 (0 or 1)')
    frequency_arg = DeclareLaunchArgument('frequency', default_value='96', description='Publisher frequency in Hz')

    detector_args = [ninshiki_config_path, "dnn", "color"]

    return LaunchDescription([
        gpu_arg,
        myriad_arg,
        frequency_arg,
        Node(
            package='shisen_cpp',
            executable='camera',
            name='camera',
            output='screen',
            arguments=[shisen_config_path],
            respawn=True,
            respawn_delay=1
        ),
        Node(
            package='ninshiki_cpp',
            executable='detector',
            name='detector',
            output='screen',
            arguments=[
                ninshiki_config_path,
                "dnn",
                "color",
                "--GPU", LaunchConfiguration('gpu'),
                "--MYRIAD", LaunchConfiguration('myriad'),
                "--frequency", LaunchConfiguration('frequency'),
            ],
            respawn=True,
            respawn_delay=1
        ),
    ])
