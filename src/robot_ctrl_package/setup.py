from setuptools import setup
from glob import glob
import os

package_name = 'robot_ctrl_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob(os.path.join('launch', '*launch.py')))

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='put',
    maintainer_email='put@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'control_node = robot_ctrl_package.control_node:main',
            'transform_node = robot_ctrl_package.transform_node:main',
            'arm_node = robot_ctrl_package.arm_node:main'
        ],
    },
)
