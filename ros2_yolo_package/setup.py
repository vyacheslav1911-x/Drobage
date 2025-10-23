from setuptools import setup

package_name = 'my_yolo_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'camera_node_r = my_yolo_package.camera_node_r:main',
            'yolo_node_r = my_yolo_package.yolo_node_r:main',
            'visualizer_node = my_yolo_package.visualizer_node:main',
        ],
    },
)
