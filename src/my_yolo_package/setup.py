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
            'inference_node = my_yolo_package.inference_node:main',
            'visualizer_node = my_yolo_package.visualizer_node:main',
            'oak_camera_node = my_yolo_package.oak_camera_node:main',
            'tracking_logger = my_yolo_package.tracking_logger:main'
        ],
    },
)
