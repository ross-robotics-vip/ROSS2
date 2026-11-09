from setuptools import find_packages, setup

package_name = 'ross2_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sanjay',
    maintainer_email='sanjay@todo.todo',
    description='Vision package for ROSS2',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'esp32_cam_node = ross2_vision.esp32_cam_node:main',
            'yolov6_node = ross2_vision.yolov6_node:main',
            'klt_slam_node = ross2_vision.klt_slam_node:main',
        ],
    },
)
