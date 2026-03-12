from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'obedience_robot'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Obedience Team',
    maintainer_email='obedience@robot.dev',
    description='Autonomous bipedal robot for hospital medicine delivery',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'robot = obedience_robot.integrated_robot_node:main',
            'thinking_node = obedience_robot.thinking_node:main',
            'health_node = obedience_robot.health_node:main',
            'fault_injection_gui = obedience_robot.health.fault_injection:main_gui',
            'mission_control = obedience_robot.mission_control_gui:main',
        ],
    },
)
