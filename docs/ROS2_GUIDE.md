# OBEDIENCE - Guía de ROS2

## Estructura del Sistema ROS2

```
ros2_ws/
└── src/
    └── obedience_robot/
        ├── obedience_robot/
        │   ├── mujoco_bridge_node.py    # Nodo de simulación
        │   └── walking_controller_node.py # Nodo de control
        └── launch/
            └── obedience_launch.py      # Launch file
```

## Instalación Rápida

```bash
# 1. Entrar al workspace
cd /root/Obedience-Biped-a-Thinking-Autonomous-System/ros2_ws

# 2. Compilar
source /opt/ros/humble/setup.bash
colcon build --symlink-install

# 3. Sourcing
source install/setup.bash
```

## Ejecución

### Opción 1: Script de conveniencia
```bash
cd /root/Obedience-Biped-a-Thinking-Autonomous-System
chmod +x launch_ros2.sh
./launch_ros2.sh
```

### Opción 2: Launch directo
```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 launch obedience_robot obedience_launch.py
```

### Opción 3: Nodos individuales
```bash
# Terminal 1: Bridge
ros2 run obedience_robot mujoco_bridge --ros-args -p scene_xml:=hospital_scene.xml

# Terminal 2: Walking Controller  
ros2 run obedience_robot walking_controller
```

## Tópicos ROS2

### Publicados (desde MuJoCo Bridge)

| Tópico | Tipo | Descripción |
|--------|------|-------------|
| `/joint_states` | `sensor_msgs/JointState` | Posiciones, velocidades, torques |
| `/imu/data` | `sensor_msgs/Imu` | Acelerómetro y giroscopio |
| `/odom` | `nav_msgs/Odometry` | Odometría del robot |
| `/walking/state` | `std_msgs/String` | Estado del controlador |

### Suscritos (comandos al robot)

| Tópico | Tipo | Descripción |
|--------|------|-------------|
| `/cmd_vel` | `geometry_msgs/Twist` | Comandos de velocidad |
| `/joint_commands` | `sensor_msgs/JointState` | Comandos articulares directos |

## Comandos Útiles

### Ver tópicos activos
```bash
ros2 topic list
```

### Ver datos de articulaciones
```bash
ros2 topic echo /joint_states --once
```

### Ver IMU
```bash
ros2 topic echo /imu/data --once
```

### Comandar movimiento
```bash
# Caminar hacia adelante (0.3 m/s)
ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 0.3}}" --once

# Girar (0.5 rad/s)
ros2 topic pub /cmd_vel geometry_msgs/Twist "{angular: {z: 0.5}}" --once

# Caminar y girar
ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 0.2}, angular: {z: 0.3}}" --rate 10

# Detener
ros2 topic pub /cmd_vel geometry_msgs/Twist "{}" --once
```

### Ver frecuencia de publicación
```bash
ros2 topic hz /joint_states
ros2 topic hz /imu/data
```

### Ver grafo de nodos
```bash
ros2 node list
ros2 node info /mujoco_bridge
```

## Parámetros

### MuJoCo Bridge
```bash
ros2 param list /mujoco_bridge
ros2 param get /mujoco_bridge publish_rate
ros2 param set /mujoco_bridge realtime_factor 0.5
```

### Walking Controller
```bash
ros2 param list /walking_controller
ros2 param get /walking_controller max_forward_vel
```

## Diagrama de Nodos

```
                    ┌─────────────────────┐
                    │   Terminal/GUI      │
                    │   (teleop)          │
                    └──────────┬──────────┘
                               │
                         /cmd_vel
                               │
                               ▼
                    ┌─────────────────────┐
                    │ walking_controller  │
                    │                     │
                    │ Capture Point algo  │
                    └──────────┬──────────┘
                               │
                      /joint_commands
                               │
                               ▼
┌─────────────────────────────────────────────────────┐
│                  mujoco_bridge                       │
│                                                      │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐ │
│  │ MuJoCo Sim │───▶│  Sensors   │───▶│ Publishers │ │
│  └────────────┘    └────────────┘    └────────────┘ │
│                                                      │
└──────────────────────────┬──────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        /joint_states   /imu/data    /odom
```

## Integración con Navigation Stack

Para navegación autónoma, agregar:

```bash
# Instalar nav2
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup

# Lanzar con nav2
ros2 launch nav2_bringup navigation_launch.py
```

## Troubleshooting

### Error: "Scene file not found"
```bash
# Verificar que hospital_scene.xml existe
ls -la ~/Obedience-Biped-a-Thinking-Autonomous-System/models/xml/
```

### Error: "No module named 'mujoco'"
```bash
# Instalar MuJoCo en el entorno ROS2
pip3 install mujoco
```

### Nodo no responde a cmd_vel
```bash
# Verificar conexión
ros2 topic info /cmd_vel
# Debe mostrar 1 publisher y 1+ subscribers
```
