# OBEDIENCE - Arquitectura de Sistema Autónomo

## 1. Datos Sensoriales Disponibles en MuJoCo

MuJoCo proporciona acceso completo al estado físico del robot. Estos son los datos que podemos extraer:

### 1.1 Datos Propioceptivos (Estado Interno del Robot)

| Sensor | Variable MuJoCo | Descripción | Unidades |
|--------|-----------------|-------------|----------|
| **Posición articular** | `data.qpos[joint_addr]` | Ángulo de cada articulación | rad |
| **Velocidad articular** | `data.qvel[joint_addr]` | Velocidad angular | rad/s |
| **Aceleración articular** | `data.qacc[joint_addr]` | Aceleración angular | rad/s² |
| **Torque aplicado** | `data.qfrc_actuator[joint_addr]` | Par motor aplicado | N·m |
| **Torque externo** | `data.qfrc_passive[joint_addr]` | Par por fuerzas externas | N·m |
| **Torque de contacto** | `data.qfrc_constraint[joint_addr]` | Par por restricciones | N·m |

### 1.2 Datos de Cuerpos Rígidos

| Sensor | Variable MuJoCo | Descripción | Unidades |
|--------|-----------------|-------------|----------|
| **Posición (mundo)** | `data.xpos[body_id]` | Posición 3D en marco mundo | m |
| **Orientación (matriz)** | `data.xmat[body_id]` | Matriz de rotación 3x3 | - |
| **Orientación (quat)** | `data.xquat[body_id]` | Cuaternión [w,x,y,z] | - |
| **Velocidad lineal** | `data.cvel[body_id][3:6]` | Velocidad del CoM | m/s |
| **Velocidad angular** | `data.cvel[body_id][0:3]` | Velocidad rotacional | rad/s |
| **Aceleración** | `data.cacc[body_id]` | Aceleración 6D | m/s², rad/s² |

### 1.3 Datos de Contacto (Exteroceptivos)

| Sensor | Variable MuJoCo | Descripción | Unidades |
|--------|-----------------|-------------|----------|
| **Número de contactos** | `data.ncon` | Contactos activos | - |
| **Posición contacto** | `data.contact[i].pos` | Punto de contacto | m |
| **Normal contacto** | `data.contact[i].frame` | Dirección normal | - |
| **Fuerza contacto** | `data.contact[i].efc_force` | Magnitud de fuerza | N |
| **Geoms en contacto** | `data.contact[i].geom1/2` | IDs de geometrías | - |

### 1.4 Sensores Adicionales (Configurables en XML)

```xml
<sensor>
    <!-- IMU en torso -->
    <accelerometer name="imu_acc" site="torso_imu"/>
    <gyro name="imu_gyro" site="torso_imu"/>
    
    <!-- Sensores de fuerza en pies -->
    <force name="left_foot_force" site="left_foot_site"/>
    <force name="right_foot_force" site="right_foot_site"/>
    <torque name="left_foot_torque" site="left_foot_site"/>
    <torque name="right_foot_torque" site="right_foot_site"/>
    
    <!-- Posición/velocidad articular -->
    <jointpos name="right_knee_pos" joint="right_knee"/>
    <jointvel name="right_knee_vel" joint="right_knee"/>
    <actuatorfrc name="right_knee_torque" actuator="right_knee_act"/>
</sensor>
```

---

## 2. Integración MuJoCo + ROS2

### 2.1 Arquitectura de Comunicación

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ROS2 ECOSYSTEM                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   Mission    │───▶│  Navigation  │───▶│   Walking            │  │
│  │   Planner    │    │   Node       │    │   Controller         │  │
│  │              │    │              │    │                      │  │
│  │  /mission/*  │    │  /nav/*      │    │  /walking/cmd_vel    │  │
│  └──────────────┘    └──────────────┘    └──────────┬───────────┘  │
│         │                   │                       │               │
│         │                   │                       ▼               │
│         │                   │            ┌──────────────────────┐  │
│         │                   │            │    MuJoCo Bridge     │  │
│         │                   │            │    Node              │  │
│         │                   └───────────▶│                      │  │
│         │                                │  /joint_states       │  │
│         └───────────────────────────────▶│  /robot_state        │  │
│                                          │  /imu                │  │
│                                          │  /contact_sensors    │  │
│                                          └──────────┬───────────┘  │
│                                                     │               │
└─────────────────────────────────────────────────────│───────────────┘
                                                      │
                                                      ▼
                                          ┌──────────────────────┐
                                          │     MuJoCo           │
                                          │     Simulation       │
                                          │                      │
                                          │  model, data         │
                                          └──────────────────────┘
```

### 2.2 Nodo Bridge: mujoco_ros2_bridge

El nodo puente es el corazón de la integración. Ejecuta la simulación y publica/suscribe tópicos ROS2.

```python
# Pseudocódigo del nodo bridge
class MuJoCoBridge(Node):
    def __init__(self):
        # Publishers (datos del robot → ROS2)
        self.pub_joint_states = Publisher('/joint_states', JointState)
        self.pub_imu = Publisher('/imu', Imu)
        self.pub_contact = Publisher('/contact_sensors', ContactArray)
        self.pub_odom = Publisher('/odom', Odometry)
        
        # Subscribers (comandos ROS2 → robot)
        self.sub_joint_cmd = Subscriber('/joint_commands', JointCommand)
        self.sub_velocity = Subscriber('/cmd_vel', Twist)
        
        # Timer para ciclo de simulación
        self.timer = create_timer(0.001, self.simulation_step)  # 1kHz
```

---

## 3. Definición de Objetos para Arquitectura de Constelación

### 3.1 Clase: `Robot` (Entidad Principal)

```python
@dataclass
class RobotState:
    """Estado completo del robot en un instante."""
    timestamp: float
    
    # Posición global
    position: np.ndarray      # [x, y, z] en marco mundo
    orientation: np.ndarray   # Cuaternión [w, x, y, z]
    velocity: np.ndarray      # [vx, vy, vz]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    
    # Estado articular
    joint_positions: Dict[str, float]   # nombre → rad
    joint_velocities: Dict[str, float]  # nombre → rad/s
    joint_torques: Dict[str, float]     # nombre → N·m
    
    # Centro de masa
    com_position: np.ndarray  # [x, y, z] relativo a torso
    com_velocity: np.ndarray  # [vx, vy, vz]

@dataclass  
class RobotConfig:
    """Configuración estática del robot."""
    name: str = "obedience"
    
    # Dimensiones
    mass: float = 12.6        # kg total
    height: float = 1.6       # m (altura de torso)
    
    # Articulaciones
    joints: List[str] = field(default_factory=lambda: [
        "right_hip_z_j", "right_hip_y_j", "right_hip", "right_knee", "right_foot_j",
        "left_hip_z_j", "left_hip_y_j", "left_hip", "left_knee", "left_foot_j"
    ])
    
    # Límites articulares (rad)
    joint_limits: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "right_hip_z_j": (-0.35, 0.35),
        "right_hip_y_j": (-0.35, 0.35),
        "right_hip": (-1.05, 1.05),
        "right_knee": (-2.09, -0.087),
        "right_foot_j": (-1.05, 1.05),
        # ... simétrico para izquierda
    })
    
    # Límites de torque (N·m)
    torque_limits: Dict[str, float] = field(default_factory=lambda: {
        "hip_z": 15.0,
        "hip_y": 30.0,
        "hip": 30.0,
        "knee": 30.0,
        "ankle": 10.0,
    })
```

### 3.2 Clase: `Leg` (Subsistema)

```python
@dataclass
class LegState:
    """Estado de una pierna."""
    side: str  # "left" | "right"
    
    # Articulaciones (5-DOF)
    hip_yaw: JointState
    hip_roll: JointState
    hip_pitch: JointState
    knee: JointState
    ankle: JointState
    
    # Posición del pie
    foot_position: np.ndarray     # [x, y, z] en marco torso
    foot_velocity: np.ndarray     # [vx, vy, vz]
    
    # Contacto
    is_stance: bool
    contact_force: np.ndarray     # [fx, fy, fz]
    contact_normal: np.ndarray    # Dirección normal
    
    # Jacobiano
    jacobian: np.ndarray          # 4x5 para posición + yaw

@dataclass
class JointState:
    """Estado de una articulación."""
    name: str
    position: float      # rad
    velocity: float      # rad/s
    acceleration: float  # rad/s²
    torque: float        # N·m (aplicado)
    torque_external: float  # N·m (fuerzas externas)
```

### 3.3 Clase: `IMU` (Sensor)

```python
@dataclass
class IMUData:
    """Datos de unidad de medición inercial."""
    timestamp: float
    
    # Acelerómetro
    linear_acceleration: np.ndarray  # [ax, ay, az] m/s²
    
    # Giroscopio
    angular_velocity: np.ndarray     # [wx, wy, wz] rad/s
    
    # Orientación estimada (si hay filtro)
    orientation: Optional[np.ndarray] = None  # Cuaternión
    
    # Covarianzas (para fusión sensorial)
    acc_covariance: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.01)
    gyro_covariance: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.001)
```

### 3.4 Clase: `ContactSensor` (Sensor de Fuerza)

```python
@dataclass
class FootContact:
    """Datos de contacto de un pie."""
    side: str
    timestamp: float
    
    # Estado de contacto
    is_contact: bool
    contact_count: int  # Número de puntos de contacto
    
    # Fuerzas
    force: np.ndarray        # [fx, fy, fz] en marco mundo
    torque: np.ndarray       # [tx, ty, tz] en marco pie
    
    # Centro de presión (CoP)
    cop_position: np.ndarray  # [x, y] relativo al pie
    
    # Para detección de resbalón
    normal_force: float      # Fz
    tangential_force: float  # sqrt(fx² + fy²)
    friction_ratio: float    # tangential / normal
```

### 3.5 Clase: `Battery` (Sistema de Energía)

```python
@dataclass
class BatteryState:
    """Estado del sistema de batería."""
    timestamp: float
    
    # Estado de carga
    soc: float              # 0.0 - 1.0 (State of Charge)
    voltage: float          # V
    current: float          # A (positivo = descarga)
    temperature: float      # °C
    
    # Estimaciones
    remaining_capacity: float  # Wh
    estimated_range: float     # metros restantes
    time_to_empty: float       # segundos
    
    # Alertas
    is_low: bool
    is_critical: bool
    health_status: str         # "good" | "degraded" | "replace"
```

### 3.6 Clase: `WalkingController` (Comportamiento)

```python
@dataclass
class WalkingState:
    """Estado del controlador de caminata."""
    # Fase de marcha
    stance_leg: str          # "left" | "right"
    swing_leg: str
    phase: float             # 0.0 - 1.0 dentro del paso
    
    # Capture Point
    capture_point: np.ndarray    # [x, y] objetivo del pie
    current_dcm: np.ndarray      # Divergent Component of Motion
    
    # Comandos de velocidad
    cmd_forward: float       # m/s
    cmd_lateral: float       # m/s  
    cmd_turn: float          # rad/s
    
    # Parámetros de paso
    step_length: float       # m
    step_width: float        # m
    step_height: float       # m
    step_duration: float     # s

class WalkingControllerConfig:
    """Parámetros del controlador."""
    # Ganancias
    k_capture_point: float = -1.0
    k_height: float = 5.0
    k_orientation: float = -15.0
    
    # Límites
    max_velocity: float = 0.5      # m/s
    max_turn_rate: float = 1.0     # rad/s
    min_step_time: float = 0.2     # s
    desired_step_time: float = 0.4 # s
    
    # Altura objetivo
    target_com_height: float = 1.45  # m
    swing_height: float = 0.2        # m
```

---

## 4. Tópicos ROS2 para Arquitectura de Constelación

### 4.1 Tópicos de Estado (Publicados por MuJoCo Bridge)

| Tópico | Tipo | Frecuencia | Descripción |
|--------|------|------------|-------------|
| `/joint_states` | `sensor_msgs/JointState` | 500 Hz | Posiciones, velocidades, esfuerzos |
| `/odom` | `nav_msgs/Odometry` | 100 Hz | Odometría del robot |
| `/imu/data` | `sensor_msgs/Imu` | 1000 Hz | Datos IMU |
| `/contact/left_foot` | `geometry_msgs/WrenchStamped` | 500 Hz | Fuerza pie izquierdo |
| `/contact/right_foot` | `geometry_msgs/WrenchStamped` | 500 Hz | Fuerza pie derecho |
| `/robot_state` | `obedience_msgs/RobotState` | 100 Hz | Estado completo custom |
| `/battery/state` | `sensor_msgs/BatteryState` | 1 Hz | Estado batería |

### 4.2 Tópicos de Comando (Suscritos por MuJoCo Bridge)

| Tópico | Tipo | Descripción |
|--------|------|-------------|
| `/cmd_vel` | `geometry_msgs/Twist` | Comando de velocidad |
| `/joint_commands` | `trajectory_msgs/JointTrajectory` | Trayectoria articular |
| `/walking/cmd` | `obedience_msgs/WalkingCommand` | Comando de caminata |

### 4.3 Servicios

| Servicio | Tipo | Descripción |
|----------|------|-------------|
| `/walking/start` | `std_srvs/Trigger` | Iniciar caminata |
| `/walking/stop` | `std_srvs/Trigger` | Detener caminata |
| `/battery/get_state` | `obedience_srvs/GetBatteryState` | Consultar batería |
| `/mission/start` | `obedience_srvs/StartMission` | Iniciar misión |

---

## 5. Nodos de la Arquitectura Constelación

```
                    ┌─────────────────────┐
                    │   Mission Planner   │
                    │   (Decisiones)      │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
    ┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
    │   Navigation    │ │   Battery   │ │   Safety        │
    │   Controller    │ │   Manager   │ │   Monitor       │
    └────────┬────────┘ └──────┬──────┘ └────────┬────────┘
             │                 │                 │
             └────────┬────────┴────────┬────────┘
                      │                 │
                      ▼                 ▼
            ┌─────────────────┐ ┌─────────────────┐
            │   Walking       │ │   State         │
            │   Controller    │ │   Estimator     │
            └────────┬────────┘ └────────┬────────┘
                     │                   │
                     └─────────┬─────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   MuJoCo Bridge     │
                    │   (Simulation)      │
                    └─────────────────────┘
```

### Nodos:

1. **mujoco_bridge**: Ejecuta simulación, publica sensores, recibe comandos
2. **state_estimator**: Fusión sensorial (IMU + contacto + odometría)
3. **walking_controller**: Algoritmo Capture Point
4. **navigation_controller**: Planificación de ruta local
5. **battery_manager**: Monitoreo y gestión de energía
6. **safety_monitor**: Detección de caídas, límites articulares
7. **mission_planner**: Orquestación de misiones de entrega

---

## 6. Implementación Paso a Paso

### Fase 1: Bridge Básico ✅
- [x] Extraer datos de MuJoCo (posición, velocidad, torques, contactos)
- [x] Crear clase MuJoCoBridge (`src/ros2/mujoco_bridge.py`)
- [x] Definir estructuras de datos (`src/core/`)
- [ ] Integrar con ROS2 publishers/subscribers

### Fase 2: Control de Locomoción ✅
- [x] Controlador Capture Point 5-DOF (`src/walking/capture_point_5dof.py`)
- [x] Cálculo de Jacobianos (`src/walking/jacobian.py`)
- [x] Suscripción a comandos de velocidad
- [ ] Publicación de estado de caminata en ROS2

### Fase 3: Estimación de Estado
- [ ] Filtro Kalman para orientación
- [ ] Fusión odométrica (IMU + kinematics)
- [x] Detección de contacto robusta (`src/core/sensors.py`)

### Fase 4: Navegación
- [x] Planificador de misión simple (`src/mission/mission_planner.py`)
- [ ] Planificador local (DWA o similar)
- [ ] Evasión de obstáculos
- [ ] Seguimiento de trayectoria

### Fase 5: Misiones ✅
- [x] Máquina de estados de misión
- [x] Gestión de batería integrada (`src/battery/battery_model.py`)
- [ ] Recovery behaviors
- [ ] Recarga automática

---

## 7. Implementación Actual

### Archivos Implementados

```
src/
├── core/                    # Estructuras de datos base
│   ├── __init__.py
│   ├── robot_state.py       # RobotState, LegState, TorsoState, etc.
│   ├── sensors.py           # IMUData, FootContact, ContactArray
│   └── commands.py          # VelocityCommand, WalkingCommand, etc.
│
├── walking/                 # Control de locomoción
│   ├── capture_point_5dof.py  # Controlador principal
│   ├── jacobian.py          # Cinemática diferencial
│   └── utils.py             # Funciones auxiliares
│
├── battery/                 # Sistema de energía
│   └── battery_model.py     # Modelo de batería no-lineal
│
├── mission/                 # Planificación de misión
│   └── mission_planner.py   # Secuenciador de waypoints
│
└── ros2/                    # Integración ROS2
    └── mujoco_bridge.py     # Bridge MuJoCo ↔ ROS2
```

### Resumen de Datos Propioceptivos Disponibles

| Dato | Clase | Atributo | Fuente MuJoCo |
|------|-------|----------|---------------|
| Posición articular | `JointState` | `position` | `data.qpos` |
| Velocidad articular | `JointState` | `velocity` | `data.qvel` |
| Aceleración articular | `JointState` | `acceleration` | `data.qacc` |
| Torque comandado | `JointState` | `torque_commanded` | `data.qfrc_actuator` |
| Torque externo | `JointState` | `torque_external` | `data.qfrc_passive` |
| Posición del torso | `TorsoState` | `position` | `data.xpos[torso_id]` |
| Orientación | `TorsoState` | `orientation_mat` | `data.xmat[torso_id]` |
| Vel. lineal | `TorsoState` | `linear_velocity` | `data.cvel[id][3:6]` |
| Vel. angular | `TorsoState` | `angular_velocity` | `data.cvel[id][0:3]` |
| Acelerómetro | `IMUData` | `linear_acceleration` | `data.cacc[id] + g` |
| Giroscopio | `IMUData` | `angular_velocity` | `data.cvel[id][0:3]` |
| Fuerza de contacto | `FootContact` | `force` | `mj_contactForce()` |
| Estado de contacto | `FootContact` | `is_contact` | `data.contact[i]` |
