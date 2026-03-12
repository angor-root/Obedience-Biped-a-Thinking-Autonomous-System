# Obedience

**A Bipedal Robot for Autonomous Medicine Delivery**

> *"Obedience is the proper duty of a reasonable soul."*  
> — Michel de Montaigne, *Essais*, II, XII

---

## Overview

**Obedience** is an autonomous bipedal robot designed to deliver medications to patient bedsides on a scheduled basis. The system integrates capture-point-based walking control with intelligent power management and fault-tolerant behaviors.

### Key Features

- **Periodic medicine delivery** to designated patient locations
- **Intelligent battery management** with pre-trip charge verification
- **Automatic return-to-charger** behavior when low battery is detected
- **Anomalous discharge detection** with predictive reach estimation
- **Safe sit-down posture** for graceful degradation when charging station is unreachable
- **Critical state notifications** to healthcare staff

---

## System Architecture

This project follows a **constellation architecture** for autonomous systems, implemented using **ROS 2**:

```
┌─────────────────────────────────────────────────────────────┐
│                    OBEDIENCE CONSTELLATION                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Mission   │  │  Navigation │  │   Battery   │         │
│  │   Planner   │──│  Controller │──│   Monitor   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          │                                  │
│                  ┌───────┴───────┐                          │
│                  │   Walking     │                          │
│                  │   Controller  │                          │
│                  │ (Capture Point)│                          │
│                  └───────────────┘                          │
│                          │                                  │
│                  ┌───────┴───────┐                          │
│                  │    Robot      │                          │
│                  │   Hardware    │                          │
│                  └───────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Walking Controller

The bipedal locomotion is based on the **Capture Point** control method, which provides dynamically stable walking without requiring complex trajectory optimization.

### Base Implementation

This project builds upon the bipedal walking implementation from **[The5439Workshop](https://github.com/The5439Workshop)**:

- **Original Repository**: [Bipedal_walking_capture_point](https://github.com/The5439Workshop/Bipedal_walking_capture_point)
- **Reference Video**: [How to Make a Robot Walk (No AI, Just Physics)](https://www.youtube.com/watch?v=RXGrTD71FMc)

The capture point controller calculates optimal foot placement using inverted pendulum dynamics to maintain balance during walking.

---

## Project Structure

```
obedience/
├── config/                     # Configuration files
├── launch/                     # ROS 2 launch files
├── src/
│   ├── walking/               # Capture point walking controller
│   │   ├── capture_point.py   # Main walking controller
│   │   ├── jacobian.py        # Kinematic jacobian computations
│   │   └── utils.py           # Utility functions
│   ├── battery/               # Battery management system
│   ├── navigation/            # Navigation and path planning
│   └── mission/               # Mission planning and scheduling
├── models/
│   └── xml/                   # MuJoCo robot models
│       └── biped_3d_feet.xml
├── urdf/                       # URDF robot descriptions
└── tests/                      # Unit and integration tests
```

---

## Dependencies

| Package    | Version |
|------------|---------|
| Python     | ≥ 3.11  |
| MuJoCo     | ≥ 3.4   |
| NumPy      | ≥ 2.3   |
| Pinocchio  | ≥ 3.8   |
| ROS 2      | Humble+ |

---

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/obedience.git
cd obedience

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install mujoco numpy pinocchio
```

---

## Quick Start

```bash
# Run walking simulation
python src/walking/capture_point.py
```

---

## Acknowledgments

- Walking controller based on work by **The5439Workshop** (#The5439Workshop)
- Capture Point theory from Pratt et al., *"Capture Point: A Step toward Humanoid Push Recovery"*

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{obedience2026,
  title={Obedience: A Bipedal Robot for Autonomous Medicine Delivery},
  author={Your Name},
  year={2026},
  url={https://github.com/YOUR_USERNAME/obedience}
}
```
