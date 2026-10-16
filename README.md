# Multi-Agent Disaster Response Coordination System Using Cloud Infrastructure

## Project Documentation

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Objectives](#2-objectives)
3. [Tech Stack](#3-tech-stack)
4. [Proposed System Architecture](#4-proposed-system-architecture)
5. [Framework Design](#5-framework-design)
6. [Module-Wise Detailed Design](#6-module-wise-detailed-design)
7. [Work Completed So Far](#7-work-completed-so-far)
8. [Literature Survey](#8-literature-survey)

---

## 1. Problem Statement

Natural and man-made disasters — earthquakes, floods, fires, and structural collapses — demand immediate, coordinated response to minimise loss of life. Traditional disaster response relies heavily on human decision-making under extreme pressure: dispatchers must simultaneously track dozens of field units, triage victims with incomplete information, route ambulances through damaged infrastructure, and balance hospital loads — all within a rapidly evolving and chaotic environment. This approach is inherently limited by human cognitive bandwidth and the speed at which information can be collected, communicated, and processed.

**The core challenge this project addresses is:**

> *How can a fleet of heterogeneous autonomous agents (drones, ground robots, ambulances, boats, heavy lifters) be coordinated in real time using reinforcement learning, digital twin simulation, and cloud-based infrastructure to maximise victim rescue rates, optimise resource utilisation, and minimise response time during a multi-hazard disaster event?*

Specifically, the problem encompasses:

- **Exploration under Fog of War:** Agents begin with zero knowledge of the disaster zone and must efficiently explore a large, dynamically changing environment to locate victims.
- **Heterogeneous Multi-Agent Coordination:** Different agent types possess vastly different capabilities (aerial vs. ground vs. water-based movement, rescue ability, payload capacity, sensor range). Effective coordination requires role-specialised policies that complement each other.
- **Dynamic Environment:** The disaster landscape is not static. Fire spreads through cellular automata, floods expand, buildings collapse, and roads become blocked — agents must continuously adapt their strategies.
- **Constrained Resources:** Agents have limited battery, communication range, and payload. Hospitals have finite ICU beds and blood supply. The system must reason about resource constraints at every level.
- **Real-Time Decision Making:** With victim health degrading over time (critical > serious > stable), every second of delay reduces survival probability. The system must make near-instantaneous decisions at scale.
- **Scalability via Cloud Infrastructure:** The computational demands of running high-fidelity digital twin simulations, training deep RL policies across thousands of parallel environments, and executing real-time GNN-based traffic prediction necessitate a cloud-native, GPU-accelerated architecture.

---

## 2. Objectives

### 2.1 Primary Objectives

1. **Design and build a Digital Twin engine** that procedurally generates realistic urban disaster environments with multiple terrain types, road networks, buildings, hospitals, victims, and dynamic hazards (fire, flood, earthquake).

2. **Develop a Reinforcement Learning (RL) training pipeline** using Proximal Policy Optimization (PPO) to train autonomous agents that can independently scout, rescue, and transport victims in the digital twin environment.

3. **Implement GPU-accelerated training** by rewriting the entire simulation environment in JAX for 100% on-GPU execution, eliminating CPU–GPU synchronisation bottlenecks and achieving 100x–200x speedups over CPU-based training.

4. **Build a real-time backend server** (FastAPI + WebSocket) that runs the digital twin at 10 Hz, integrates trained RL models for live inference, and exposes REST APIs for dynamic disaster injection.

5. **Create an interactive frontend dashboard** (React + Three.js) for real-time 2D/3D visualization of the autonomous fleet, victim detection, and disaster progression.

6. **Implement a Cloud Coordinator AI** that orchestrates fleet-level decisions using centralised dispatch, GNN-based traffic prediction, hospital load balancing via bipartite matching, and traffic light preemption for emergency vehicles.

### 2.2 Secondary Objectives

7. **Multi-Agent Reinforcement Learning (MARL):** Transition from independent single-agent PPO to cooperative MARL using PettingZoo, enabling agents to learn collaborative strategies.

8. **Computer Vision Integration:** Incorporate OpenCV and YOLO (Ultralytics) for simulated drone camera feeds and real-time victim detection from imagery.

9. **Advanced Route Optimisation:** Upgrade from A\* pathfinding to fleet-level Vehicle Routing Problem (VRP) solvers using Google OR-Tools and NetworkX.

10. **Predictive Disaster Spread:** Extend the Cloud Coordinator to predict future disaster propagation (e.g., flood expansion modelling) and pre-emptively route agents to anticipated hotspots.

---

## 3. Tech Stack

### 3.1 Core Languages & Runtime

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Primary Language | Python | ≥ 3.10 | Backend, RL, simulation engine, predictive AI |
| Frontend Language | JavaScript (JSX) | ES2022 | Web-based dashboard and visualization |

### 3.2 Digital Twin & Simulation

| Component | Library | Purpose |
|-----------|---------|---------|
| Grid Engine | NumPy ≥ 1.24 | 2D terrain grid, fog of war, spatial operations |
| Disaster Physics | SciPy ≥ 1.11 | 2D convolution for fire/flood CA, bipartite matching |
| Configuration | PyYAML ≥ 6.0 | YAML-based city & hyperparameter configuration |
| Debug Renderer | Pygame ≥ 2.5 | 2D interactive visualization and debugging |
| Plotting | Matplotlib ≥ 3.8 | Training curves, analytics |
| Image Processing | Pillow ≥ 10.0 | Image export and manipulation |

### 3.3 Reinforcement Learning (CPU Pipeline)

| Component     | Library                 | Purpose                                            |
|---------------|-------------------------|----------------------------------------------------|
| RL Framework  | Stable Baselines3 ≥ 2.1 | PPO algorithm, vectorised environments, callbacks  |
| Env Interface | Gymnasium ≥ 0.29        | Standard RL environment API (`reset`, `step`)      |
| MARL Interface| PettingZoo ≥ 1.24       | Multi-agent parallel environment API               |
| Deep Learning | PyTorch ≥ 2.0           | Custom CNN feature extractor, GNN, policy networks |
| Logging       | TensorBoard ≥ 2.15      | Training metrics and reward curve monitoring       |

### 3.4 Reinforcement Learning (GPU-Accelerated Pipeline)

| Component       | Library         | Purpose                                                   |
|-----------------|-----------------|-----------------------------------------------------------|
| GPU Runtime     | JAX (CUDA 12)   | JIT compilation, `vmap`, `lax.scan` for 100% GPU training |
| Neural Networks | Flax (Linen API)| Actor-Critic CNN with LayerNorm and strided convolutions  |
| Optimiser       | Optax           | Adam with optional learning rate annealing                |
| Distributions   | Distrax         | Categorical action distributions for policy sampling      |

### 3.5 Predictive AI Layer

| Component          | Library                        | Purpose                                               |
|--------------------|--------------------------------|-------------------------------------------------------|
| Traffic GNN        | PyTorch (native GCN)           | Graph Convolutional Network for congestion prediction |
| Hospital Optimiser | SciPy (`linear_sum_assignment`)| Bipartite matching for patient-hospital assignment    |
| Cloud Coordinator  | Custom Python                  | Centralised dispatch, traffic light preemption        |

### 3.6 Backend & API

| Component       | Library           | Purpose                                  |
|-----------------|-------------------|------------------------------------------|
| Web Framework   | FastAPI ≥ 0.104   | REST API + async WebSocket server        |
| ASGI Server     | Uvicorn ≥ 0.24    | Production-grade async server            |
| Real-time Comms | WebSockets ≥ 12.0 | 10 Hz state broadcast to frontend clients|
| Data Validation | Pydantic ≥ 2.0    | Request/response schema validation       |

### 3.7 Frontend & Visualization

| Component | Library | Purpose |
|-----------|---------|---------|
| UI Framework | React 18 | Component-based reactive UI |
| Build Tool | Vite | Fast HMR dev server and bundler |
| 3D Rendering | Three.js / React Three Fiber | Tactical 3D scene visualization |
| 2D Rendering | HTML5 Canvas | Strategic 2D grid map with real-time updates |

### 3.8 Computer Vision (Planned)

| Component | Library | Purpose |
|-----------|---------|---------|
| Image Processing | OpenCV ≥ 4.8 | Simulated camera feed processing |
| Object Detection | Ultralytics (YOLOv8) ≥ 8.0 | Real-time victim detection from drone imagery |
| Augmentation | Albumentations ≥ 1.3 | Training data augmentation for CV pipeline |

### 3.9 Route Optimisation (Planned)

| Component | Library | Purpose |
|-----------|---------|---------|
| VRP Solver | Google OR-Tools ≥ 9.7 | Multi-vehicle routing problem optimisation |
| Graph Library | NetworkX ≥ 3.1 | Road network graph analysis and shortest paths |

### 3.10 Development & Testing

| Component | Library | Purpose |
|-----------|---------|---------|
| Testing | pytest ≥ 7.4 | Unit and integration testing |
| Coverage | pytest-cov ≥ 4.1 | Code coverage reporting |
| Linting | Ruff ≥ 0.1 | Fast Python linting and formatting |
| Type Checking | mypy ≥ 1.7 | Static type analysis |

---

## 4. Proposed System Architecture

### 4.1 High-Level Architecture Overview

The system follows a layered architecture with clear separation of concerns between the simulation engine, AI/ML layers, backend orchestration, and frontend visualisation.

```
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND (React + Three.js)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ Strategic 2D │  │ Tactical 3D  │  │   God Mode Panel   │    │
│  │   Map View   │  │  Scene View  │  │  (Disaster Inject) │    │
│  └──────┬───────┘  └──────┬───────┘  └─────────┬──────────┘    │
│         └─────────────────┴────────────────────┘               │
│                           │ WebSocket (10 Hz) + REST API       │
└───────────────────────────┼────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│                    BACKEND LAYER (FastAPI)                      │
│  ┌────────────────┐  ┌─────────────┐  ┌──────────────────┐      │
│  │  SimRunner     │  │  REST API   │  │  WebSocket       │      │
│  │  (10 Hz Loop)  │  │  Endpoints  │  │  Broadcaster     │      │
│  └───────┬────────┘  └──────┬──────┘  └─────────┬────────┘      │
│          │                  │                    │              │
│  ┌───────┴──────────────────┴────────────────────┘              │
│  │         AI Cloud Coordinator                                 │
│  │  ┌──────────────┐ ┌──────────────┐ ┌───────────────────┐     │
│  │  │ Traffic GNN  │ │  Hospital    │ │ Traffic Light      │    │
│  │  │ Predictor    │ │  Optimizer   │ │ Preemption         │    │
│  │  └──────────────┘ └──────────────┘ └───────────────────┘     │
│  └──────────────────────────────────────────────────────────    │
└───────────────────────────┼─────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│               REINFORCEMENT LEARNING LAYER                      │
│  ┌─────────────────────┐    ┌──────────────────────────────┐    │
│  │  CPU Pipeline (SB3) │    │  GPU Pipeline (JAX)          │    │
│  │  • DisasterEnv      │    │  • JAX Environment (vmap)    │    │
│  │  • DisasterCNN      │    │  • Flax ActorCritic CNN      │    │
│  │  • SubprocVecEnv    │    │  • PPO via lax.scan          │    │
│  │  • PPO Training     │    │  • 1024 parallel envs on GPU │    │
│  └─────────┬───────────┘    └──────────────┬───────────────┘    │
│            └────────────────┬──────────────┘                    │
│                             │ Trained Models (.zip / .pkl)      │
└─────────────────────────────┼───────────────────────────────────┘
                              │
┌─────────────────────────────┼───────────────────────────────────┐
│               DIGITAL TWIN ENGINE (Core)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │ CityGenerator│  │  WorldState  │  │  PhysicsEngine     │     │
│  │ (Procedural) │  │  (Truth)     │  │  (Movement/Action) │     │
│  └──────────────┘  └──────────────┘  └────────────────────┘     │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │  Disaster    │  │  Disaster    │  │  CityRenderer      │     │
│  │  Dynamics CA │  │  Injector    │  │  (Pygame Debug)    │     │
│  └──────────────┘  └──────────────┘  └────────────────────┘     │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │ Pathfinding  │  │  City Config │                             │
│  │ (A*)         │  │  & Presets   │                             │
│  └──────────────┘  └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow

1. **Training Phase:** The RL layer creates hundreds/thousands of parallel digital twin instances. Each instance generates observations → feeds them to the neural network → receives actions → steps the physics → computes rewards. Trained model weights are saved to disk.

2. **Inference Phase:** The backend server loads trained model weights. At 10 Hz, it queries the neural network for each agent's optimal action, steps the physics engine, advances disaster dynamics, runs the Cloud Coordinator, and broadcasts the full serialised world state to all connected frontend clients over WebSockets.

3. **User Interaction:** The frontend allows users to observe the autonomous fleet in real-time and dynamically inject disasters (earthquake, flood) via REST API calls, forcing the AI to adapt on-the-fly.

### 4.3 Agent Type Hierarchy

The system models 7 distinct agent types with heterogeneous capabilities:

| Agent Type | Movement | Speed | Sensor Range | Can Rescue | Can Transport | Terrain Access |
|-----------|----------|-------|-------------|-----------|--------------|----------------|
| Drone | Aerial | 3.0 | 8 cells | ✗ | ✗ | All terrain |
| Ambulance | Road-only | 4.0 | 3 cells | ✓ | ✓ | Road, Hospital, Charging |
| Ground Robot | Ground | 1.5 | 4 cells | ✓ | ✗ | Road, Debris, Park, etc. |
| Boat | Water-only | 2.0 | 5 cells | ✓ | ✓ | Water only |
| Heavy Lifter | Ground | 0.5 | 2 cells | ✗ (clears debris) | ✗ | Road, Debris, Park, etc. |
| Citizen | Ground | 0.5 | 2 cells | ✗ | ✗ | Road, Debris, Park |
| Traffic Light | Stationary | 0.0 | 5 cells | ✗ | ✗ | Road (fixed) |

---

## 5. Framework Design

### 5.1 Digital Twin Framework

The digital twin is built on a **procedural generation → mutable state → physics step** paradigm:

#### 5.1.1 Terrain Model (10 Types)

The environment is represented as a 2D integer grid where each cell holds one of 10 terrain types:

| ID | Terrain | Description |
|----|---------|-------------|
| 0 | EMPTY | Open ground, passable by ground units |
| 1 | ROAD | Paved surface, required for ambulances |
| 2 | BUILDING | Intact structure, impassable |
| 3 | BUILDING_DAMAGED | Collapsed structure, impassable, may harbour victims |
| 4 | DEBRIS | Rubble, slow traversal for ground units |
| 5 | WATER | River/flood zone, only boats can traverse |
| 6 | FIRE | Active fire, blocks ground units |
| 7 | HOSPITAL | Medical facility, ambulance drop-off point |
| 8 | CHARGING_STATION | Battery recharge point |
| 9 | PARK | Open green space, potential victim location |

#### 5.1.2 Disaster Dynamics — Cellular Automata

- **Fire Spread:** Uses 2D convolution (`scipy.signal.convolve2d`) with a 3×3 ignition kernel biased by wind direction. Burning cells ignite adjacent combustible cells (buildings, debris, parks) probabilistically. Fires burn out stochastically (5% per tick → Debris).

- **Flood Spread:** Orthogonal-only spread kernel. Water expands into permeable cells (empty, road, park, debris) with 20% probability per tick. Buildings block water flow.

#### 5.1.3 Victim Model

Victims have three severity levels affecting rescue priority and reward:
- **Critical:** Degrades fastest (1.0/step), reward multiplier ×3
- **Serious:** Moderate degradation (0.5/step), reward multiplier ×2
- **Stable:** Slowest degradation (0.2/step), reward multiplier ×1

#### 5.1.4 Communication Model

Agents operate under a **mesh networking** constraint:
- Each agent maintains a **local explored map** and **local known victims set**.
- Agents within `communication_range` (default 20 cells, Manhattan distance) automatically synchronise their local maps and victim knowledge.
- Agents near a hospital (base station) sync with the global network, receiving all discovered information.

### 5.2 Reinforcement Learning Framework

#### 5.2.1 Observation Space (6 Channels × FOV × FOV)

The agent's observation is a multi-channel spatial tensor representing its local Field of View:

| Channel | Name | Encoding |
|---------|------|----------|
| 0 | Terrain Map | Normalised terrain integer (0–9) / num_terrains |
| 1 | Fog of War | Binary — 1.0 = explored by this agent |
| 2 | Self Position | Binary — 1.0 at centre (agent's position) |
| 3 | Other Agents | Binary — 1.0 where other agents are located |
| 4 | Known Victims | Severity-encoded (0.2–1.0) for known victims |
| 5 | Target Compass | Single 1.0 pixel clamped to FOV edge indicating target direction |

#### 5.2.2 Action Space (6 Discrete Actions)

| Action | ID | Description |
|--------|-----|-------------|
| STAY | 0 | Hold position |
| UP | 1 | Move one cell north |
| DOWN | 2 | Move one cell south |
| LEFT | 3 | Move one cell west |
| RIGHT | 4 | Move one cell east |
| INTERACT | 5 | Context-sensitive: Rescue victim / Drop-off at hospital / Charge battery |

#### 5.2.3 Reward Structure

| Signal | Value | Trigger |
|--------|-------|---------|
| Exploration | +0.005 per cell | Newly revealed fog-of-war cells |
| Victim Rescue | +100.0 × severity multiplier | Agent rescues a victim |
| Hospital Delivery | +50.0 | Victim delivered to hospital |
| Collision Penalty | −5.0 | Agent hits obstacle or another agent |
| Time Penalty | −0.01 per step | Constant per-step cost to encourage speed |

#### 5.2.4 Dual Training Pipelines

**CPU Pipeline (Stable Baselines3):**
- 20 parallel `SubprocVecEnv` workers on CPU
- Custom `DisasterCNN` feature extractor (Conv2d → MaxPool → Conv2d → Flatten → Dense)
- PPO algorithm running on GPU for gradient updates
- ~2,000–5,000 steps/second throughput

**GPU Pipeline (JAX — Pure GPU):**
- 1,024 parallel environments via `jax.vmap` on a single GPU
- City pool of 128 pre-generated cities transferred to GPU once
- Flax `ActorCritic` CNN with strided convolutions and LayerNorm
- PPO via `jax.lax.scan` rollouts with GAE computation
- ~100,000–300,000 steps/second throughput (50x–100x speedup)

### 5.3 Cloud Coordinator Framework

The Cloud Coordinator acts as the "god-view" centralised dispatcher with three subsystems:

1. **Traffic GNN Predictor:** A 3-layer Graph Convolutional Network built on the road network graph. Node features: vehicle density, road damage, weather, disaster proximity, time-of-day cycle. Output: per-node congestion probability (0–1).

2. **Hospital Load Balancer:** Uses `scipy.optimize.linear_sum_assignment` (Hungarian algorithm) to solve the bipartite matching problem between victims-in-transit and hospital slots, minimising total transport distance while respecting ICU bed and blood supply constraints.

3. **Traffic Light Preemption:** Ambulances approaching within 3 cells of a traffic light force it to `preempted_green`, clearing the path for emergency vehicles.

---

## 6. Module-Wise Detailed Design

### 6.1 `disaster_sim/digital_twin/` — Digital Twin Core

| File | Lines | Role |
|------|-------|------|
| `city_config.py` | ~383 | Data classes: `Terrain` enum, `AgentTypeSpec`, `Severity`, `CityConfig` with YAML I/O, 3 built-in presets (small/medium/large), and 7 agent type specifications |
| `city_generator.py` | ~700 | Procedural city generation: road network layout, building placement, river/flood zone generation, hospital/charging station/park spawning, victim scattering, agent spawn point allocation |
| `world_state.py` | ~360 | Mutable runtime state: `AgentState`, `VictimState`, `HospitalState`, `SensorState`, `WorldState` class with fog-of-war reveal, coverage metrics, serialisation to dict for WebSocket transmission |
| `disaster_dynamics.py` | ~133 | Cellular automata for fire spread (wind-biased convolution) and flood spread (orthogonal permeation) |
| `disaster_injector.py` | ~116 | Dynamic disaster injection: `trigger_earthquake()` — distance/magnitude-based building damage; `trigger_flood()` — BFS-based flood expansion |

### 6.2 `disaster_sim/engine/` — Physics & Rendering

| File | Lines | Role |
|------|-------|------|
| `physics.py` | ~169 | `PhysicsEngine`: step-by-step agent movement, collision detection, battery drain, context-sensitive interaction (rescue/charge/drop-off), mesh network communication sync |
| `pathfinding.py` | ~93 | A\* pathfinding with "optimistic under uncertainty" heuristic — unexplored cells are assumed passable, enabling exploration behaviour |
| `renderer.py` | ~409 | Pygame-based 2D renderer with terrain colour coding, agent markers (diamond shapes), victim markers (coloured circles), hospital crosses, fog-of-war overlay, interactive camera (pan/zoom), and info panel |

### 6.3 `disaster_sim/rl/` — Reinforcement Learning

| File | Lines | Role |
|------|-------|------|
| `env.py` | ~263 | Single-agent Gymnasium environment (`DisasterEnv`) with interactive play mode and random baseline |
| `pz_env.py` | ~179 | PettingZoo `ParallelEnv` for multi-agent training with simultaneous actions |
| `observation.py` | ~143 | `ObservationBuilder`: constructs the 6-channel spatial tensor with target compass |
| `reward.py` | ~51 | `RewardCalculator`: dense + sparse reward signals |
| `networks.py` | ~44 | `DisasterCNN`: custom SB3 feature extractor (2-layer CNN + FC) |
| `train.py` | ~75 | SB3 PPO training script with `SubprocVecEnv` (20 workers) |
| `train_ppo.py` | ~80 | Configurable PPO training with presets and CLI arguments |
| `jax_env.py` | ~430 | Pure-JAX environment: `EnvState`/`EnvParams` pytrees, city pool, vmap-compatible reset/step |
| `jax_networks.py` | ~60 | Flax `ActorCritic` CNN with strided convolutions and LayerNorm |
| `jax_ppo.py` | ~260 | Pure-JAX PPO loop: `lax.scan` rollouts, GAE, clipped updates |
| `train_jax.py` | ~95 | CLI entry point for JAX training with argparse + YAML config |
| `tune_jax.py` | ~80 | Hyperparameter tuning script for JAX pipeline |
| `test_jax.py` | ~300 | JAX environment unit tests and validation |
| `callbacks.py` | ~40 | SB3 training callbacks for logging |

### 6.4 `disaster_sim/predictive/` — AI Cloud Coordinator

| File | Lines | Role |
|------|-------|------|
| `cloud_coordinator.py` | ~84 | Centralised coordinator: integrates traffic prediction, hospital optimisation, traffic light preemption into a single `tick()` cycle |
| `traffic_gnn.py` | ~167 | `TrafficGNN`: 3-layer GCN for road network congestion prediction. `TrafficPredictorSystem`: extracts road graph, builds normalised adjacency matrix, extracts real-time node features, runs inference |
| `hospital_optimizer.py` | ~100 | `HospitalOptimizer`: builds victim-hospital cost matrix (Manhattan distance), solves via Hungarian algorithm, applies assignments while respecting ICU and blood constraints |

### 6.5 `disaster_sim/routing/` — Pathfinding

| File | Lines | Role |
|------|-------|------|
| `astar.py` | ~67 | Standalone A\* implementation for agent-type-aware pathfinding with terrain passability checks |

### 6.6 `disaster_sim/backend/` — Server

| File | Lines | Role |
|------|-------|------|
| `server.py` | ~178 | FastAPI application: `SimRunner` class manages the 10 Hz simulation loop, loads trained models for inference, broadcasts state via WebSocket, exposes REST endpoints for earthquake/flood injection |

### 6.7 `frontend/` — Web Dashboard

| File | Role |
|------|------|
| `src/App.jsx` | Main application: view mode toggle (2D/3D), disaster injection panel, live telemetry sidebar |
| `src/components/StrategicMap.jsx` | Canvas-based 2D grid renderer with terrain colouring, agent/victim markers, traffic heatmap overlay, click-to-select for disaster targeting |
| `src/components/TacticalView.jsx` | Three.js-based 3D scene for tactical visualization |
| `src/hooks/useDisasterState.js` | WebSocket hook with auto-reconnect, REST API helpers for disaster injection |

### 6.8 `configs/` — Configuration Files

| File | Purpose |
|------|---------|
| `city_small.yaml` | Small city preset (50×50) |
| `city_medium.yaml` | Medium city preset (200×200) |
| `city_large.yaml` | Large city preset (400×400) |
| `ppo_jax.yaml` | JAX PPO hyperparameters optimised for RTX 4060 (8 GB VRAM) |
| `ppo_small.yaml` | Small-scale PPO training configuration |
| `ppo_jax_drone_best.yaml` | Best drone training hyperparameters |

### 6.9 `tests/` & `scripts/`

| File | Purpose |
|------|---------|
| `tests/test_city_generator.py` | 20+ unit tests: config validation, grid generation, road connectivity, victim/agent placement, seed reproducibility, world state serialisation, terrain passability |
| `scripts/experiment_runner.py` | Automated benchmarking: runs N headless episodes, collects statistics (rescue rate, coverage, energy, collisions), exports to CSV |

---

## 7. Work Completed So Far

### 7.1 Summary of Milestones

| #  | Milestone                        | Status | Details |
|----|---------------------------------|--------|---------|
| 1  | Digital Twin Core Engine        | ✅ Complete | Procedural city generation, 10 terrain types, 7 agent types, 3 city presets (50×50, 200×200, 400×400) |
| 2  | Disaster Dynamics               | ⏳ P | Cellular automata for fire (wind-biased convolution) and flood spread, earthquake/flood injection API |
| 3  | Physics & Interaction           | ⏳ P | Movement, collision detection, battery management, rescue/charge/transport mechanics, mesh network communication |
| 4  | RL Environment (Single Agent)   | ⏳ P | Gymnasium wrapper, 6-channel observation builder, reward function, interactive play mode |
| 5  | RL Environment (Multi Agent)    | ⏳ P | PettingZoo `ParallelEnv` with simultaneous action execution |
| 6  | PPO Training — CPU Pipeline     | ⏳ P | SB3-based, custom CNN, 20× `SubprocVecEnv` parallelism, TensorBoard logging |
| 7  | PPO Training — GPU Pipeline     | ⏳ P | Full JAX rewrite: 1024 parallel envs on GPU, 100k–300k SPS, Flax ActorCritic CNN, trained drone model saved (`drone_jax_ppo.pkl`, 4 MB) |
| 8  | Backend Server                  | ⏳ P | FastAPI + WebSocket at 10 Hz, trained model inference, disaster injection REST API |
| 9  | Frontend Dashboard              | ⬜ Planned | React + Vite, 2D strategic map (Canvas), 3D tactical view (Three.js), telemetry sidebar, God Mode disaster injection panel |
| 10 | Cloud Coordinator AI            | ⬜ Planned | Traffic GNN predictor, hospital load balancer (Hungarian algorithm), traffic light preemption |
| 11 | Pathfinding                     | ⬜ Planned | A\* with optimistic-under-uncertainty heuristic + standalone routing A\* |
| 12 | Testing & Benchmarking          | ⬜ Planned | 20+ unit tests, experiment runner for headless benchmarking |
| 13 | Computer Vision (YOLO)          | ⬜ Planned  | OpenCV + Ultralytics integration for simulated camera feeds |
| 14 | Advanced Route Optimisation (VRP) | ⬜ Planned | Google OR-Tools + NetworkX for fleet-level routing |
| 15 | Predictive Disaster Spread      | ⬜ Planned | ML-based disaster trajectory prediction |
| 16 | Full MARL Training              | ⬜ Planned | Cooperative multi-agent RL via PettingZoo with shared rewards |


### 7.2 Key Technical Achievements

1. **50–100× Training Speedup:** By rewriting the entire environment in JAX and running 1,024 parallel environments on a single RTX 4060, the training throughput increased from ~2,000–5,000 SPS (CPU + SB3) to ~100,000–300,000 SPS (GPU + JAX).

2. **Target Compass Observation:** Replaced computationally expensive per-step A\* pathfinding in the observation builder with a lightweight compass channel that points toward the agent's current target (nearest unexplored area for drones, nearest victim/hospital for ambulances). This reduced observation computation time while maintaining policy quality.

3. **Mesh Networking Constraint:** Agents don't have global visibility — they must physically explore and communicate with nearby agents or base stations to share information, creating a realistic partial observability challenge.

4. **Trained Drone Model:** A working PPO-trained drone model (`drone_jax_ppo.pkl`, 4 MB) has been saved from the JAX pipeline, demonstrating end-to-end training viability.

### 7.3 Lines of Code Summary

| Module | Approx. LoC | Language |
|--------|-------------|----------|
| `digital_twin/` | ~1,600 | Python |
| `engine/` | ~670 | Python |
| `rl/` | ~2,600 | Python |
| `predictive/` | ~350 | Python |
| `routing/` | ~70 | Python |
| `backend/` | ~180 | Python |
| `frontend/` | ~450 | JavaScript (JSX) |
| `tests/` | ~320 | Python |
| `scripts/` | ~75 | Python |
| `configs/` | ~100 | YAML |
| **Total** | **~6,400** | — |

---

## 8. Literature Survey

### Shubhhra Prakash (Detailed at: /literature_survey/LiteratureReview_Shubhra.pdf)
When we look at all five of these papers together, a really exciting picture emerges for our cloud-based disaster response system! The big theme connecting everything is the incredible potential of combining Digital Twins which are basically live, virtual copies of the real world with smart, learning AI agents to make split-second decisions in chaotic environments.
Instead of just using AI as a static tool, these studies show how we can build active ecosystems where multiple AI specialists talk to each other and learn on the fly. For example, the research highlights how we can use a main orchestrator AI to coordinate various sub-agents during a crisis, breaking down data silos so that rescue teams and logistics planners get exactly the insights they need. We also have clear, structured blueprints for how to make these distributed agents monitor catastrophic events and communicate seamlessly across a network.
At the same time, we see massive proof that plugging Reinforcement Learning into these digital twins allows the system to constantly adapt to its environment. This kind of setup can actively figure out safe routes and coordinate movements while completely avoiding collisions. It can also predict exactly what needs attention first by looking at real-time data and system constraints, which is perfect for prioritizing our scarce resources during an emergency.
What is really inspiring is that we do not even have to start from scratch to test these ideas. There are open-source simulation frameworks out there designed specifically to handle these digital models, and they allow us to distribute the heavy computational work across cloud clusters to keep everything running fast.
Overall, by taking the real-time routing and predictive ideas from the factory-focused papers and merging them with the disaster-specific multi-agent architectures, we have a complete and practical blueprint. We can build a coordination system that doesn't just react to emergencies, but actively anticipates them. 

### Kaustubh Kanodia (Detailed at: /literature_survey/LiteratureReview_Kaustubh.pdf)
The strongest idea across the literature is that digital twins and reinforcement learning can be combined to create intelligent, adaptive decision-making systems that operate in real time. Rather than using simulation only for offline testing, these studies show that digital twins can become active environments where AI agents learn, adapt, and improve continuously as conditions change. This is especially important for disaster response, where the environment is unpredictable and every second matters.
A major theme across the papers is the rise of multi-agent and orchestrated intelligence. The work on agentic AI systems and intelligent multi-agent digital twins shows that complex crises are best handled by distributed specialized agents rather than one monolithic model. This supports the idea of a central coordinator that can manage multiple sub-agents, share situational awareness, and reduce information silos between sensing, routing, and resource allocation tasks. In our system, this directly maps to the cloud coordinator layer, where different agents such as drones, ambulances, boats, and ground robots can operate with shared context but different responsibilities.
Another strong contribution from the papers is the proof that reinforcement learning inside digital twins can improve operational decision-making. The maintenance prioritization and production scheduling papers demonstrate that RL can learn useful policies for ranking tasks, adapting schedules, and optimizing actions under changing constraints. In a disaster setting, this is highly relevant because the same logic can be extended to victim prioritization, route selection, hospital assignment, and dynamic resource balancing. The key insight is that RL is not just for abstract games or static control problems — it can be used to make practical decisions in environments that are incomplete, uncertain, and constantly evolving.
What is also especially valuable is that these studies show simulation-driven systems are becoming practical and scalable, not just theoretical. The digital model playground and other digital twin frameworks provide evidence that these environments can support repeated experimentation, rapid policy testing, and cloud-based execution. This is a strong match for our project because disaster response requires both high-fidelity simulation for training and fast inference for deployment. By combining these ideas with cloud infrastructure, the system can scale to many parallel scenarios, train faster, and remain responsive during real-time operation.
Overall, these papers give us a strong foundation for building a disaster response platform that is not only reactive but predictive and coordinated. They support a system where digital twins simulate the crisis, reinforcement learning learns better response strategies, and multi-agent orchestration ensures that the right resources reach the right place at the right time. For our project, this means we are not simply building a simulator or an AI model — we are building an integrated decision-support ecosystem for intelligent disaster management.

### Jayant Sharma (Detailed at: /literature_survey/LiteratureReview_Jayant.pdf)
The reviewed literature highlights that Digital Twin technology is becoming a key enabler for intelligent disaster management by providing real-time virtual representations of disaster-affected environments. Unlike traditional monitoring systems, digital twins continuously integrate data from IoT devices, remote sensing, GIS platforms, and emergency services to provide an accurate and up-to-date operational picture. This significantly improves situational awareness and enables emergency responders to make faster and more informed decisions during rapidly evolving disaster scenarios.
Another common finding across the selected studies is the growing integration of artificial intelligence and reinforcement learning into digital twin environments. Rather than relying on fixed rules or static planning methods, reinforcement learning enables decision-support systems to continuously learn from simulated disaster scenarios and optimize actions such as rescue routing, evacuation planning, resource allocation, and emergency logistics. These adaptive capabilities make disaster response systems more resilient when operating under uncertain and dynamic conditions.
The literature also emphasizes the importance of coordinated decision-making across multiple stakeholders and intelligent systems. Modern disaster management requires seamless collaboration between emergency responders, drones, hospitals, communication networks, and command centers. Digital twins provide a unified platform where information from these heterogeneous sources can be shared in real time, improving coordination while minimizing communication delays and information gaps.
Despite these advancements, several challenges remain. The studies identify issues such as interoperability between different emergency systems, computational overhead associated with maintaining high-fidelity digital twins, limited real-world validation, and the difficulty of scaling these frameworks to support multiple disaster types simultaneously. These limitations indicate that further research is needed to develop integrated, cloud-based solutions capable of supporting large-scale disaster response operations.
Overall, the reviewed papers provide a strong foundation for the proposed project. They demonstrate that combining Digital Twins, Reinforcement Learning, cloud computing, and intelligent decision-support techniques can significantly enhance disaster preparedness and emergency response. Building upon these recent developments, the proposed system aims to deliver a scalable cloud-based disaster management platform that supports real-time monitoring, adaptive decision-making, efficient resource utilization, and coordinated emergency response across diverse disaster scenarios.

---