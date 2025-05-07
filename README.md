# Trust-Aware Motion Planning in Human-Robot Collaboration under Distribution Temporal Logic Specifications

This repository accompanies the simulation study in our ICRA 2024 paper:

**Trust-aware motion planning for human-robot collaboration under distribution temporal logic specifications**

## Table of Contents

- [Project Overview](#project-overview)
- [Simulation Environment](#simulation-environment)
- [Experimental Procedure](#experimental-procedure)
  - [Participant Instructions](#participant-instructions)
  - [Task 1: Manual Takeover When Necessary](#task-1-manual-takeover-when-necessary)
  - [Task 2: Real-Time Trust Level Adjustment](#task-2-real-time-trust-level-adjustment)
- [Control Interface](#control-interface)
- [Data Recording](#data-recording)
- [Citation](#citation)

---

## Project Overview

This project investigates human trust dynamics in human-robot collaboration (HRC) settings using a trust-aware autonomous driving simulator. The framework is built upon the **partially observable Markov decision process (POMDP)** model, incorporating **syntactically co-safe linear distribution temporal logic (scLDTL)** to formally specify trust-related behavioral goals.

We propose an optimal policy synthesis algorithm based on:
- A probabilistically labelled belief MDP,
- A product construction between the belief MDP and scLDTL automaton,
- Modified point-based value iteration (PBVI) for scalable planning.

A human-subject study using the **CARLA simulator** was conducted with 21 participants to validate the approach. Participants interacted with a trust-aware self-driving car system and reported trust levels in real-time across multiple routes and scenarios.

---

## Simulation Environment

The simulation is conducted in [CARLA 0.9.13](https://carla.org), a high-fidelity driving simulator. The setup includes:
- **Autonomous driving with trust-aware control**,
- **Manual override** by participants when needed,
- Real-time **trust level reporting**,
- Four predefined driving routes and one warm-up route,
- Mixed road scenarios with both dynamic (pedestrian, vehicle) and static (barrier, parked object) obstacles.

Participants interact with the simulator using a **steering wheel setup**, including:
- Pedals for throttle and brake,
- Steering for directional control,
- Custom button bindings for trust adjustment and control mode switching.

---

## Experimental Procedure

### Setup

1. Launch the CARLA server in Terminal 1:
    ```bash
    conda activate CARLA_VIRTUAL_ENV
    cd CARLA_0.9.13_folder
    ./CarlaUE4.sh
    ```
    > If a core dump error occurs, restart CARLA by rerunning the above commands.

2. Launch the simulation script in Terminal 2:
    ```bash
    conda activate CARLA_VIRTUAL_ENV
    cd trust_aware_hrc
    ```

3. Run a test trial (warm-up):
    ```bash
    python3 manual_control_steeringwheel_trust_aware_hrc_fix_route.py --sync --autopilot --trial_id 3 --if_baseline_trial 0 --driver_id 0
    ```

4. For each participant, run three full trials and corresponding baseline (use unique `--driver_id`):
    ```bash
    # Trial 1
    python3 manual_control_steeringwheel_trust_aware_hrc_fix_route.py --sync --autopilot --trial_id 1 --if_baseline_trial 0 --driver_id 0
    python3 manual_control_steeringwheel_trust_aware_hrc_fix_route.py --sync --autopilot --trial_id 1 --if_baseline_trial 1 --driver_id 0

    # Trial 2
    python3 manual_control_steeringwheel_trust_aware_hrc_fix_route.py --sync --autopilot --trial_id 2 --if_baseline_trial 0 --driver_id 0
    python3 manual_control_steeringwheel_trust_aware_hrc_fix_route.py --sync --autopilot --trial_id 2 --if_baseline_trial 1 --driver_id 0
    ```

### Participant Instructions

Each participant completes:
- 1 warm-up route (for familiarization),
- 4 experimental routes. For each route, participants must simultaneously complete **two core tasks**:

#### Task 1: Manual Takeover When Necessary

You are expected to monitor the vehicle's behavior and **take manual control only when you believe it is necessary** to ensure safety. For example, you may decide to intervene when:

- A pedestrian, bicycle, or truck appears ahead,
- You do not trust the car’s current decision.

To take over, press **R3 (Button 10)** on the steering wheel to switch from autopilot to manual control. You may then steer and control the pedals manually.

**Important rules for takeover:**
- You may **only take over once per road segment**.
- You **must return to autopilot** (press R3 again) before the car reaches an intersection (indicated by a traffic light).
- Always ensure you **return to the right lane** before switching back to autopilot.

#### Task 2: Real-Time Trust Level Adjustment

While the car is driving in autopilot mode, you must **regularly assess and adjust your trust level** in its decisions.

- The trust level ranges from **1 (lowest)** to **7 (highest)**.
- Initial trust level is set to 4.
- Adjustments are made using the steering wheel buttons:
  - **L2 (Button 7)**: Increase trust level (+1)
  - **L3 (Button 11)**: Decrease trust level (–1)

You should adjust the trust level based on your **subjective satisfaction or concern** with the car's actions. For example:
- If the car successfully avoids an obstacle, you might increase your trust.
- If it makes an unsafe or suspicious maneuver, you might decrease it.

**Minimum requirement:** You must assess and (if needed) update the trust level **at least once per road segment or at each intersection**.

> 🛑 Ensure `Autopilot: ON` before intersections and press `TAB` to switch to the driver's view when simulation starts.

---

## Control Interface

| Function                  | Control Input     |
|---------------------------|-------------------|
| Brake                     | Middle pedal      |
| Throttle                  | Right pedal       |
| Reverse                   | R2 (6)            |
| Toggle manual/autopilot  | R3 (10)           |
| Increase trust level     | L2 (7)            |
| Decrease trust level     | L3 (11)           |

> Please **do not press** other buttons — this may disrupt the simulation.

---

## Data Recording

Trust level responses for each participant are recorded in:
```
/trust_level_record/
```
Each participant should generate **5 files** (1 warm-up + 4 experimental routes).

---

## Citation

If you use this simulation setup or build upon our work, please cite:

Yu, Pian, Dong, Shuyang, Sheng, Shili, Feng, Lu, and Kwiatkowska, Marta.  
**Trust-aware motion planning for human-robot collaboration under distribution temporal logic specifications.**  
*In Proceedings of the 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 12949–12955. IEEE, 2024.*

### BibTeX
```bibtex
@inproceedings{yu2024trust,
  title={Trust-aware motion planning for human-robot collaboration under distribution temporal logic specifications},
  author={Yu, Pian and Dong, Shuyang and Sheng, Shili and Feng, Lu and Kwiatkowska, Marta},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={12949--12955},
  year={2024},
  organization={IEEE}
}


