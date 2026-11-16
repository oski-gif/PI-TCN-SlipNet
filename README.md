# PI-TCN-SlipNet
# Closed-Loop Tactile Slip Detection and Grip Control

## Overview

This project explores how a robotic hand can detect object slippage and respond by adjusting its grip in real time.

The system was developed in MuJoCo, using simulated tactile and physical sensor data to study the relationship between contact dynamics and object slip. Temporal Convolutional Networks (TCNs) were used to process sequential sensor information, while reinforcement learning (PPO) was incorporated into the robotic control framework.

The main objective was to move beyond simply detecting slip and investigate how a robotic system could use that information to respond before losing its grasp.

## Research Objectives

The project focused on three main questions:

- Can temporal neural networks reliably identify slip from sequential sensor measurements?
- Can physical information about contact and friction improve slip detection?
- Can a robotic hand use learned predictions and adaptive control to respond to slipping objects?

## System Architecture

The project was built around a simulated closed-loop robotic system:

1. **Simulation:** MuJoCo provides the robotic hand, objects, and contact physics.
2. **Data Collection:** Sensor measurements are collected over time during object manipulation.
3. **Slip Detection:** A Temporal Convolutional Network processes sequential observations to identify slip.
4. **Control:** The system uses its predictions and reinforcement learning-based control to adjust its grip.
5. **Feedback:** The updated grip changes the physical interaction, generating new observations for the next control cycle.

This creates a feedback loop between perception and physical action.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Primary development language |
| MuJoCo | Physics simulation and robotic manipulation |
| PyTorch | Neural network development |
| TCNs | Temporal slip detection |
| PPO | Reinforcement learning |
| NumPy | Numerical computation |
| Matplotlib | Data visualization and analysis |

## Experimental Evaluation

The experimental study included **1,005 matched trials** to evaluate system behavior under controlled conditions.

The experiments focused on comparing model performance, examining the contribution of physical information, and evaluating the system's ability to respond to slip during object manipulation.

The evaluation considered both predictive performance and closed-loop behavior rather than relying exclusively on classification accuracy.

## Repository Structure

The repository contains the simulation environment, model implementations, experimental scripts, and analysis used throughout the project.

## Running the Project

The project was developed in Python using MuJoCo and PyTorch.

Installation instructions, dependencies, and execution commands will be documented alongside the relevant source files.

## Limitations and Future Work

This project uses simulated environments rather than physical robotic hardware. Although MuJoCo provides a useful environment for studying contact dynamics, simulation results do not necessarily translate directly to real-world robotic manipulation.

Future work could investigate:

- Transfer of the trained models to physical robotic hardware.
- More complex object geometries and material properties.
- Improved modeling of tactile sensor behavior.
- More robust control under changing friction and contact conditions.

## Author

Independent research project exploring machine learning, robotic manipulation, and closed-loop control.


