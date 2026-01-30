# Vision-Based Control Framework for a Mobile Robot  
*(Work in Progress)*

## Overview

This repository contains a control framework for a vision-guided mobile robot.  
The project focuses on designing, implementing, and evaluating a modular control system that enables a wheeled robot to perceive its environment, estimate relative target position, and generate motion commands in real time.

The system follows a **model-based and modular robotics approach**, combining perception outputs with classical control strategies. While the current implementation relies on deterministic control logic, the framework is designed to support future extensions toward learning-based methods.

---

## Aim of the Project

The primary aim of this project is to develop a **reliable and interpretable control architecture** for a mobile robot operating based on visual feedback.

The project specifically aims to:
- Bridge perception and control in a real-time robotics system
- Validate control strategies using simulation and real hardware
- Provide a baseline system suitable for further research and learning-based extensions

---

## Project Goals

The main goals of the project are:
- Design a modular control system for a wheeled mobile robot
- Use vision-based measurements (distance and lateral offset) as control inputs
- Implement closed-loop control using classical control methods
- Handle perception uncertainty and transient failures
- Ensure real-time feasibility and debuggability
- Establish a foundation for future imitation or policy learning

---

## General System Architecture

The system follows a layered architecture commonly used in mobile robotics:
