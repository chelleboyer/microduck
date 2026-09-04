# Microduck Hopscotch — Project Brief

> How we're building it: [microduck-hopscotch-architecture.md](./microduck-hopscotch-architecture.md)

## Goal

Train Pollen Robotics’ Microduck to perform hopscotch, starting entirely in simulation and ultimately transferring the learned behavior to the real robot.

## What I Want to Accomplish

Use Microduck’s existing MuJoCo reinforcement-learning environment as the foundation rather than building a robotics stack from scratch.

The project should:

- Create a simulated **hopscotch course** for Microduck.
- Teach Microduck to **jump, land, balance, and progress through the course**.
- Train progressively, beginning with simple hopping and eventually learning a recognizable hopscotch sequence.
- Use **reinforcement learning**, allowing the robot to improve through repeated simulated attempts.
- Run the computationally intensive training on **Hugging Face Jobs**, since my Windows laptop does not have a GPU.
- Save the resulting trained policy/model.
- Test the trained behavior in MuJoCo.
- Ultimately deploy the successful policy to my **physical Microduck**.

## Development Approach

Start with the existing Microduck walking/training project and prove that Hugging Face Jobs can successfully run its normal training. Once that pipeline works, modify the simulation and training objective specifically for hopscotch.

The first milestone isn't a perfect hopscotch routine.

**Success #1 = Microduck learns to intentionally hop forward and land upright.**

From there, build toward accurate landings, consecutive hops, and finally the full hopscotch pattern.
