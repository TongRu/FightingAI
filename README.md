# FightingAI

**Note:** including code for both training and testing!

**AI Solutions for Fighting Game Based on Traditional Reinforcement Learning Algorithms**

two traditional reinforcement algorithms Q-learning and Monte Carlo

This is the official implementation with *training* code for SiamMask (CVPR2019). For technical details, please refer to:

**[[Paper](http)] [[DemoVideo](https://github.com/TongRu/FightingAI/blob/master/demo/demo-P2(designedFightingAI).mp4)] [[Project Page](https://github.com/TongRu/FightingAI)]** <br />

<div align="center">
  <img src="https://github.com/TongRu/FightingAI/blob/master/demo/FightingAI.png" width="600px" />
</div>

## Contents

1. [Environment Setup](#environment-setup)
2. [Code structure](#code-structure)

## Environment setup

This code has been tested on Window 10, Python 3.7

- Clone the repository 

```
git clone https://github.com/TongRu/FightingAI.git
```

- Install OpenJDK (java environment)

  1. Download the installation package for window from [https://jdk.java.net/14/](https://jdk.java.net/14/)

  2. Unzip it to a fixed path, such as D:\jdk-14.0.1

  3. Add java program to system path: Right-click to open My Computer → Properties → Advanced System Settings → open the environment variables → edit and create a new path "D:\jdk-14.0.1\bin" in the user variable -Path

  4. whether java is successfully installed: open the cmd command window and enter “java -version” to display java information.

  if you apply the code in in Linux, please directly execute the following instructions to install OpenJDK

  ```
  $ sudo apt-get install openjdk-8-jre
  ```

- Setup python environment
```
conda create -n FightingAI python
source activate FightingAI
pip install gym
pip install py4j
pip install port_for
pip install opencv-python
```

## Code-structure

.\fightingice_env.py: starting code of the Fighting Game Platform;

.\data\ai\MctsAi.jar: Opponent bot based on java env;

.\train-MC.py: code for training traditional Q-learning algorithm;

.\train-QLearning.py: code for training traditional MC algorithm;

.\train-ImprovedQL.py: code for training improved Q-learning algorithm;

.\test-MC.py: code for testing traditional Q-learning algorithm;

.\test-QLearning.py: code for testing traditional MC algorithm;

.\test-ImprovedQL.py: code for testing improved Q-learning algorithm.