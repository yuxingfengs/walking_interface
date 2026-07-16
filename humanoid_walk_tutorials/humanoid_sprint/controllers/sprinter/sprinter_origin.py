from controller import Robot, Motion, Supervisor
from collections import deque
from torch.nn.utils.rnn import pad_sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
import torch.nn.utils.rnn as rnn_utils
import matplotlib.pyplot as plt
import math
import os,csv,json
import copy
import statistics
from pathlib import Path
from datetime import datetime
from helperfunction import (
    load_initial_actions_csv,
    _prepare_logging,
    _append_metric,
    _save_meta,
    save_checkpoint,
    load_checkpoint,
    plot_q_reward_over_updates_gs,
    plot_q_reward_from_csv_gs,
    rotate_metrics_csv,
    load_bc,
    load_actor_from_checkpoint
)
STATE_DIM = 18
ACTION_DIM = 100
TRAIN_EPISODES=200
NUM_EPISODES =100

SEQUENCE_LENGTH=35
INITIAL_STATE=[2.813063620922607e-07, -2.8354001787624697e-07, 2.661361167672216e-07, 2.680192760864757e-07, 2.967163694572752e-07, 2.9883194662671264e-07, 2.4820736847423014e-07, 2.4949180207911914e-07, 1.9713857411518186e-07, 1.9756885276656546e-07, -1.866298555697199e-07, 1.8715301552270632e-07, [-5.021522061737612,  0.9999547666320551, 0.7058214811932133]]
ACTOR_LR=1e-6
CRITIC_LR=1e-6
MAX_FALLS= 20
CKPTPATH="ddpg_checkpoint.pth"
CKPTLPATH="ddpg_checkpoint_false.pth"

SMOOTH_K=5 
INITIAL_ACTIONS = load_initial_actions_csv(
    "initial_actions.csv",   # 同目录下的文件名
    expected_dim=10,         # 动作维度
    to_radians=False,        # 如果是度就设 True
    device="cpu",
    dtype=torch.float32
)

class ReplayBuffer:

    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, action, reward, next_state, done, next_action=None):

        def to_list(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().tolist()
            return x

        s = to_list(state)
        a = to_list(action)
        ns = to_list(next_state)
        na = to_list(next_action) if next_action is not None else None
        r = float(reward) if not isinstance(reward, torch.Tensor) else reward.item()
        d = bool(done)    if not isinstance(done,    torch.Tensor) else bool(done.item())

        self.buffer.append((s, a, r, ns, d, na))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample_batch(self, batch_size):
        
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones, next_actions = zip(*batch)

        states      = torch.tensor(states,      dtype=torch.float32)
        actions     = torch.tensor(actions,     dtype=torch.float32)
        rewards     = torch.tensor(rewards,     dtype=torch.float32).unsqueeze(-1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones       = torch.tensor(dones,       dtype=torch.float32).unsqueeze(-1)

        if next_actions[0] is not None:
            next_actions = torch.tensor(next_actions, dtype=torch.float32)
        else:
            next_actions = None

        return states, actions, rewards, next_states, dones, next_actions

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)





class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim):
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        layers = []
        inp = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(inp, h), nn.ReLU()]
            inp = h
        layers.append(nn.Linear(inp, action_dim))
        self.net = nn.Sequential(*layers)

        self.register_buffer("joint_min", torch.tensor([
            -0.379, -1.774, -0.092, -1.189, -0.397,
            -0.738, -1.774, -0.092, -1.186, -0.769
        ], dtype=torch.float32))
        self.register_buffer("joint_max", torch.tensor([
            0.790, 0.484, 2.113, 0.923, 0.769,
            0.450, 0.484, 2.113, 0.932, 0.38
        ], dtype=torch.float32))
        
    def forward(self, state):

        if isinstance(state, (list, tuple, np.ndarray)):
            state = torch.tensor(state, dtype=torch.float32, device=self.net[0].weight.device)
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.net[0].weight.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)
        raw = self.net(state)

        action = torch.max(torch.min(raw, self.joint_max), self.joint_min)
        return action
        

        
        

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        self.l1  = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.l2  = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.l3  = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
      
        if state.dim()  == 1: state  = state.unsqueeze(0)
        if action.dim() == 1: action = action.unsqueeze(0)
        x = F.relu(self.bn1(self.l1(state)))          # [B, hidden_dim]
        x = torch.cat([x, action], dim=-1)            # [B, hidden_dim+action_dim]
        x = F.relu(self.bn2(self.l2(x)))              # [B, hidden_dim]
        q = self.l3(x)                                # [B, 1]
        return q



class DDPG:
    def __init__(self, state_dim, action_dim, hidden_dim,
                 actor_lr, critic_lr, device,
                 gamma=0.98, tau=1e-4):
        
        self.device = device
        self.gamma = gamma
        self.tau   = tau
        self.actor  = Actor(state_dim, 256, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)

        self.actor_target  = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.device = device 
        self.batch_size = 64
        self.min_buffer_size =3
        self.replay_buffer = ReplayBuffer(capacity=3000)
     
        self.last_train_metrics={}
        self._train_logs = {
            "rewards": [], "target_qs": [], "current_qs": [],
            "actor_losses": [], "critic_losses": [], "q_vs_r_pairs": [] 
        }
        self.actor_update_start = 0

    def soft_update(self, net, net_target):
        for param, target_param in zip(net.parameters(), net_target.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * param.data)
    


    def train(self, critic_updates: int = 2):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0

        states, actions, rewards, next_states, dones, _ = \
            self.replay_buffer.sample_batch(self.batch_size)

        device = self.device
        states      = states.to(device)
        actions     = actions.to(device)
        rewards     = rewards.to(device).view(-1)
        next_states = next_states.to(device)
        dones       = dones.to(device).view(-1)

        # --- 先前向，拿到 current / target Q（便于统计）---
        with torch.no_grad():
            next_a   = self.actor_target(next_states)
            next_q   = self.critic_target(next_states, next_a).view(-1)
            target_q = rewards + self.gamma * (1 - dones) * next_q

        current_q = self.critic(states, actions).view(-1)

        # ★ TD-error 分布（收敛检测 #2）
        td_error = (target_q - current_q).detach().cpu()

        # ★ Bellman consistency（收敛检测 #3）
        q_current_mean = float(current_q.mean().item())
        q_target_mean  = float(target_q.mean().item())
        bellman_gap    = q_target_mean - q_current_mean

        # 其余已有统计
        td_mse   = F.mse_loss(current_q, target_q).item()
        policy_q = self.critic(states, self.actor(states)).mean().item()

        m = {
            "replay_size": len(self.replay_buffer),
            "batch_size": int(self.batch_size),
            "critic_updates": int(critic_updates),

            # —— Q/Reward 的统计 —— #
            "current_q_mean": q_current_mean,
            "current_q_std":  float(current_q.std(unbiased=False).item()),
            "current_q_min":  float(current_q.min().item()),
            "current_q_max":  float(current_q.max().item()),

            "target_q_mean":  q_target_mean,
            "bellman_gap":    float(bellman_gap),           # ★ 新增

            "reward_mean":    float(rewards.mean().item()),
            "reward_std":     float(rewards.std(unbiased=False).item()),
            "reward_min":     float(rewards.min().item()),
            "reward_max":     float(rewards.max().item()),

            "policy_q_mean":  float(policy_q),
            "td_mse":         float(td_mse),
            "td_error_mean":  float(td_error.mean().item()),# ★ 新增
            "td_error_std":   float(td_error.std().item()), # ★ 新增
            "done_ratio":     float(dones.float().mean().item()),
        }

        # --- 工具：全局梯度范数 ---
        def _grad_global_norm(module):
            tot = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    tot += p.grad.data.pow(2).sum().item()
            return tot ** 0.5

        # --- Critic 多次更新 ---
        avg_critic_loss = 0.0
        critic_grad_norms = []
        if any(p.requires_grad for p in self.critic.parameters()):
            total_critic_loss = 0.0
            for _ in range(critic_updates):
                with torch.no_grad():
                    next_a   = self.actor_target(next_states)
                    next_q   = self.critic_target(next_states, next_a).view(-1)
                    target_q = rewards + self.gamma * (1 - dones) * next_q

                current_q   = self.critic(states, actions).view(-1)
                critic_loss = F.mse_loss(current_q, target_q)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_grad_norms.append(_grad_global_norm(self.critic))
                self.critic_optimizer.step()
                self.soft_update(self.critic, self.critic_target)
                total_critic_loss += critic_loss.item()

            avg_critic_loss = total_critic_loss / critic_updates

        # --- Actor 更新 ---
        pred_a     = self.actor(states)
        actor_loss = - self.critic(states, pred_a).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = _grad_global_norm(self.actor)
        self.actor_optimizer.step()
        self.soft_update(self.actor, self.actor_target)

        # ★ Policy 输出稳定性（收敛检测 #1）
        actor_action_var = float(pred_a.var(dim=0).mean().item())
        if not hasattr(self, "_last_actions"):
            self._last_actions = pred_a.detach().clone()
            actor_action_diff = None
        else:
            actor_action_diff = float((pred_a - self._last_actions).pow(2).mean().item())
            self._last_actions = pred_a.detach().clone()

        # --- 缓存给外部（train_model 会读这些键） ---
        m.update({
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(avg_critic_loss),
            "actor_grad_norm": float(actor_grad_norm),
            "critic_grad_norm_mean": float((sum(critic_grad_norms) / max(1, len(critic_grad_norms))) if critic_grad_norms else 0.0),
            "actor_lr": float(self.actor_optimizer.param_groups[0]["lr"]),
            "critic_lr": float(self.critic_optimizer.param_groups[0]["lr"]),

            # ★ 新增：Policy 稳定性
            "actor_action_var":  float(actor_action_var),
            "actor_action_diff": float(actor_action_diff) if actor_action_diff is not None else None,

            # ★ 可选：Q 值采样（小样本）
            "q_values_sample": current_q[:100].detach().cpu().numpy().tolist(),
        })
        self.last_train_metrics = m

        return actor_loss.item(), avg_critic_loss




    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        action = self.actor(state)
        return action.squeeze(0)
               
class Sprinter(Supervisor):
    def __init__(self):
        super().__init__()

        self.timeStep = 40
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # —— 把 device 传给 DDPG —— 
        self.ddpg = DDPG(
            state_dim=STATE_DIM,
            action_dim=10,
            hidden_dim=512,
            actor_lr=ACTOR_LR,
            critic_lr=CRITIC_LR,
            device=self.device,        
            gamma=0.98,
        )
        # （此时 DDPG.__init__ 里已经完成 .to(self.device)）
    
    def set_global_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def initialize(self):
        self.timeStep = 40
        self.RShoulderPitch = self.getDevice('RShoulderPitch')
        self.LShoulderPitch = self.getDevice('LShoulderPitch')
        self.RShoulderPitch.setPosition(1.1)
        self.LShoulderPitch.setPosition(1.1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # —— 关节设备（Webots 名称） ——
        self.joint_names = [
            'RHipYawPitch', 'LHipYawPitch',
            'RHipRoll', 'LHipRoll',
            'RHipPitch', 'LHipPitch',
            'RKneePitch', 'LKneePitch',
            'RAnklePitch', 'LAnklePitch',
            'RAnkleRoll', 'LAnkleRoll'
        ]
        # 字典，便于查找
        self.motors = {name: self.getDevice(name) for name in self.joint_names}
        # 你 apply_action 的动作顺序（务必与 Actor 输出一致）
        self.motor_order = [
            'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll',
            'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll'
        ]
        # 顺序列表，供 capture/hard_reset 使用
        self.motors_list = [self.motors[name] for name in self.motor_order]

        # 位置传感器
        self.sensors = {}
        for name in self.joint_names:
            sensor_name = name + "S"
            sensor = self.getDevice(sensor_name)
            sensor.enable(self.timeStep)
            self.sensors[sensor_name] = sensor
            setattr(self, sensor_name.replace("S", "Sensor"), sensor)

        # IMU / 加速度计 / 陀螺
        self.accelerometer = self.getDevice("accelerometer")
        self.imu = self.getDevice("inertial unit")
        self.gyro = self.getDevice("gyro")
        self.accelerometer.enable(self.timeStep)
        self.imu.enable(self.timeStep)
        self.gyro.enable(self.timeStep)

        # 预热几步
        for _ in range(5):
            self.step(self.timeStep)

        # 世界节点
        self.nao = self.getFromDef("NAO")
        self.translation = self.nao.getField("translation")
        self.rotation    = self.nao.getField("rotation")

        # —— 重要：设置 supervisor 引用，供 resetPhysics 使用 ——
        self.supervisor = self  # Sprinter 继承自 Supervisor，可直接用

        # 拍初态（位姿 + 关节角）
        self.capture_initial_state()

        # 一些缓存（与 hard_reset 一致）
        self.pre_position   = self.get_translation()
        self.start_position = self.get_translation()
        self.current_position = self.get_translation()
        self.pre_yaw   = self.get_yaw()
        self.initYaw   = self.get_yaw()
        self.start_yaw = self.get_yaw()
        self.current_yaw = self.get_yaw()
        self.steps = []

        self.columns = [
            "phase", "roll", "yaw", "gyro_x", "gyro_y", "cm_norm",
            "RHipYawPitch","LHipYawPitch","RHipRoll","LHipRoll",
            "RHipPitch","LHipPitch","RKneePitch","LKneePitch",
            "RAnklePitch","LAnklePitch","RAnkleRoll","LAnkleRoll"
        ]

        print("[init] motors_list size =", len(self.motors_list))



    def summarize_eval_results(results):
        success = [0 if r["fell"] else 1 for r in results]
        completion_rate = sum(success) / len(success)

        # 只统计成功 episode 的时间/速度
        valid_time = [r["reach_time"] for r in results if (not r["fell"]) and (r["reach_time"] is not None)]
        valid_lateral = [abs(r["lateral"]) for r in results]

        summary = {
            "num_runs": len(results),
            "completion_rate": completion_rate,
            "time_mean": float(np.mean(valid_time)) if valid_time else None,
            "time_std": float(np.std(valid_time, ddof=1)) if len(valid_time) > 1 else 0.0,
            "lateral_mean": float(np.mean(valid_lateral)) if valid_lateral else None,
            "lateral_std": float(np.std(valid_lateral, ddof=1)) if len(valid_lateral) > 1 else 0.0,
            "steps_mean": float(np.mean([r["steps"] for r in results])),
            "steps_std": float(np.std([r["steps"] for r in results], ddof=1)) if len(results) > 1 else 0.0,
        }
        return summary    



    def print_joint_ranges(self):
        for name in self.joint_names:
            motor = self.getDevice(name)
            min_pos = motor.getMinPosition()
            max_pos = motor.getMaxPosition()
            print(f"{name}: Min = {min_pos:.3f}, Max = {max_pos:.3f}")
  

    def get_translation(self):
        return self.translation.getSFVec3f()
        
    def get_yaw(self):
        """获取 NAO 机器人绕 Z 轴的旋转角度（偏航角），直接通过 IMU 获取"""
        roll, pitch, yaw = self.imu.getRollPitchYaw()  # 获取 IMU 的 roll, pitch, yaw
        return yaw  # 返回 yaw 角（单位：度）
        
    def get_plane_movement(self, yaw_threshold=5.0):
        """
        计算 NAO 机器人在平面上的运动情况，并返回：
          - 'distance_moved'：本步沿世界 X+ 方向的净前进量
          - 'total_distance'：从起始位置沿 X 轴的累计净前进量
          - 'total_angle'：当前相对于 initYaw 的偏航角
          - 'angle_reward'：基于偏航变化的奖励
        """
        # 1) 读取当前位置与偏航
        self.current_position = self.get_translation()  # [x, y, z]
        self.current_yaw      = self.get_yaw()
    
        # 2) 本步平移向量
        dx = self.current_position[0] - self.pre_position[0]
        dy = self.current_position[1] - self.pre_position[1]
    
        # 3) 投影到世界 X+ 轴上的前进量 (cos 0 = 1, sin 0 = 0)
        distance_moved = dx # 简化为 dx
        

        # 4) 累计净前进量（从 start_position 投影到 X 轴）
        total_distance = self.current_position[0] - self.start_position[0]
    
        # 5) 计算偏航角奖励
        first_angle  = self.start_yaw - self.initYaw
        second_angle = self.current_yaw - self.initYaw
        total_angle  = second_angle
        angle_diff   = abs(abs(first_angle) - abs(second_angle))
        # 这里保留你原来的逻辑，也可以简化为 -k*|second_angle|
        if angle_diff < 0.001:
            angle_reward = 0.0
        else:
            # 偏离越大，奖励越低；收敛时奖励正值
            angle_reward = (abs(first_angle) - abs(second_angle)) * 100
    
        # 6) 更新 pre_position
        self.pre_position = list(self.current_position)
    
        return {
            'distance_moved': distance_moved,
            'total_distance': total_distance,
            'total_angle':    total_angle,
            'angle_reward':   angle_reward
        }
    
    
    def get_line_movement():
        return 

    def apply_action_to_robot(self, action):
        """将一个动作向量映射并应用到机器人各个关节"""
        joint_order = [
            'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll',
            'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll'
        ]
        for i, name in enumerate(joint_order):
            self.motors[name].setPosition(action[i].item())
    

    def get_rotation(self):
        return self.rotation.getSFRotation()  # [ax, ay, az, theta]

    def get_center_of_mass(self):
     
        return self.get_translation()[2]
    def get_init_position(self):
        RHipRoll_position = self.RHipRollSensor.getValue()
        LHipRoll_position = self.LHipRollSensor.getValue()
        RHipYawPitch_position = self.RHipYawPitchSensor.getValue()
        LHipYawPitch_position = self.LHipYawPitchSensor.getValue()
        RHipPitch_position = self.RHipPitchSensor.getValue()
        LHipPitch_position = self.LHipPitchSensor.getValue()
        RKneePitch_position = self.RKneePitchSensor.getValue()
        LKneePitch_position = self.LKneePitchSensor.getValue()
      
        RAnklePitch_position = self.RAnklePitchSensor.getValue()
        LAnklePitch_position = self.LAnklePitchSensor.getValue()
        RAnkleRoll_position = self.RAnkleRollSensor.getValue()
        LAnkleRoll_position = self.LAnkleRollSensor.getValue()
        init_postion=self.get_translation()
        return([RHipRoll_position, LHipRoll_position,RHipYawPitch_position,LHipYawPitch_position,RHipPitch_position,LHipPitch_position, RKneePitch_position,LKneePitch_position,
                RAnklePitch_position,LAnklePitch_position,RAnkleRoll_position , LAnkleRoll_position,init_postion])
        

    def get_current_state(self, step):
        # === 1. 姿态信息 ===
        roll, pitch, yaw1= self.imu.getRollPitchYaw()
        
        # 读取世界坐标系下的绝对朝向（axis–angle → yaw）
        x, y, z, theta = self.rotation.getSFRotation()
        qw = math.cos(theta / 2)
        sin_t2 = math.sin(theta / 2)
        qx = x * sin_t2;  qy = y * sin_t2;  qz = z * sin_t2
        yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        
        gyro_x, gyro_y, gyro_z = self.gyro.getValues()
        #print(gyro_z)value is nan

        # === 2. 步态相位 ===
        phase = (step % SEQUENCE_LENGTH) / SEQUENCE_LENGTH

        # === 3. 关节角度 ===
        joint_order = [
            'RHipYawPitch','LHipYawPitch',
            'RHipRoll',    'LHipRoll',
            'RHipPitch',   'LHipPitch',
            'RKneePitch',  'LKneePitch',
            'RAnklePitch', 'LAnklePitch',
            'RAnkleRoll',  'LAnkleRoll'
        ]
        joint_angles = [self.sensors[name+'S'].getValue() for name in joint_order]

        # === 4. 质心高度 cm 归一化 ===
        cm_raw = self.get_translation()[2]   # 例如 0.2–0.5 米
        CM_MIN, CM_MAX = 0.6, 0.72            # 根据实际机器人高度范围调整
        cm_norm = (cm_raw - CM_MIN) / (CM_MAX - CM_MIN)
        cm_norm = min(max(cm_norm, 0.0), 1.0)  # 截断到 [0,1]

        # === 5. 拼接状态向量 ===
        state = [
            phase,
            roll,
            yaw,
            gyro_x,
            gyro_y,
            cm_norm
        ] + joint_angles
        #print(state)
        # === 6. 记录日志 ===


        return state
        
     
    def calculate_reward(self, next_state, prev_state, action, prev_action,
                         angle_reward, dm,
                         break_flag, trunc, step_idx, base_reward):
        reward = 0.0
    
        # 基本量
        phase = next_state[0]
        step_ratio = (step_idx + 1) / float(trunc)
        roll, yaw = next_state[1:3]
        gyro = next_state[3:5]
        prev_roll, prev_yaw = prev_state[1:3]
    
        # —— 角度惩罚（不做归一化：直接用 yaw/10）——
        yaw_reward = abs(yaw / 10)
        #reward -= yaw_reward
    
        # —— 位移奖励（限幅到 ±0.2）——
        move_reward = dm * 10.0
        move_reward = max(min(0.5, move_reward), -0.5)
        reward += move_reward
    
        # —— 摔倒惩罚（随进度加重）——
        if break_flag:
            if step_ratio < 0.8:
                done_penalty = -0.1
            else:
                a = 10.0
                max_growth = math.exp(a * 0.2) - 1.0
                growth = math.exp(a * (step_ratio - 0.8)) - 1.0
                scaled = growth / max_growth if max_growth > 0 else 1.0
                done_penalty = -0.01 - (4.99 * scaled)
        else:
            done_penalty = 0.0
        reward += done_penalty
    
        # —— 平滑性惩罚（动作变化过大）——
        smoothness_threshold = 0.2
        action_diff = torch.sum((action - prev_action) ** 2).item()
    
        if step_idx == 0:
            if action_diff > 3.5:
                action_diff = 0.2
            else:
                action_diff = 0.0
        elif action_diff > smoothness_threshold:
            # 惩罚动作变化过大
            action_diff = action_diff
        else:
            action_diff = 0.0
    
        reward -= action_diff
    
        # —— 限制总奖励范围 —— 
        reward = max(min(reward, 3.0), -3.0)
        if reward == 3.0 or reward == -3.0:
            print("extrem value")
    
        # 可选：打印分解
        """
        print(f"[Reward Breakdown] total: {reward:.3f} | "
            f"yaw_reward: {yaw_reward:.3f} | "
            f"done_penalty: {done_penalty:.3f} | "
            f"move_reward: {move_reward:.3f} | "
            f"action_diff: {-action_diff:.3f}")
        #print("reward",reward)
        """
        return reward
    # 放在类里（Sprinter 内）
    def _set_default_motor_velocity(self, scale: float = 0.8):
        """把每个电机的速度设为最大速度的某个比例，确保 setPosition 能动起来。"""
        for m in getattr(self, "motors_list", []):
            try:
                vmax = m.getMaxVelocity()
            except Exception:
                vmax = 6.0  # 兜底
            m.setVelocity(float(vmax) * float(scale))

    def capture_initial_state(self):
        # 底座初态
        self.init_translation = self.get_translation()   # [x,y,z]
        self.init_rotation    = self.get_rotation()      # [ax, ay, az, theta]
        self.initYaw = float(self.get_yaw())  
        # 关节初始目标角（与 motor_order 一一对应）
        self.init_q = []
        for m in getattr(self, "motors_list", []):
            try:
                q0 = m.getTargetPosition()
            except Exception:
                q0 = 0.0
            self.init_q.append(q0)

        # RL 侧缓存
        self.prev_action    = torch.zeros_like(INITIAL_ACTIONS[0])
        self.pre_position   = self.init_translation
        self.start_position = self.init_translation

        print("[init] captured initial state: q0 len =", len(self.init_q))

    
    def hard_reset(self, settle_steps: int = 64, margin: float = 1e-3, use_supervisor: bool = True):
        """
        恢复到与初始尽可能一致的状态：姿态、关节、速度、控制器缓存，并冷却若干步。
        """
        # 1) 恢复姿态与关节角（如果你有 resetRobot(INITIAL_STATE)，先使用）
        try:
            self.resetRobot(INITIAL_STATE)
        except Exception:
            pass

        # 2) 关节命令回到初始角，速度归零并留边际
        for m, q0 in zip(getattr(self, "motors_list", []), getattr(self, "init_q", [])):
            try:
                lo = m.getMinPosition() + margin
                hi = m.getMaxPosition() - margin
            except Exception:
                lo, hi = -1e9, 1e9
            m.setVelocity(0.0)
            m.setPosition(min(max(q0, lo), hi))

        # 3) 清物理量（线/角速度、累积力），有 Supervisor 时更彻底
        if use_supervisor and getattr(self, "supervisor", None) is not None:
            try:
                node = self.supervisor.getSelf()
                if node is not None:
                    node.resetPhysics()
            except Exception:
                pass

        # 4) 冷却
        for _ in range(int(settle_steps)):
            self.step(self.timeStep)

        # 5) RL 缓存 + 朝向基线（★ 刷新 start_yaw）
        self.pre_position   = self.get_translation()
        self.start_position = self.get_translation()
        self.prev_action    = torch.zeros_like(INITIAL_ACTIONS[0])
        self.seq_phase_idx  = 0
        self.start_yaw      = float(self.get_yaw())   # ★

        # 6) 设定“工作速度”（★ 关键：把速度从 0 恢复）
        self._set_default_motor_velocity(scale=0.8)   # ★

        if hasattr(self.ddpg, "actor"):
            self.ddpg.actor.eval()



    
    def resetRobot(self, INITIAL_STATE):

        self.step(self.timeStep)

        joint_order = [
            'RHipRoll', 'LHipRoll',
            'RHipYawPitch', 'LHipYawPitch',
            'RHipPitch', 'LHipPitch',
            'RKneePitch', 'LKneePitch',
            'RAnklePitch', 'LAnklePitch',
            'RAnkleRoll', 'LAnkleRoll'
        ]
    
        for joint_name, angle in zip(joint_order, INITIAL_STATE[:12]):
            if joint_name in self.motors:
                self.motors[joint_name].setPosition(angle)
            else:
                print(f" Motor '{joint_name}' not found!")
    
        
        self.translation.setSFVec3f(INITIAL_STATE[12])  
        self.rotation.setSFRotation([0, 0, 1, 0])        
    
        
        for _ in range(64):
            self.step(self.timeStep)
    

                
    def train_model(self,num_episodes: int,
                    best_model_path: str = "check_points/best_ddpg_model.pth",
                    *,
                    log_dir: str = "runs/walk_exp1",
                    resume_logs: bool = True,
                    min_buffer_for_train: int = None,
                    train_every_episodes: int = 10,
                    max_total_falls: int = 15,
                    bc_reset_on_eval: bool = False,
                    seed: int =1):
    
        import os, json, time, heapq, numpy as np, torch, copy
        from pathlib import Path
        self.set_global_seed(random.seed)
        log_dir = f"runs/walk_exp1/seed_{seed}"
        if not resume_logs:
            rotate_metrics_csv(log_dir)
            self.global_step = 0
            self.ep_offset   = 0
            self.best_reward = float("-inf")
        _prepare_logging(controller, log_dir=log_dir, resume=False)
    
        # =============== 安全快照：保存“到达 4.7 之前”的上一份参数 ===============
        def _snapshot_ddpg(ddpg):
            return {
                "actor":           copy.deepcopy(ddpg.actor.state_dict()),
                "critic":          copy.deepcopy(ddpg.critic.state_dict()),
                "actor_target":    copy.deepcopy(ddpg.actor_target.state_dict()),
                "critic_target":   copy.deepcopy(ddpg.critic_target.state_dict()),
                "actor_opt":       copy.deepcopy(getattr(ddpg, "actor_optimizer", None).state_dict()) if hasattr(ddpg, "actor_optimizer") else None,
                "critic_opt":      copy.deepcopy(getattr(ddpg, "critic_optimizer", None).state_dict()) if hasattr(ddpg, "critic_optimizer") else None,
                "meta": {
                    "global_step": int(getattr(self, "global_step", 0)),
                    "ep_offset":   int(getattr(self, "ep_offset", 0)),
                    "saved_at":    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                }
            }
    
        last_safe_snapshot = _snapshot_ddpg(self.ddpg)   # 先放一份初始快照
        EARLY_CKPT_PATH = "check_points/ddpg_checkpoint.pth"  # 你要的保存路径
        # ========================================================================
        
        # 会话首条标记
        if bc_reset_on_eval:
            _append_metric(self, {
                "type": "bc_eval_reset",
                "episode": int(getattr(self, "ep_offset", 0)),
                "global_step": int(getattr(self, "global_step", 0)),
                "value": True,
                "note": "bc_reset_on_eval enabled at session start"
            })
    
        rb = self.ddpg.replay_buffer
        batch_size = getattr(self.ddpg, "batch_size", 64)
        if min_buffer_for_train is None:
            min_buffer_for_train = max(10 * batch_size, 512)
        self._buffer_ready = bool(getattr(self, "_buffer_ready", False))
    
        total_falls = 0
        updates_done = 0
        episodes_since_train = 0
        prev_action = torch.zeros_like(INITIAL_ACTIONS[0])
    
        window_return_sum = 0.0
        last_update_ep_idx = -1
        stop_training = False
    
        best_info_path = os.path.join(
            os.path.dirname(best_model_path),
            os.path.basename(best_model_path) + ".meta.json"
        )
        best_window_return_alltime = float("-inf")
        try:
            if os.path.exists(best_info_path):
                with open(best_info_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                best_window_return_alltime = float(meta.get("best_window_return_alltime", float("-inf")))
        except Exception as e:
            print(f"[best][meta-load-warning] {e}")
        self.hard_reset(settle_steps=64, use_supervisor=True)
        self.step(self.timeStep)
        for ep in range(num_episodes):
            # 每个 episode 开头，刷新安全快照（表示“进入本 ep 前的状态”）
            last_safe_snapshot = _snapshot_ddpg(self.ddpg)
    
            self.step(self.timeStep)
            self.start_position = self.get_translation()
            self.pre_position   = self.get_translation()
    
            state = torch.tensor(self.get_current_state(0), dtype=torch.float32, device=self.ddpg.device)
    
            fell = False
            seq_states, seq_actions, seq_next_states = [], [], []
            seq_angle_r, seq_dm, seq_preva = [], [], []
            pm = {"total_distance": 0.0, "angle_reward": 0.0, "distance_moved": 0.0}
            self.ddpg.actor.eval()
    
            for t in range(SEQUENCE_LENGTH):
                with torch.no_grad():
                    action = self.ddpg.actor(state.unsqueeze(0)).squeeze(0)
    
                self.apply_action_to_robot(action)
                self.step(self.timeStep)
    
                # —— 触发早停：到达 4.7 —— 
                try:
                    cur_x = float(self.get_translation()[0])
                except Exception:
                    cur_x = float(self.get_translation()[0])
                if cur_x > 4.7:
                    print(f"[early-stop] x={cur_x:.3f} > 4.7 @ episode={ep}, t={t}")
                    stop_training = True
                    pm = self.get_plane_movement()
    
                    # 保存“到达4.7之前”的模型（上一份安全快照）到指定 checkpoint 路径
                    Path(EARLY_CKPT_PATH).parent.mkdir(parents=True, exist_ok=True)
                    try:
                        torch.save(last_safe_snapshot, EARLY_CKPT_PATH)
                        print(f"[early-stop][saved] pre-4.7 model -> {EARLY_CKPT_PATH}")
                        _append_metric(self, {
                            "type": "pre_x4.7_saved",
                            "episode": int(self.ep_offset) + ep,
                            "global_step": int(getattr(self, "global_step", 0)),
                            "path": EARLY_CKPT_PATH,
                            "x_value": float(cur_x),
                        })
                    except Exception as e:
                        print(f"[early-stop][save-error] {e}")
                    break
    
                next_state = torch.tensor(self.get_current_state((t + 1) % SEQUENCE_LENGTH),
                                          dtype=torch.float32, device=self.ddpg.device)
    
                if self.get_center_of_mass() < 0.6:
                    total_falls += 1
                    fell = True
    
                pm = self.get_plane_movement()
                seq_states.append(state.cpu())
                seq_actions.append(action.detach().cpu())
                seq_next_states.append(next_state.cpu())
                seq_angle_r.append(pm["angle_reward"])
                seq_dm.append(pm["distance_moved"])
                seq_preva.append(prev_action)
    
                state = next_state
                prev_action = action.detach().clone()
    
                if fell:
                    self.hard_reset(settle_steps=64, use_supervisor=True)
                    self.step(self.timeStep)
                    break
    
            if total_falls >= max_total_falls:
                print(f"[stop] total_falls={total_falls} >= {max_total_falls}")
                break
    
            base_reward = pm["total_distance"]
            seq_len = len(seq_states)
            seq_dones = ([False] * max(0, seq_len - 1) + [True]) if (fell and seq_len > 0) else ([False] * seq_len)
    
            ep_total_reward = 0.0
            for i in range(seq_len):
                s  = seq_states[i].to(self.ddpg.device)
                a  = seq_actions[i].to(self.ddpg.device)
                ns = seq_next_states[i].to(self.ddpg.device)
                pa = seq_preva[i].to(self.ddpg.device)
    
                r  = self.calculate_reward(ns, s, a, pa,
                                           seq_angle_r[i], seq_dm[i],
                                           fell, seq_len, i, base_reward)
                ep_total_reward += float(r)
                self.ddpg.replay_buffer.push(s, a, r, ns, seq_dones[i])
    
            # 记录 episode
            episode_row = {
                "type": "episode",
                "episode": int(self.ep_offset) + ep,
                "global_step": int(getattr(self, "global_step", 0)),
                "reward": float(ep_total_reward),
                "fell": bool(fell),
                "seq_len": int(seq_len),
            }
            if bc_reset_on_eval and ep == 0:
                episode_row["bc_eval_reset"] = True
            _append_metric(self, episode_row)
    
            if stop_training:
                _append_metric(self, {
                    "type": "early_stop",
                    "episode": int(self.ep_offset) + ep,
                    "global_step": int(getattr(self, "global_step", 0)),
                    "reason": "x>4.7",
                    "x_value": float(cur_x),
                })
                print("[stop] early stop by x>4.7")
                break
    
            window_return_sum += float(ep_total_reward)
    
            if not self._buffer_ready and len(self.ddpg.replay_buffer) >= min_buffer_for_train:
                self._buffer_ready = True
                print(f"[buffer] ready now: len={len(self.ddpg.replay_buffer)}")
    
            episodes_since_train += 1
    
            if self._buffer_ready and (episodes_since_train >= train_every_episodes) and (len(self.ddpg.replay_buffer) > batch_size):
    
                # —— 历史 best 逻辑（原样保留）——
                if window_return_sum > best_window_return_alltime:
                    best_window_return_alltime = window_return_sum
                    Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
                    try:
                        save_checkpoint(host=controller.ddpg, filepath="check_points/best_ddpg_model.pth")
                        with open(best_info_path, "w", encoding="utf-8") as f:
                            json.dump({
                                "best_window_return_alltime": float(best_window_return_alltime),
                                "best_model_path": best_model_path,
                                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                "episode": int(self.ep_offset) + ep,
                                "global_step": int(getattr(self, "global_step", 0)),
                                "window_size": int(episodes_since_train),
                            }, f, ensure_ascii=False, indent=2)
    
                        print(f"[best][all-time] saved to {best_model_path} (window_sum={window_return_sum:.3f})")
                        _append_metric(self, {
                            "type": "best_model_alltime",
                            "episode": int(self.ep_offset) + ep,
                            "global_step": int(getattr(self, "global_step", 0)),
                            "window_return_sum": float(window_return_sum),
                            "window_size": int(episodes_since_train),
                            "path": best_model_path,
                        })
                    except Exception as e:
                        print(f"[best][save-error] {e}")
    
                # —— 一次参数更新 —— 
                window_return_sum = 0.0
                last_update_ep_idx = int(self.ep_offset) + ep
    
                a_loss, c_loss = self.ddpg.train()
                episodes_since_train = 0
                self.global_step = int(getattr(self, "global_step", 0)) + 1
    
                # ★ 更新后刷新安全快照（下一次如 early-stop，就保存这份）
                last_safe_snapshot = _snapshot_ddpg(self.ddpg)
    
                m = getattr(self.ddpg, "last_train_metrics", {}) or {}
                _append_metric(self, {
                    "type": "update",
                    "episode": int(self.ep_offset) + ep,
                    "global_step": int(self.global_step),
                    "actor_loss": float(a_loss),
                    "critic_loss": float(c_loss),
                    "buffer_len": int(len(self.ddpg.replay_buffer)),
                    "total_falls": int(total_falls),
    
                    "actor_action_var":  m.get("actor_action_var"),
                    "actor_action_diff": m.get("actor_action_diff"),
                    "td_error_mean":     m.get("td_error_mean"),
                    "td_error_std":      m.get("td_error_std"),
                    "td_mse":            m.get("td_mse"),
                    "q_current_mean":    m.get("current_q_mean"),
                    "q_target_mean":     m.get("target_q_mean"),
                    "bellman_gap":       m.get("bellman_gap"),
                    "actor_grad_norm":       m.get("actor_grad_norm"),
                    "critic_grad_norm_mean": m.get("critic_grad_norm_mean"),
                    "actor_lr":              m.get("actor_lr"),
                    "critic_lr":             m.get("critic_lr"),
                    "q_values": json.dumps(m.get("q_values_sample", [])),
                })
    
            if total_falls >= max_total_falls:
                print(f"[stop] total_falls={total_falls} >= {max_total_falls}")
                break
    
        self.ep_offset = int(getattr(self, "ep_offset", 0)) + (ep + 1)
        _save_meta(self)
        
    
    def run_trained_actor(self,
                        checkpoint_path: str = "check_points/best_ddpg_model.pth",
                        
                        num_episodes: int = 2,
                        stop_x: float = 4.5,
                        max_steps: int = 10000,
                        sequence_length: int = None):
        """
        评估已训练的 Actor。为了与训练一致，传给 get_current_state() 的 step 索引
        使用步态周期长度（默认取训练的 SEQUENCE_LENGTH 或 35）进行取模/夹紧。
        """

        print(f"Loading checkpoint from {checkpoint_path} …")

        load_checkpoint(host=self.ddpg, filepath=checkpoint_path)
        self.ddpg.actor.eval()

        # 与训练保持一致的步态序列长度
        seq_len = (sequence_length
                if sequence_length is not None
                else (getattr(self, "SEQUENCE_LENGTH", None)
                        or globals().get("SEQUENCE_LENGTH", None)
                        or 35))  # 兜底 35

        results = []
        dt = self.timeStep / 1000.0

        for ep in range(num_episodes):
            print(f"\n[Eval] Episode {ep + 1}/{num_episodes}")
            #self.resetRobot(INITIAL_STATE)
            self.hard_reset(settle_steps=64, use_supervisor=True)
            self.step(self.timeStep)
            start_pos = self.get_translation()
            start_x, start_y = start_pos[0], start_pos[1]

            # 初始状态严格用周期内索引 0
            raw = self.get_current_state(0)
            state = torch.tensor(raw, dtype=torch.float32, device=self.ddpg.device)

            fell = False
            reach_time = None
            step_counter = 0  # 累计步数（仿真步）

            for t in range(max_steps):
                with torch.no_grad():
                    action = self.ddpg.actor(state.unsqueeze(0)).squeeze(0)

                self.apply_action_to_robot(action)
                self.step(self.timeStep)
                step_counter += 1

                pos = self.get_translation()
                cur_x, cur_y = pos[0], pos[1]

                if stop_x is not None and cur_x >= stop_x:
                    reach_time = step_counter * dt
                    print(f"Reached x ≥ {stop_x:.3f} m at t={reach_time:.2f} s")
                    break


                raw_next = self.get_current_state(t+1)
                state = torch.tensor(raw_next, dtype=torch.float32, device=self.ddpg.device)

                if self.get_center_of_mass() < 0.6:
                    print("  Fell at step", step_counter)
                    fell = True
                    break

            end_pos = self.get_translation()
            forward_dist = end_pos[0] - start_x
            lateral_dist = end_pos[1] - start_y
            elapsed_time = step_counter * dt

            print(f"→ Episode {ep + 1}: forward={forward_dist:.3f} m, "
                f"lateral={lateral_dist:.3f} m, fell={fell}, "
                f"time={elapsed_time:.2f} s, steps={step_counter}, reach_time={reach_time}")

            results.append({
                "episode": ep + 1,
                "forward": forward_dist,
                "lateral": lateral_dist,
                "fell": fell,
                "time": elapsed_time,
                "steps": step_counter,
                "reach_time": reach_time
            })

        return results


    def run_walk_sequence_until_x(self,
                                   target_x: float = 4.5,
                                   gait_sequence = INITIAL_ACTIONS,
                                   max_loops: int = 200):

        print(f" Running fixed gait until reaching x = {target_x} m …")
        dt = self.timeStep / 1000.0  
    
        #self.resetRobot(INITIAL_STATE)
        self.hard_reset(settle_steps=64, use_supervisor=True)
        self.step(self.timeStep)
        start_pos = self.get_translation()
        start_x = start_pos[0]
    
        fell = False
        step_counter = 0
    
        for loop in range(max_loops):
            for t in range(len(gait_sequence)):
                action = gait_sequence[t]  # 已经是 tensor
                self.apply_action_to_robot(action)
                self.step(self.timeStep)
                step_counter += 1
    
                pos = self.get_translation()
                cur_x = pos[0]
    
                if self.get_center_of_mass() < 0.6:
                    print(f" Fell at step {step_counter}")
                    fell = True
                    break
    
                if cur_x >= target_x:
                    elapsed_time = step_counter * dt
                    print(f" Reached x = {target_x:.2f} m in {elapsed_time:.2f} s over {step_counter} steps")
                    return {
                        "reached": True,
                        "time": elapsed_time,
                        "steps": step_counter,
                        "forward": cur_x - start_x,
                        "fell": False
                    }
    
            if fell:
                break
    

        print(f" Did not reach target x = {target_x:.2f} m, final x = {cur_x:.2f} m")
        return {
            "reached": False,
            "time": step_counter * dt,
            "steps": step_counter,
            "forward": cur_x - start_x,
            "fell": True
        }
        

    




controller = Sprinter()  
controller.initialize()
ckpt_path = CKPTPATH
ckptl_path = CKPTLPATH


#load_actor_from_checkpoint( host=controller.ddpg)
"""
load_checkpoint( host=controller.ddpg)
controller.train_model(TRAIN_EPISODES,resume_logs=True,bc_reset_on_eval=False,seed=1)
save_checkpoint( host=controller.ddpg)
"""

"""
load_checkpoint( host=controller.ddpg)

load_bc(host=controller.ddpg)


controller.train_model(TRAIN_EPISODES,resume_logs=True,bc_reset_on_eval=True)
save_checkpoint( host=controller.ddpg)
"""


#controller.run_trained_actor(checkpoint_path = "check_points/best_ddpg_model.pth")
#controller.run_trained_actor(checkpoint_path = "check_points/ddpg_checkpoint.pth")

"""
load_checkpoint( host=controller.ddpg,filepath="check_points/ohneinitmodel")
controller.train_model(TRAIN_EPISODES,resume_logs=True,bc_reset_on_eval=False,log_dir="runs/walk_exp2",best_model_path="check_points/best_ddpg_model_woinit.pth")
save_checkpoint( host=controller.ddpg,filepath="check_points/ohneinitmodel")
"""
"""
load_checkpoint( ckpt_path)
#load_bc(host=self.ddpg)
controller.train_model(NUM_EPISODES,use_noise=False) 
save_checkpoint(host=self.ddpg,ckpt_path )
"""
#controller.train_model(NUM_EPISODES,use_noise=False,freeze_critic_steps=10) 

"""
results = controller.run_trained_actor(
    checkpoint_path="check_points/best_ddpg_model.pth",
    num_episodes=1,
    stop_x=4.5,          
    max_steps=6000        
)
"""
#controller.run_walk_sequence_until_x(target_x=4.5)
#checkpoint_path="ddpg_checkpoint.pth"
#checkpoint_path="best_ddpg_model.pth",

#controller.plot_ab()









