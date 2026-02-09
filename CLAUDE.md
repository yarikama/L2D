# L2D - Learning to Dispatch for Job Shop Scheduling via Deep RL

NeurIPS 2020 paper 的 PyTorch 實作。用 PPO 訓練 GNN-based policy 來解 Job Shop Scheduling Problem (JSSP)。

## Project Structure

```
l2d/                          # main package
├── config.py                 # argparse 超參數（device, env, network, PPO）
├── env/
│   ├── jssp.py               # 核心 Gym 環境 SJSSP（step/reset）
│   ├── get_machine_neighbors.py         # getActionNbghs() 機台前驅/後繼工序
│   ├── end_time_lb.py        # calEndTimeLB() 完工時間下界
│   ├── left_shift.py         # permissibleLeftShift() 可行左移排程
│   └── instance_gen.py       # generate_uniform_times_and_machines_assignment() 隨機 JSSP instance
├── models/
│   ├── actor_critic.py       # ActorCritic：GraphCNN + Actor + Critic + action masking
│   ├── graphcnn.py           # GraphCNN 圖卷積 message passing
│   └── mlp.py                # MLP / MLPActor / MLPCritic
└── training/
    ├── ppo.py                # Memory + PPO 訓練器 + main()
    ├── agent_utils.py        # select_action / greedy_select_action / eval_actions
    ├── validation.py         # validate() greedy rollout 評估
    └── mb_agg.py             # aggr_obs() / g_pool_cal() batch 聚合

scripts/                      # 進入點腳本
├── train.py                  # （未來可用，目前用 l2d-train）
├── test.py                   # 隨機 instance 測試
├── test_benchmark.py         # Taillard 等 benchmark 測試
├── env_lab.py                # 環境 sanity check
├── generate_data.py          # 預生成 instances
└── plot.py                   # training/validation 曲線繪圖

data/
├── benchmarks/               # benchmark numpy 資料
├── checkpoints/              # 訓練好的模型 .pth
└── generated/                # 預生成的隨機 instances
```

## Commands

```bash
uv sync                                       # 安裝 dependencies
uv run python -m l2d.training.ppo             # 訓練
uv run python scripts/test.py                 # 測試（隨機 instance）
uv run python scripts/test_benchmark.py       # 測試（benchmark）
uv run python scripts/generate_data.py        # 生成資料
```
