def validate(vali_set, model):
    N_JOBS = vali_set[0][0].shape[0]
    N_MACHINES = vali_set[0][0].shape[1]

    import numpy as np
    import torch

    from l2d.config import configs
    from l2d.env.jssp import SJSSP
    from l2d.training.agent_utils import greedy_select_action
    from l2d.training.mb_agg import g_pool_cal
    env = SJSSP(num_jobs=N_JOBS, num_machines=N_MACHINES)
    device = torch.device(configs.device)
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size([1, env.num_operations, env.num_operations]),
                             n_nodes=env.num_operations,
                             device=device)
    make_spans = []
    # rollout using model
    for data in vali_set:
        reset_result = env.reset(data)
        adj, fea, candidate, mask = reset_result.adjacency_matrix, reset_result.features, reset_result.omega, reset_result.mask
        rewards = - env.init_quality
        while True:
            fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
            adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
            candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
            mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
            with torch.no_grad():
                pi, _ = model(x=fea_tensor,
                              graph_pool=g_pool_step,
                              padded_nei=None,
                              adj=adj_tensor,
                              candidate=candidate_tensor.unsqueeze(0),
                              mask=mask_tensor.unsqueeze(0))
            # action = sample_select_action(pi, candidate)
            action = greedy_select_action(pi, candidate)
            step_result = env.step(action.item())
            adj, fea, reward, done, candidate, mask = step_result.adjacency_matrix, step_result.features, step_result.reward, step_result.done, step_result.omega, step_result.mask
            rewards += reward
            if done:
                break
        make_spans.append(rewards - env.pos_rewards)
        # print(rewards - env.pos_rewards)
    return np.array(make_spans)


if __name__ == '__main__':

    import argparse

    import numpy as np

    from l2d.config import configs
    from l2d.env.uni_instance_gen import generate_uniform_times_and_machines_assignment

    parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')
    parser.add_argument('--Pn_j', type=int, default=20, help='Number of jobs of instances to test')
    parser.add_argument('--Pn_m', type=int, default=15, help='Number of machines instances to test')
    parser.add_argument('--Nn_j', type=int, default=20, help='Number of jobs on which to be loaded net are trained')
    parser.add_argument('--Nn_m', type=int, default=15, help='Number of machines on which to be loaded net are trained')
    parser.add_argument('--low', type=int, default=1, help='LB of duration')
    parser.add_argument('--high', type=int, default=99, help='UB of duration')
    parser.add_argument('--seed', type=int, default=200, help='Cap seed for validate set generation')
    parser.add_argument('--n_vali', type=int, default=100, help='validation set size')
    params = parser.parse_args()

    N_JOBS_P = params.Pn_j
    N_MACHINES_P = params.Pn_m
    LOW = params.low
    HIGH = params.high
    N_JOBS_N = params.Nn_j
    N_MACHINES_N = params.Nn_m

    import torch

    from l2d.training.ppo import PPO

    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_j=N_JOBS_P,
              n_m=N_MACHINES_P,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim,
              hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)

    path = './{}.pth'.format(str(N_JOBS_N) + '_' + str(N_MACHINES_N) + '_' + str(LOW) + '_' + str(HIGH))
    ppo.policy.load_state_dict(torch.load(path))

    SEEDs = range(0, params.seed, 10)
    result = []
    for SEED in SEEDs:

        np.random.seed(SEED)

        vali_data = [generate_uniform_times_and_machines_assignment(n_j=N_JOBS_P, n_m=N_MACHINES_P, low=LOW, high=HIGH) for _ in range(params.n_vali)]

        makespan = - validate(vali_data, ppo.policy)
        print(makespan.mean())


    # print(min(result))

