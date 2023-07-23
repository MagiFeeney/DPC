import torch
import math
from torch.distributions.normal import Normal
from dreamer.models.rnns import RSSMState
from dreamer.models.rnns import stack_states, drop_last, repeat_first_dim, get_feat
from rlpyt.utils.buffer import buffer_method

class Planner(object):
    def __init__(self,
                 agent,
                 discount,
                 use_pcont,
                 iterations=10,
                 candidates=50,
                 N=20,
                 H=5,
                 K=10,
                 q=0.25):
        self.agent = agent
        self.discount = discount
        self.use_pcont = use_pcont
        
        self.model = agent.model
        self.action_shape = agent.model.action_shape
        self.device = agent.device
        
        self.iterations = iterations
        self.candidates = candidates
        self.N = N              # number of trajectories
        self.K = K              # number of top-K elite actions
        self.H = H              # finite horizon
        self.q = q              # quantile percentage

    def step(self, states: RSSMState):
        states = drop_last(states)
        feature_size = states.stoch.shape[:-1]
        self.T, self.B = feature_size[0], feature_size[1]
        self.A = self.action_shape
        mean, std = torch.zeros(feature_size + (self.action_shape, )).to(self.device), torch.ones(feature_size + (self.action_shape, )).to(self.device) # T x B x A
        dist = Normal(mean, std) # initial belief distribution
        for i in range(self.iterations):
            print(f"Planning iteration {i}")
            actions = dist.sample((self.candidates, )) # C x T x B x A
            trajectories = self.sampler(states, actions) # H x N x C x T x B
            returns = self.compute_returns(trajectories) # N x C x T x B
            sample_quantiles = torch.topk(returns, self.N, dim=0, largest=False).values[self.N - math.ceil((1 - self.q) * self.N)] # C x T x B
            i0 = torch.topk(sample_quantiles, self.K, dim=0).indices.unsqueeze(-1).expand (self.K, self.T, self.B, self.A)
            i1 = torch.arange(self.T).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand (self.K, self.T, self.B, self.A)
            i2 = torch.arange(self.B).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand (self.K, self.T, self.B, self.A)
            i3 = torch.arange(self.A).expand (self.K, self.T, self.B, self.A)
            topk_actions = actions[i0, i1, i2, i3] # K x T x B x A
            mean = topk_actions.mean(0)            # T x B x A
            std = topk_actions.std(0)
            dist = Normal(mean, std)
        print("Planning Done!")
        return mean

    def sampler(self, state: RSSMState, action: torch.Tensor):
        state = repeat_first_dim(state, self.candidates) # repeat the first dim
        state = repeat_first_dim(state, self.N) # repeat the first dim
        action = action.unsqueeze(0).repeat(self.N, 1, 1, 1, 1)
        self.F = action.shape[:-1] # F = N x C x T x B (For recover the shape later)
        flat_action = action.reshape(-1, action.size(-1))
        flat_state = buffer_method(state, "reshape", action.shape[:-1].numel(), -1)
        next_states = [buffer_method(flat_state, "detach")]
        actions = [flat_action]
        with torch.no_grad():
            for i in range(self.H):
                next_flat_state = self.model.transition(flat_action, flat_state)
                next_states.append(next_flat_state)
                flat_state = next_flat_state
                flat_action, _ = self.model.policy(flat_state)
                actions.append(flat_action)
        next_states = stack_states(next_states, dim=0)
        actions = torch.stack(actions, dim=0)
        return next_states, actions

    def compute_returns(self, *traj_info):
        states = traj_info[0][0]
        actions = traj_info[0][1]
        imag_feat = get_feat(states)

        with torch.no_grad():
            imag_reward = self.model.reward_model(imag_feat[:-1]).mean # H x F x 1
            bootstrap_U_dist = self.model.value_model(imag_feat[-1], actions[-1])
            bootstrap_U = bootstrap_U_dist.sample() # F x 1

        if self.use_pcont:
            discount_arr = self.model.pcont(imag_feat[:-1]).mean
        else:
            discount_arr = self.discount * torch.ones_like(imag_reward)

        returns = bootstrap_U
        for step in reversed(range(imag_reward.size(0))):
            returns = returns * discount_arr[step] + imag_reward[step]

        return returns.reshape(self.F, -1)          # N x C x T x B
