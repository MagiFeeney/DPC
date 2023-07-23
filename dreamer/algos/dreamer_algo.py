import math
import numpy as np
import torch
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F
from rlpyt.algos.base import RlAlgorithm
from rlpyt.replays.sequence.n_step import SamplesFromReplay
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.tensor import infer_leading_dims
from tqdm import tqdm

from dreamer.algos.replay import initialize_replay_buffer, samples_to_buffer
from dreamer.algos.planner import Planner
from dreamer.models.rnns import get_feat, get_dist
from dreamer.utils.logging import video_summary
from dreamer.utils.module import get_parameters, FreezeParameters

torch.autograd.set_detect_anomaly(True)  # used for debugging gradients

loss_info_fields = [
    "model_loss",
    "vae_loss",
    "elite_loss",
    "actor_loss",
    # "planning_loss",
    "prior_entropy",
    "post_entropy",
    "divergence",
    "reward_loss",
    "image_loss",
    "pcont_loss",
]
planning_info_fields = [
    "planning_elite_loss",
    "planning_actor_loss",
    # "planning_vae_loss",
    "grad_norm_actor_planning",
    "grad_norm_elite_planning",
    # "grad_norm_vae_planning",
]
LossInfo = namedarraytuple("LossInfo", loss_info_fields)
PlanningInfo = namedarraytuple("PlanningInfo", planning_info_fields)
OptInfo = namedarraytuple(
    "OptInfo",
    ["loss", "grad_norm_model", "grad_norm_actor", "grad_norm_vae", "grad_norm_elite"]
    + loss_info_fields + planning_info_fields
)


class Dreamer(RlAlgorithm):
    def __init__(
        self,  # Hyper-parameters
        batch_size=50,
        batch_length=50,
        train_every=1000,
        train_steps=100,
        pretrain=100,
        model_lr=6e-4,
        vae_lr=8e-5,
        actor_lr=8e-5,
        elite_lr=3e-4,
        grad_clip=100.0,
        dataset_balance=False,
        discount=0.99,
        discount_lambda=0.95,
        horizon=15,
        action_dist="beta",
        action_init_std=5.0,
        expl="additive_gaussian",
        expl_amount=0.3,
        expl_decay=0.0,
        expl_min=0.0,
        OptimCls=torch.optim.Adam,
        optim_kwargs=None,
        initial_optim_state_dict=None,
        replay_size=int(5e6),
        replay_ratio=8,
        n_step_return=1,
        updates_per_sync=1,  # For async mode only. (not implemented)
        free_nats=3,
        kl_scale=1,
        num_samples=1,
        type=torch.float,
        prefill=5000,
        log_video=False,
        video_every=int(1e1),
        video_summary_t=25,
        video_summary_b=4,
        use_pcont=False,
        pcont_scale=10.0,
        quantile_parameter=0.25,
    ):
        super().__init__()
        if optim_kwargs is None:
            optim_kwargs = {}
        self._batch_size = batch_size
        del batch_size  # Property.
        save__init__args(locals())
        self.update_counter = 0

        self.optimizer = None
        self.type = type

    def initialize(
        self, agent, n_itr, batch_spec, mid_batch_reset, examples, world_size=1, rank=0
    ):
        self.agent = agent
        self.n_itr = n_itr
        self.batch_spec = batch_spec
        self.mid_batch_reset = mid_batch_reset
        self.replay_buffer = initialize_replay_buffer(self, examples, batch_spec)
        self.optim_initialize(rank)
        self.quantile_planner = Planner(self.agent, self.discount, self.use_pcont)

    def async_initialize(
        self, agent, sampler_n_itr, batch_spec, mid_batch_reset, examples, world_size=1
    ):
        self.agent = agent
        self.n_itr = sampler_n_itr
        self.batch_spec = batch_spec
        self.mid_batch_reset = mid_batch_reset
        self.replay_buffer = initialize_replay_buffer(
            self, examples, batch_spec, async_=True
        )

    def optim_initialize(self, rank=0):
        self.rank = rank
        model = self.agent.model
        self.model_modules = [
            model.observation_encoder,
            model.observation_decoder,
            model.reward_model,
            model.representation,
            model.transition,
        ]
        if self.use_pcont:
            self.model_modules += [model.pcont]
        self.actor_modules = [model.action_decoder]
        self.post_modules = [model.posterior_model]
        self.value_modules = [model.value_model]
        self.elite_modules = [model.elite_model]
        
        self.model_optimizer = torch.optim.Adam(
            get_parameters(self.model_modules), lr=self.model_lr, **self.optim_kwargs
        )
        self.actor_optimizer = torch.optim.Adam(
            get_parameters(self.actor_modules), lr=self.actor_lr, **self.optim_kwargs
        )
        self.vae_optimizer = torch.optim.Adam(
            get_parameters(self.post_modules + self.value_modules), lr=self.vae_lr, **self.optim_kwargs
        )
        self.elite_optimizer = torch.optim.Adam(
            get_parameters(self.elite_modules), lr=self.elite_lr, **self.optim_kwargs
        )

        if self.initial_optim_state_dict is not None:
            self.load_optim_state_dict(self.initial_optim_state_dict)
        # must define these fields to for logging purposes. Used by runner.
        self.opt_info_fields = OptInfo._fields

    def optim_state_dict(self):
        """Return the optimizer state dict (e.g. Adam); overwrite if using
        multiple optimizers."""
        return dict(
            model_optimizer_dict=self.model_optimizer.state_dict(),
            actor_optimizer_dict=self.actor_optimizer.state_dict(),
            vae_optimizer_dict=self.vae_optimizer.state_dict(),
            elite_optimizer_dict=self.elite_optimizer.state_dict(),
        )

    def load_optim_state_dict(self, state_dict):
        """Load an optimizer state dict; should expect the format returned
        from ``optim_state_dict().``"""
        self.model_optimizer.load_state_dict(state_dict["model_optimizer_dict"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer_dict"])
        self.vae_optimizer.load_state_dict(state_dict["vae_optimizer_dict"])
        self.elite_optimizer.load_state_dict(state_dict["elite_optimizer_dict"])

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        itr = itr if sampler_itr is None else sampler_itr
        if samples is not None:
            # Note: discount not saved here
            self.replay_buffer.append_samples(samples_to_buffer(samples))

        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        if itr < self.prefill:
            return opt_info
        if itr % self.train_every != 0:
            return opt_info
        for i in tqdm(range(self.train_steps), desc="Imagination"):
            samples_from_replay = self.replay_buffer.sample_batch(
                self._batch_size, self.batch_length
            )
            buffed_samples = buffer_to(samples_from_replay, self.agent.device)
            model_loss, actor_loss, vae_loss, elite_loss, grad_norm_model, grad_norm_vae, grad_norm_elite, grad_norm_actor, loss_info = self.update(
                buffed_samples, itr, i
            )
            print("main update finished!")

            new_samples_from_replay = self.replay_buffer.sample_batch(
                5, 50,
            )
            
            new_samples = buffer_to(new_samples_from_replay, self.agent.device)
            planning_info = self.planning(new_samples)
            
            with torch.no_grad():
                loss = model_loss + actor_loss + vae_loss + elite_loss
            opt_info.loss.append(loss.item())
            if isinstance(grad_norm_model, torch.Tensor):
                opt_info.grad_norm_model.append(grad_norm_model.item())
                opt_info.grad_norm_actor.append(grad_norm_actor.item())
                opt_info.grad_norm_vae.append(grad_norm_vae.item())
                opt_info.grad_norm_elite.append(grad_norm_elite.item())
            else:
                opt_info.grad_norm_model.append(grad_norm_model)
                opt_info.grad_norm_actor.append(grad_norm_actor)
                opt_info.grad_norm_vae.append(grad_norm_vae)
                opt_info.grad_norm_elite.append(grad_norm_elite)
            for field in loss_info_fields:
                if hasattr(opt_info, field):
                    getattr(opt_info, field).append(getattr(loss_info, field).item())
            for field in planning_info_fields:
                if hasattr(opt_info, field):
                    getattr(opt_info, field).append(getattr(planning_info, field).item())

        return opt_info

    def update(self, samples: SamplesFromReplay, sample_itr: int, opt_itr: int):
        model = self.agent.model

        observation = samples.all_observation  # [t, t+batch_length+1]
        action = samples.all_action[1:]  # [t-1, t+batch_length] -> [t, t+batch_length]
        reward = samples.all_reward[1:]  # [t-1, t+batch_length] -> [t, t+batch_length]
        reward = reward.unsqueeze(2)
        done = samples.done
        done = done.unsqueeze(2)

        lead_dim, batch_t, batch_b, img_shape = infer_leading_dims(observation, 3)
        batch_t -= 1            # obs has an additional boostrap time step
        
        batch_size = batch_t * batch_b

        # normalize image
        observation = observation.type(self.type) / 255.0 - 0.5
        # embed the image
        embed = model.observation_encoder(observation)

        prev_state = model.representation.initial_state(
            batch_b, device=action.device, dtype=action.dtype
        )

        # Rollout model by taking the same series of actions as the real model
        prior, post = model.rollout.rollout_representation(
            batch_t, embed, action, prev_state
        )

        # update world model
        all_feat = get_feat(post)   # stoch + deter
        feat = all_feat[:-1]
        with torch.no_grad():
            next_feat = all_feat[1:]

        image_pred = model.observation_decoder(feat)
        reward_pred = model.reward_model(feat)
        reward_loss = -torch.mean(reward_pred.log_prob(reward))
        image_loss = -torch.mean(image_pred.log_prob(observation[:-1]))
        pcont_loss = torch.tensor(0.0)  # placeholder if use_pcont = False
        if self.use_pcont:
            pcont_pred = model.pcont(feat)
            pcont_target = self.discount * (1 - done.float())
            pcont_loss = -torch.mean(pcont_pred.log_prob(pcont_target))
        prior_dist = get_dist(prior)
        post_dist = get_dist(post)
        div = torch.mean(kl_divergence(post_dist, prior_dist))
        div = torch.max(div, div.new_full(div.size(), self.free_nats))
        model_loss = self.kl_scale * div + reward_loss + image_loss
        if self.use_pcont:
            model_loss += self.pcont_scale * pcont_loss

        self.model_optimizer.zero_grad()
        model_loss.backward()
        grad_norm_model = torch.nn.utils.clip_grad_norm_(
            get_parameters(self.model_modules), self.grad_clip
        )
        self.model_optimizer.step()

        # update posterior and value distribution
        with torch.no_grad():
            next_actions = model.action_decoder(next_feat).sample() # T x B x A
            next_U_dist = model.value_model(next_feat, next_actions)
            next_U_sample = next_U_dist.sample()
            target_U = reward + self.discount * (1 - done.float()) * next_U_sample
            
        feat = feat.detach()
        policy = model.action_decoder(feat) # T x B x A
        posterior = model.posterior_model(feat, target_U)
        kl_loss = kl_divergence(posterior, policy).sum(-1).mean()
        
        z = posterior.rsample((self.num_samples, )) # ns x T x B x A

        with torch.no_grad():
            repeated_feat = feat.unsqueeze(0).repeat(self.num_samples, 1, 1, 1)
        U_dist = model.value_model(repeated_feat, z)

        expanded_target_U = target_U.unsqueeze(0).repeat(self.num_samples, 1, 1, 1)
        reconstruction_loss = U_dist.log_prob(expanded_target_U).mean()

        vae_loss = self.kl_scale * kl_loss - reconstruction_loss
        
        self.vae_optimizer.zero_grad() # post + value dist
        vae_loss.backward()
        grad_norm_vae = torch.nn.utils.clip_grad_norm_(
            get_parameters(self.post_modules + self.value_modules), self.grad_clip
        )
        self.vae_optimizer.step()
        
        # Ranking distributions by the q-quantile
        new_action = model.elite_model(feat)
        U_dist = model.value_model(feat, new_action)
        Uq = U_dist.mean + U_dist.stddev * math.sqrt(2) * torch.erfinv(torch.FloatTensor([2 * self.quantile_parameter - 1]))

        elite_loss = -Uq.mean()
        
        self.elite_optimizer.zero_grad()
        elite_loss.backward()
        grad_norm_elite = torch.nn.utils.clip_grad_norm_(
            get_parameters(self.elite_modules), self.grad_clip
        )        
        self.elite_optimizer.step()

        # project policy to the posterior
        with torch.no_grad():
            elite_action = model.elite_model(feat)
            elite_U_dist = model.value_model(feat, elite_action)
            elite_Uq = elite_U_dist.mean + elite_U_dist.stddev * math.sqrt(2) * torch.erfinv(torch.FloatTensor([2 * self.quantile_parameter - 1]))
            print((elite_Uq > Uq).sum())
            posterior = model.posterior_model(feat, elite_Uq)
            
        policy = model.action_decoder(feat)
        actor_loss = kl_divergence(policy, posterior).sum(-1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        grad_norm_actor = torch.nn.utils.clip_grad_norm_(
            get_parameters(self.actor_modules), self.grad_clip
        )        
        self.actor_optimizer.step()

        # loss info
        with torch.no_grad():
            prior_ent = torch.mean(prior_dist.entropy())
            post_ent = torch.mean(post_dist.entropy())
            loss_info = LossInfo(
                model_loss,
                vae_loss,
                elite_loss,
                actor_loss,
                prior_ent,
                post_ent,
                div,
                reward_loss,
                image_loss,
                pcont_loss,
            )

            if self.log_video:
                if (
                    opt_itr == self.train_steps - 1
                    and sample_itr % self.video_every == 0
                ):

                    self.write_videos(
                        observation[:-1],
                        action,
                        image_pred,
                        post[:-1],
                        step=sample_itr,
                        n=self.video_summary_b,
                        t=self.video_summary_t,
                    )
                    print("write successful!")

        return model_loss, actor_loss, vae_loss, elite_loss, grad_norm_model, grad_norm_vae, grad_norm_elite, grad_norm_actor, loss_info

    
    def planning(self, samples: SamplesFromReplay):
        model = self.agent.model

        observation = samples.all_observation  # [t, t+batch_length+1]
        action = samples.all_action[1:]  # [t-1, t+batch_length] -> [t, t+batch_length]
        reward = samples.all_reward[1:]  # [t-1, t+batch_length] -> [t, t+batch_length]
        reward = reward.unsqueeze(2)
        done = samples.done
        done = done.unsqueeze(2)

        lead_dim, batch_t, batch_b, img_shape = infer_leading_dims(observation, 3)
        batch_t -= 1
        
        batch_size = batch_t * batch_b

        # normalize image
        observation = observation.type(self.type) / 255.0 - 0.5
        # embed the image
        embed = model.observation_encoder(observation)

        prev_state = model.representation.initial_state(
            batch_b, device=action.device, dtype=action.dtype
        )

        prior, post = model.rollout.rollout_representation(
            batch_t, embed, action, prev_state
        )

        feat = get_feat(post)[:-1].detach()   # stoch + deter
        
        # start planning
        planning_elite_action = self.quantile_planner.step(post)
        
        # identify the posterior 
        with torch.no_grad():
            elite_U_dist = model.value_model(feat, planning_elite_action)
            elite_Uq = elite_U_dist.mean + elite_U_dist.stddev * math.sqrt(2) * torch.erfinv(torch.FloatTensor([2 * self.quantile_parameter - 1]))
            posterior = model.posterior_model(feat, elite_Uq)
            
        # policy projection
        policy = model.action_decoder(feat)
        planning_actor_loss = kl_divergence(policy, posterior).sum(-1).mean()

        self.actor_optimizer.zero_grad()
        planning_actor_loss.backward()
        grad_norm_actor_planning = torch.nn.utils.clip_grad_norm_(
            get_parameters(self.actor_modules), self.grad_clip
        )        
        self.actor_optimizer.step()
        
        # update elite network in the end
        planning_elite_loss = F.mse_loss(planning_elite_action, model.elite_model(feat))

        self.elite_optimizer.zero_grad()
        planning_elite_loss.backward()
        grad_norm_elite_planning = torch.nn.utils.clip_grad_norm_(
            get_parameters(self.elite_modules), self.grad_clip
        )        
        self.elite_optimizer.step()

        with torch.no_grad():
            planning_info = PlanningInfo(
                planning_elite_loss,
                planning_actor_loss,
                grad_norm_actor_planning,
                grad_norm_elite_planning,
                # "planning_vae_loss",
                # "grad_norm_vae_planning",
            )

        return planning_info
            
    def write_videos(self, observation, action, image_pred, post, step=None, n=4, t=25):
        """
        observation shape T,N,C,H,W
        generates n rollouts with the model.
        For t time steps, observations are used to generate state representations.
        Then for time steps t+1:T, uses the state transition model.
        Outputs 3 different frames to video: ground truth, reconstruction, error
        """
        lead_dim, batch_t, batch_b, img_shape = infer_leading_dims(observation, 3)
        model = self.agent.model
        ground_truth = observation[:, :n] + 0.5
        reconstruction = image_pred.mean[:t, :n]

        prev_state = post[t - 1, :n]
        prior = model.rollout.rollout_transition(
            batch_t - t, action[t:, :n], prev_state
        )
        imagined = model.observation_decoder(get_feat(prior)).mean
        model = torch.cat((reconstruction, imagined), dim=0) + 0.5
        error = (model - ground_truth + 1) / 2
        # concatenate vertically on height dimension
        openl = torch.cat((ground_truth, model, error), dim=3)
        openl = openl.transpose(1, 0)  # N,T,C,H,W
        video_summary("videos/model_error", torch.clamp(openl, 0.0, 1.0), step)

    def compute_return(
        self,
        reward: torch.Tensor,
        value: torch.Tensor,
        discount: torch.Tensor,
        bootstrap: torch.Tensor,
        lambda_: float,
    ):
        """
        Compute the discounted reward for a batch of data.
        reward, value, and discount are all shape [horizon - 1, batch, 1] (last element is cut off)
        Bootstrap is [batch, 1]
        """
        next_values = torch.cat([value[1:], bootstrap[None]], 0)
        target = reward + discount * next_values * (1 - lambda_)
        timesteps = list(range(reward.shape[0] - 1, -1, -1))
        outputs = []
        accumulated_reward = bootstrap
        for t in timesteps:
            inp = target[t]
            discount_factor = discount[t]
            accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
            outputs.append(accumulated_reward)
        returns = torch.flip(torch.stack(outputs), [0])
        return returns

    def transform(self, sample):
        action = sample * (self.action_space_high - \
                           self.action_space_low) + self.action_space_low

        return action
    
