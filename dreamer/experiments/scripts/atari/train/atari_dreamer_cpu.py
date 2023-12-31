import sys

# from rlpyt.agents.pg.atari import AtariLstmAgent
# from rlpyt.algos.pg.a2c import A2C
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo

# from rlpyt.experiments.configs.atari.pg.atari_lstm_a2c import configs
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.samplers.parallel.cpu.collectors import CpuWaitResetCollector
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.utils.launching.variant import load_variant, update_config
from rlpyt.utils.logging.context import logger_context

from dreamer.agents.atari_dreamer_agent import AtariDreamerAgent
from dreamer.algos.dreamer_algo import Dreamer
from dreamer.experiments.configs.atari.atari_dreamer import configs


def build_and_train(slot_affinity_code, log_dir, run_ID, config_key):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)

    sampler = CpuSampler(
        EnvCls=AtariEnv,
        env_kwargs=config["env"],
        CollectorCls=CpuWaitResetCollector,
        TrajInfoCls=AtariTrajInfo,
        **config["sampler"]
    )
    algo = Dreamer(optim_kwargs=config["optim"], **config["algo"])
    agent = AtariDreamerAgent(model_kwargs=config["model"], **config["agent"])

    runner = MinibatchRl(
        algo=algo, agent=agent, sampler=sampler, affinity=affinity, **config["runner"]
    )
    name = config["env"]["game"] + str(config["sampler"]["batch_T"])
    with logger_context(log_dir, run_ID, name, config):  # Might have to flatten config
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
