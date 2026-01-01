# NOTE: Env must be imported here in order to trigger metaclass registering

from .envs.videoagent.videoagent import VIDEOAGENT_ENV

from .parallel_env import agent_rollout_loop
