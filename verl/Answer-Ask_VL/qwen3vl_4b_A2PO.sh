# run on 8xH100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

cd /project/peilab/qjl/CODE/verl/
PROJECT_DIR="$(pwd)"
DATA_DIR=/project/peilab/qjl/CODE/DATA


nohup python /project/peilab/qjl/CODE/SERVER/async_sandbox_server.py > /project/peilab/qjl/CODE/SERVER/server.log 2>&1 &

export HF_HOME=/project/peilab/qjl/tmp
export TMPDIR=/project/peilab/qjl/tmp
export RAY_TMPDIR=/project/peilab/qjl/tmp


python3 -m verl.trainer.main_ppo \
    --config-path="/project/peilab/qjl/CODE/verl/examples/Answer-Ask_VL" \
    --config-name='a2agent_multiturn_a2po' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/project/peilab/qjl/MODEL/Qwen3-VL-4B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='answer-ask_async_rl' \
    trainer.experiment_name='qwen3-4b-a2po-sgl-multi-w-tool' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="/project/peilab/qjl/CODE/verl/examples/Answer-Ask_VL/a2agent_tool_config.yaml" \
    trainer.total_epochs=15 $@ 2>&1 | tee /project/peilab/qjl/CODE/verl/examples/Answer-Ask_VL/qwen3vl_4b_A2PO.log