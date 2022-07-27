from trainer.trainer import Trainer


model_name = "large-model"
substrate_name = "commons_harvest_uniandes"
agent_algorithm = "A3C"
prob_type = "meltingpot"
map_name = "meltingpot"

n_steps = 20000
checkpoint_freq = 1
keep_checkpoints_num = 1
num_workers = 1
num_cpus_per_worker = 1
max_gpus = 1
independent_learners = True

substrate_config = {"prob_type": prob_type,
                    "map_name": map_name}

experiment_name = f"map_{map_name}_prob_{prob_type}_independent_{independent_learners}"


trainer = Trainer(model_name=model_name, substrate_name=substrate_name, agent_algorithm=agent_algorithm,
                  n_steps=n_steps, checkpoint_freq=checkpoint_freq, keep_checkpoints_num=keep_checkpoints_num,
                  num_workers=num_workers, num_cpus_per_worker=num_cpus_per_worker,
                  substrate_config=substrate_config, independent_learners=independent_learners,
                  experiment_name=experiment_name, max_gpus=max_gpus)

results = trainer.start_training()
