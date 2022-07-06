from trainer.Trainer import Trainer


model_name = "large-model"
substrate_name = "commons_harvest_uniandes"
agent_algorithm = "A3C"
n_steps = 16000000
checkpoint_freq = 1
keep_checkpoints_num = 1
num_workers = 1
max_gpus = 1


trainer = Trainer(model_name=model_name, substrate_name=substrate_name, agent_algorithm=agent_algorithm,
                  n_steps=n_steps, checkpoint_freq=checkpoint_freq, keep_checkpoints_num=keep_checkpoints_num,
                  num_workers=num_workers, experiment_name=None, max_gpus=max_gpus)

results = trainer.start_training()
