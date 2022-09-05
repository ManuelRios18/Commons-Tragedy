import utils
import matplotlib.pyplot as plt


path_dreamer = "logs/dreamer_multiagent_test/events.out.tfevents.1661978438.PCcasa.523154.0.v2"
path_ppo = "logs/dreamer_multiagent_test/progress_ppo.csv"

dreamer_metrics = utils.parse_dreamer_logs(path_dreamer, "scalars/train_return")
ppo_metrics = utils.parse_metrics(path_ppo)

dreamer_smoothed = utils.smooth(dreamer_metrics["metric"], 0.95)

plt.figure()
plt.plot(ppo_metrics["episodes"][:4000], ppo_metrics["efficiency"][:4000], label="PPO")
plt.plot(dreamer_metrics["episodes"], dreamer_smoothed, label="Dreamer-MA")
plt.axhline(y=20.87, color='black', linestyle='--', label="Random Agents")
plt.legend()
plt.grid()
plt.xlabel("Episode")
plt.ylabel("Efficiency")
plt.show()


"""
import os
import utils
import shutil
import matplotlib.pyplot as plt

experiment_name = "map_parolat_prob_meltingpot"
experiment_id = "A3C_meltingpot_d30c9_00000_0_2022-07-09_18-28-23"

experiment_path = os.path.join("logs", experiment_name, experiment_id)

metrics = utils.parse_metrics(os.path.join(experiment_path, "progress.csv"))

target_dir = os.path.join(experiment_path, "metrics")
if os.path.isdir(target_dir):
    shutil.rmtree(target_dir)
os.mkdir(target_dir)

plt.figure()
plt.plot(metrics["episodes"], metrics["efficiency"], color="turquoise")
plt.xlabel("Episodes")
plt.ylabel("Efficiency")
plt.title("Efficiency vs Episodes")
plt.grid()
plt.savefig(os.path.join(target_dir, "efficiency.pdf"))

plt.figure()
plt.plot(metrics["episodes"], metrics["peacefulness"], color="turquoise")
plt.xlabel("Episodes")
plt.ylabel("Peacefulness")
plt.title("Peacefulness vs Episodes")
plt.grid()
plt.savefig(os.path.join(target_dir, "peacefulness.pdf"))

plt.figure()
plt.plot(metrics["episodes"], metrics["equality"], color="turquoise")
plt.xlabel("Episodes")
plt.ylabel("Equality")
plt.title("Equality vs Episodes")
plt.grid()
plt.savefig(os.path.join(target_dir, "equality.pdf"))
"""