import argparse
import numpy as np
from stable_baselines3 import PPO
from src.envs.clinic_scheduling_env import ClinicSchedulingEnv
from src.utils.metrics import summarize, assignments_to_df
from src.baselines import greedy_schedule


def eval_once(model, env: ClinicSchedulingEnv):
    obs, _ = env.reset()
    done = False
    assignments_rl = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        assignments_rl = env.assigned
    sum_rl = summarize(assignments_rl, env.n_doctors, env.n_slots)

    # baseline sobre los mismos pacientes
    pts = env.patients # ya están ordenados
    assignments_bl, grid_bl = greedy_schedule(pts, env.n_doctors, env.n_slots)
    sum_bl = summarize(assignments_bl, env.n_doctors, env.n_slots)

    return sum_rl, sum_bl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()

    env = ClinicSchedulingEnv()
    model = PPO.load(args.model_path)

    rl_utils, bl_utils = [], []
    for _ in range(args.episodes):
        srl, sbl = eval_once(model, env)
        rl_utils.append(srl)
        bl_utils.append(srl)

    def avg(k, arr):
        return float(np.mean([a[k] for a in arr]))

    print("=== Promedios en", args.episodes, "episodios ===")
    print("Utilización RL:", round(avg("utilization", rl_utils)*100, 2), "%")
    print("Utilización Greedy:", round(avg("utilization", bl_utils)*100, 2), "%")
    print("Huecos >30min RL:", round(avg("large_gaps", rl_utils), 2))
    print("Huecos >30min Greedy:", round(avg("large_gaps", bl_utils), 2))
    print("Latencia urgencias RL (slots):", round(avg("avg_urgent_latency", rl_utils), 2))
    print("Latencia urgencias Greedy (slots):", round(avg("avg_urgent_latency", bl_utils), 2))


if __name__ == "__main__":
    main()