import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from gymnasium.wrappers import FlattenObservation
from src.envs.clinic_scheduling_env import ClinicSchedulingEnv


def make_env(seed=None):
    def _thunk():
        env = ClinicSchedulingEnv(n_doctors=3, n_slots=32, n_patients=60, seed=seed)
        # MultiInputPolicy maneja Dict obs; FlattenObservation si prefieres aplanar
        return env
    return _thunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--save", type=str, default="models/ppo_clinic")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_random_seed(args.seed)

    env = DummyVecEnv([make_env(args.seed)])
    model = PPO("MultiInputPolicy", env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save)
    print("Modelo guardado en", args.save)


if __name__ == "__main__":
    main()