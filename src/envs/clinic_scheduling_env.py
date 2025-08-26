import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ClinicSchedulingEnv(gym.Env):
    """
    One-day scheduling environment (multi-day episode).
    Action space:
      - Discrete(n_doctors * n_slots + 1): (doctor, slot) or NO_OP at the last index.
    Observation:
      - "grid":    (n_doctors, n_slots) 0=free, 1=occupied
      - "patient": (n_slots,) availability window (0/1)
      - "day":     (1,) current day index in [0, max_days-1]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, n_doctors=3, n_slots=32, n_patients=60, seed=42, max_days=5, debug=False, external_patients=None):
        super().__init__()
        self.n_doctors = n_doctors
        self.n_slots = n_slots
        self.n_patients = n_patients
        self.seed = seed
        self.max_days = max_days
        self.debug = debug
        self.external_patients = external_patients

        # Constants / helpers
        self.NO_OP = self.n_doctors * self.n_slots  # last action index

        # Spaces
        self.action_space = spaces.Discrete(self.n_doctors * self.n_slots + 1)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0, 1, shape=(self.n_doctors, self.n_slots), dtype=np.int8),
            "patient": spaces.Box(0, 1, shape=(self.n_slots,), dtype=np.int8),
            "day": spaces.Box(0, self.max_days - 1, shape=(1,), dtype=np.int8),
        })

        # Episode state (set in reset)
        self.day_count = 0
        self.grid = None
        self.patients = None
        self.patient_idx = 0
        self.assigned = []
        self.invalid_tries = 0  # invalid attempts for current patient

        # Cross-day accumulators
        self.total_assigned = 0       # counts assignments across all days
        self.total_large_gaps = 0     # sum of day gaps across days

        self.reset(seed=self.seed)

    # ---------- Helpers ----------
    def _encode_action(self, d, s):
        return int(d) * self.n_slots + int(s)

    def _decode_action(self, a):
        d = a // self.n_slots
        s = a % self.n_slots
        return int(d), int(s)

    def _obs(self):
        return {
            "grid": self.grid.copy(),
            "patient": self._current_patient_vec(),
            "day": np.array([self.day_count], dtype=np.int8),
        }

    # ---------- Gym API ----------
    def reset(self, seed=None, options=None):
        # Let Gymnasium handle seeding and create self.np_random
        if seed is not None:
            super().reset(seed=seed)
        else:
            # Different seed per episode improves diversity
            super().reset(seed=np.random.randint(0, 1_000_000))

        self.day_count = 0
        self.grid = np.zeros((self.n_doctors, self.n_slots), dtype=np.int8)

        if self.external_patients is not None:
            self.patients = self.external_patients
        else:
            self.patients = self._sample_patients()
        self.patient_idx = 0
        self.assigned = []
        self.invalid_tries = 0

        # Reset accumulators
        self.total_assigned = 0
        self.total_large_gaps = 0

        return self._obs(), {}

    def step(self, action: int):
        terminated = False
        truncated = False
        info = {}
        reward = 0.0

        # Early end: no more patients for current day
        if self.patient_idx >= len(self.patients):
            # before ending/transition, account today's gaps once
            day_gaps = sum(self._large_gaps(self.grid[d]) for d in range(self.n_doctors))
            self.total_large_gaps += day_gaps

            if self.day_count < self.max_days - 1:
                # Move to next day
                self.day_count += 1
                self.grid = np.zeros((self.n_doctors, self.n_slots), dtype=np.int8)
                if self.external_patients is not None:
                    self.patients = self.external_patients
                else:
                    self.patients = self._sample_patients()
                self.patient_idx = 0
                self.assigned = []
                self.invalid_tries = 0

                reward += 50.0  # day completion bonus
                info.update({"day_completed": True, "new_day": self.day_count, "reason": "day_transition"})
                return self._obs(), reward, terminated, truncated, info
            else:
                # Episode ends
                terminated = True
                total_slots = self.n_doctors * self.n_slots * (self.day_count + 1)
                total_utilization = (self.total_assigned / total_slots) if total_slots > 0 else 0.0

                # Utilization bonuses
                if total_utilization > 0.3:
                    reward += 50.0
                if total_utilization > 0.5:
                    reward += 100.0
                if total_utilization > 0.7:
                    reward += 200.0

                # Gap penalty across all days
                reward += -0.5 * self.total_large_gaps

                # Multi-day completion bonus
                reward += self.day_count * 100.0

                info.update({
                    "reason": "max_days_reached",
                    "days_completed": self.day_count + 1,
                    "total_utilization": total_utilization,
                    "total_large_gaps": self.total_large_gaps,
                    "total_assigned": self.total_assigned,
                    "final_reward": reward,
                })
                return self._obs(), reward, terminated, truncated, info

        # ---- Process current patient ----
        patient = self.patients[self.patient_idx]
        print(f"🔍 RL Step: Patient {patient['paciente_id']} (urgent={patient['urgencia']}, avail_start={patient['avail_start_slot']})")
        print(f"🔍 Action: {action} -> Doctor {action // self.n_slots}, Slot {action % self.n_slots}")

        info["current_patient"] = {
            "id": patient["paciente_id"],
            "urgent": patient["urgencia"],
            "avail_start": patient["avail_start_slot"],
            "avail_end": patient["avail_end_slot"],
        }

        # No-op
        if action == self.NO_OP:
            has_valid = self._exists_valid_slot(patient)
            if has_valid:
                reward += -10.0  # skipping when assignable
                info["no_op_reason"] = "valid_slots_exist"
            else:
                reward += -1.0   # nothing you can do anyway
                info["no_op_reason"] = "no_valid_slots"

            # move to next patient
            self.patient_idx += 1
            self.invalid_tries = 0
            return self._obs(), reward, terminated, truncated, info

        # Decode action
        d, s = self._decode_action(action)
        valid = self._is_valid_assignment(patient, d, s)
        info["assignment_attempt"] = {"doctor": d, "slot": s, "valid": bool(valid)}

        if valid:
            self.grid[d, s] = 1
            self.assigned.append((patient["paciente_id"], d, s, patient["urgencia"], patient["avail_start_slot"]))
            print(f"✅ RL Assignment: Patient {patient['paciente_id']} -> Doctor {d}, Slot {s}, Urgent: {patient['urgencia']}")
            self.total_assigned += 1

            # Base reward
            reward += 15.0

            # Urgency shaping
            if patient["urgencia"] == 1:
                reward += 25.0
                early_bonus = max(0.0, (self.n_slots - s) / self.n_slots) * 20.0
                reward += early_bonus
                # Penaliza latencia (llegar tarde dentro de su ventana)
                latency = max(0, s - patient["avail_start_slot"])
                reward += -0.5 * latency

            # Compactness shaping (doctor-level)
            gaps = self._large_gaps(self.grid[d])
            reward += -0.1 * gaps
            small_gaps = self._count_small_gaps(self.grid[d])
            reward += 1.0 * small_gaps

            # Utilization nudge (per doctor)
            doctor_utilization = self.grid[d].sum() / self.n_slots
            if doctor_utilization > 0.2:
                reward += 2.0

            # advance to next patient
            self.patient_idx += 1
            self.invalid_tries = 0
            info["assignment_success"] = True

        else:
            # Invalid action: penalize, do NOT advance immediately
            reward += -2.0
            self.invalid_tries += 1
            info["assignment_success"] = False

            # If agent keeps failing, skip patient to prevent loops
            if self.invalid_tries >= 3:
                reward += -3.0  # extra penalty for giving up
                self.patient_idx += 1
                self.invalid_tries = 0

        return self._obs(), reward, terminated, truncated, info

    # ---------- Validity & masks ----------
    def _is_valid_assignment(self, patient, d, s):
        if d < 0 or d >= self.n_doctors or s < 0 or s >= self.n_slots:
            return False
        if self.grid[d, s] == 1:
            return False
        # within patient window
        return (patient["avail_start_slot"] <= s <= patient["avail_end_slot"])

    def _exists_valid_slot(self, patient):
        s0, s1 = patient["avail_start_slot"], patient["avail_end_slot"]
        sub = self.grid[:, s0:s1 + 1]
        return np.any(sub == 0)

    def get_action_mask(self):
        """
        Boolean vector of length action_space.n with valid actions.
        - (d, s) valid if slot within window and free.
        - NO_OP always allowed.
        """
        n_actions = self.n_doctors * self.n_slots + 1
        mask = np.zeros(n_actions, dtype=bool)

        if self.patient_idx >= len(self.patients):
            mask[-1] = True
            return mask

        p = self.patients[self.patient_idx]
        s0, s1 = p["avail_start_slot"], p["avail_end_slot"]

        valid = np.zeros_like(self.grid, dtype=bool)
        valid[:, s0:s1 + 1] = (self.grid[:, s0:s1 + 1] == 0)

        mask[:-1] = valid.reshape(-1)
        mask[-1] = True
        return mask

    # ---------- Sampling & features ----------
    def _sample_patients(self):
        """Generate diverse patients using the env RNG (self.np_random)."""
        rng = getattr(self, "np_random", np.random)
        patients = []
        for i in range(self.n_patients):
            urgent = int(rng.choice(np.array([0, 1], dtype=np.int8), p=np.array([0.85, 0.15])))
            min_width = 4
            max_width = min(16, self.n_slots // 2)

            start = int(rng.integers(0, self.n_slots - min_width))
            width = int(rng.integers(min_width, max_width + 1))
            end = min(self.n_slots - 1, start + width - 1)

            if end >= self.n_slots:
                end = self.n_slots - 1
                start = max(0, end - width + 1)

            patients.append({
                "paciente_id": i,
                "urgencia": urgent,
                "avail_start_slot": start,
                "avail_end_slot": end,
            })

        if self.debug:
            print(f"Generated {len(patients)} patients; urgent={sum(p['urgencia'] for p in patients)}")
        return patients

    def _current_patient_vec(self):
        vec = np.zeros(self.n_slots, dtype=np.int8)
        if self.patient_idx >= len(self.patients):
            return vec
        p = self.patients[self.patient_idx]
        vec[p["avail_start_slot"]:p["avail_end_slot"] + 1] = 1
        return vec

    def _large_gaps(self, schedule):
        # Count excess zeros beyond 2 in any zero-run
        gaps = 0
        count = 0
        for slot in schedule:
            if slot == 0:
                count += 1
            else:
                if count > 2:
                    gaps += count - 2
                count = 0
        if count > 2:
            gaps += count - 2
        return gaps

    def _count_small_gaps(self, schedule):
        # Count zero-runs of length 1 or 2
        small_gaps = 0
        count = 0
        for slot in schedule:
            if slot == 0:
                count += 1
            else:
                if 1 <= count <= 2:
                    small_gaps += 1
                count = 0
        if 1 <= count <= 2:
            small_gaps += 1
        return small_gaps

    # ---------- Rendering ----------
    def render(self):
        print(self.grid)
