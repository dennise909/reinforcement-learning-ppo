# app.py
import streamlit as st
import numpy as np
import pandas as pd

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.clinic_scheduling_env import ClinicSchedulingEnv  # Debe tener external_patients= y get_action_mask()
from src.baselines import greedy_schedule
from src.utils.metrics import summarize, assignments_to_df

st.set_page_config(page_title="RL Clinic Scheduling", layout="wide")
st.title("Scheduling inteligente para clínica (RL + MaskablePPO)")

# ========================
# Helpers de calendario ⬇️
# ========================
SLOT_MIN = 15
DAY_START_MIN = 9 * 60  # 09:00

def hhmm(minutes: int) -> str:
    return f"{minutes//60:02d}:{minutes%60:02d}"

def slot_label(slot_idx: int) -> str:
    return hhmm(DAY_START_MIN + slot_idx * SLOT_MIN)

def make_calendar(assignments, n_doctors: int, n_slots: int) -> pd.DataFrame:
    """
    Crea una tabla estilo calendario (filas = tiempo, columnas = doctores)
    assignments: lista (pid, d, s, urgent, avail_start)
    """
    times = [slot_label(s) for s in range(n_slots)]
    cols = [f"Doctor_{d}" for d in range(n_doctors)]
    cal = pd.DataFrame("—", index=times, columns=cols)
    
    for assignment in assignments:
        # Handle both 4-tuple and 5-tuple formats
        if len(assignment) == 4:
            pid, d, s, urgent = assignment
        else:
            pid, d, s, urgent, _ = assignment
            
        d = int(d); s = int(s); urgent = int(urgent); pid = int(pid)
        row = slot_label(s)
        col = f"Doctor_{d}"
        if row in cal.index and col in cal.columns:
            marker = "🟥" if urgent == 1 else "🟩"
            cal.at[row, col] = f"{marker} Paciente_{pid}"
    return cal

def large_gaps(row: np.ndarray) -> int:
    gaps, run = 0, 0
    for v in row:
        if v == 0:
            run += 1
        else:
            if run > 2:
                gaps += (run - 2)
            run = 0
    if run > 2:
        gaps += (run - 2)
    return gaps

def build_grid_from_assignments(assignments, n_doctors, n_slots):
    g = np.zeros((n_doctors, n_slots), dtype=int)
    for (_, d, s, _urgent, _start) in assignments:
        g[int(d), int(s)] = 1
    return g

def mask_fn(env):
    # Acción válida = (doctor,slot) libre en la ventana del paciente; NO_OP siempre permitido
    return env.get_action_mask()
# ========================

with st.sidebar:
    st.header("Parámetros del día")
    n_doctors = st.slider("Doctores", 1, 6, 3)
    n_patients = st.slider("Pacientes (para entrenar)", 0, 120, 60, step=5)
    n_slots = 32
    seed = st.number_input("Seed", value=42)
    timesteps = st.number_input("Entrenamiento (timesteps)", min_value=1000, value=500_000, step=1000)
    data_src = st.radio("Fuente de PACIENTES para SIMULAR", ["CSV (input/patients.csv)", "Generar aquí"], index=0)
    st.caption("💡 Entrena con muestreo; SIMULA con un dataset fijo (CSV o generado) para comparar justo vs baseline.")

col1, col2 = st.columns(2)

if "model" not in st.session_state:
    st.session_state.model = None

# ========================
# Generar/mostrar data de ejemplo (opcional)
# ========================
if st.button("📊 Generar datos de ejemplo (solo vista)"):
    from src.data.generate_data import gen_doctors, gen_patients
    np.random.seed(int(seed))
    doctors_df = gen_doctors(n_doctors)
    patients_df = gen_patients(n_patients)
    st.subheader("📋 Doctores generados")
    st.dataframe(doctors_df, use_container_width=True)
    st.subheader("🏥 Pacientes generados")
    st.dataframe(patients_df, use_container_width=True)
    st.subheader("📈 Estadísticas")
    st.metric("Pacientes urgentes", int(patients_df['urgencia'].sum()))
    st.metric("Total pacientes", len(patients_df))
    st.metric("Total doctores", len(doctors_df))

# ========================
# ENTRENAMIENTO
# ========================
if st.button("Entrenar / Reentrenar MaskablePPO"):
    with st.spinner("x..."):
        try:
            # Sanity check rápido en un solo env
            env_test = ClinicSchedulingEnv(n_doctors=n_doctors, n_slots=n_slots, n_patients=n_patients, seed=int(seed))
            obs, _ = env_test.reset()
            st.info("🔍 Probando ambiente...")
            st.write(f"Observación inicial: Grid {obs['grid'].shape}, Patient vec {obs['patient'].shape}")
            total_reward = 0
            for _ in range(5):
                action = env_test.action_space.sample()
                obs, reward, terminated, truncated, info = env_test.step(action)
                total_reward += reward
                if terminated:
                    break
            st.write(f"Test: reward total (≤5 pasos): {total_reward:.2f}")
            st.write(f"Pacientes asignados (test): {len(env_test.assigned)}")
            del env_test

            # Entrenamiento con 1 env enmascarado (DummyVecEnv)
            def make_env():
                e = ClinicSchedulingEnv(n_doctors=n_doctors, n_slots=n_slots, n_patients=n_patients, seed=None)
                return ActionMasker(e, mask_fn)
            venv = DummyVecEnv([make_env])

            # Hiperparámetros
            n_steps = max(64, min(256, n_patients))
            batch_size = max(64, min(256, n_steps))
            st.info(f"📊 Config: n_steps={n_steps}, batch_size={batch_size}, n_epochs=10")

            model = MaskablePPO(
                "MultiInputPolicy",
                venv,
                verbose=1,
                seed=int(seed),
                learning_rate=1e-3,
                ent_coef=0.01,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                tensorboard_log=f"./logs/PPO_mask_{n_doctors}docs_{n_patients}patients"
            )

            # Progreso
            progress_bar = st.progress(0)
            status_text = st.empty()
            class ProgressCallback:
                def __init__(self, total_timesteps): self.total_timesteps = total_timesteps
                def __call__(self, locals, globals):
                    cur = locals['self'].num_timesteps
                    progress_bar.progress(min(cur / self.total_timesteps, 1.0))
                    status_text.text(f"Entrenando... {cur:,}/{self.total_timesteps:,} timesteps")
                    return True
            model.learn(total_timesteps=int(timesteps), callback=ProgressCallback(int(timesteps)))

            st.session_state.model = model
            st.success(f"Modelo enmascarado entrenado en {timesteps:,} timesteps ✔️")
            if hasattr(model, 'logger') and model.logger is not None:
                st.info("📊 Estadísticas de entrenamiento en logs/")

        except Exception as e:
            st.error(f"Error durante el entrenamiento: {str(e)}")
            st.write("Detalles del error:", e)

# ========================
# SIMULAR Y COMPARAR
# ========================
def df_to_patients_list(df: pd.DataFrame):
    cols = ["paciente_id","urgencia","avail_start_slot","avail_end_slot"]
    df = df[cols].copy()
    for c in cols: df[c] = df[c].astype(int)
    return df.to_dict("records")

if st.button("Simular día y comparar"):
    try:
        # 1) Cargar/Generar pacientes de hoy (dataset fijo para la simulación)
        if data_src.startswith("CSV"):
            try:
                patients_df = pd.read_csv("input/patients.csv")
            except FileNotFoundError:
                st.error("No encuentro input/patients.csv. Cambia a 'Generar aquí' o coloca el CSV en esa ruta.")
                st.stop()
        else:
            from src.data.generate_data import gen_patients
            patients_df = gen_patients(n_patients)

        patients_today = df_to_patients_list(patients_df)

        # 2) Crear env de evaluación con pacientes EXTERNOS fijos
        eval_env = ClinicSchedulingEnv(
            n_doctors=n_doctors,
            n_slots=n_slots,
            n_patients=len(patients_today),
            seed=int(seed),
            external_patients=patients_today,   # <-- clave
        )
        eval_env.max_days = 1
        env = ActionMasker(eval_env, mask_fn)
        obs, _ = env.reset()

        # 3) Modelo (quick o entrenado). Entrenar SIEMPRE en OTRO ENV.
        if st.session_state.model is None:
            st.warning("No hay modelo entrenado. Entrenaré uno rápido solo para la demo (en otro env).")
            def make_train_env():
                e = ClinicSchedulingEnv(n_doctors=n_doctors, n_slots=n_slots, n_patients=n_patients, seed=int(seed)+999)
                return ActionMasker(e, mask_fn)
            train_venv = DummyVecEnv([make_train_env])
            model = MaskablePPO("MultiInputPolicy", train_venv, verbose=0, seed=int(seed),
                                learning_rate=3e-4, ent_coef=0.01, n_steps=512, batch_size=256)
            model.learn(total_timesteps=20_000)
            del train_venv
        else:
            model = st.session_state.model

        # 🔄 Resetear SIEMPRE el env de evaluación después de cualquier entrenamiento
        obs, _ = env.reset()

        # 4) Rollout limpio
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))

        # ===== RL outputs =====
        base_env = env.unwrapped
        ass_rl = base_env.assigned
        sum_rl = summarize(ass_rl, base_env.n_doctors, base_env.n_slots)
        df_rl = assignments_to_df(ass_rl)

        # Chequeo de consistencia grid vs assignments
        G_from_ass = build_grid_from_assignments(ass_rl, base_env.n_doctors, base_env.n_slots)
        mismatch = int((G_from_ass != base_env.grid).sum())
        if mismatch != 0:
            st.error(f"⚠️ Inconsistencia: {mismatch} celdas difieren entre grid y assignments.")
        else:
            filled = int(base_env.grid.sum())
            capacity = base_env.n_doctors * base_env.n_slots
            lg_total = sum(large_gaps(base_env.grid[d]) for d in range(base_env.n_doctors))
            st.info(f"RL — ocupación directa: {filled}/{capacity} ({filled/capacity*100:.1f}%), huecos>30: {lg_total}")

        # ===== Baseline greedy con los MISMOS pacientes =====
        ass_bl, grid_bl = greedy_schedule(patients_today, base_env.n_doctors, base_env.n_slots)
        sum_bl = summarize(ass_bl, base_env.n_doctors, base_env.n_slots)
        df_bl = assignments_to_df(ass_bl)

        # ====== Visualización por columnas ======
        with col1:
            st.subheader("RL (MaskablePPO)")
            st.metric("Utilización", f"{sum_rl['utilization']*100:.1f}%")
            st.metric("Huecos >30min", f"{sum_rl['large_gaps']}")
            st.metric("Latencia urgencias (slots)", f"{sum_rl['avg_urgent_latency']:.2f}")
            if len(ass_rl) > 0:
                # 🔍 ADD DEBUG LINES HERE (using print for terminal output)
                print("🔍 Debug - RL assignments data type:", type(ass_rl))
                print("🔍 Debug - RL assignments length:", len(ass_rl))
                print("🔍 Debug - First 5 RL assignments:", ass_rl[:5])
                print("🔍 Debug - About to create RL calendar with:", len(ass_rl), "assignments")
                
                cal_rl = make_calendar(ass_rl, base_env.n_doctors, base_env.n_slots)
                st.subheader("Calendario — PPO (RL)")
                st.dataframe(cal_rl, use_container_width=True)

        with col2:
            st.subheader("Greedy (baseline)")
            st.metric("Utilización", f"{sum_bl['utilization']*100:.1f}%")
            st.metric("Huecos >30min", f"{sum_bl['large_gaps']}")
            st.metric("Latencia urgencias (slots)", f"{sum_bl['avg_urgent_latency']:.2f}")
            if len(ass_bl) > 0:
                cal_bl = make_calendar(ass_bl, base_env.n_doctors, base_env.n_slots)
                st.subheader("Calendario — Greedy (baseline)")
                st.dataframe(cal_bl, use_container_width=True)

        st.caption("Leyenda: 🟩 Normal • 🟥 Urgencia • — Libre")

        # ===== Diagnóstico del episodio =====
        with st.expander("Diagnóstico del episodio (RL)"):
            st.write("Primeras 10 assignments RL:", [tuple(x) for x in ass_rl[:10]])
            st.write("Grid RL (sumas por doctor):", base_env.grid.sum(axis=1).tolist())
            st.write("Gaps>30 por doctor:", [large_gaps(base_env.grid[d]) for d in range(base_env.n_doctors)])

    except Exception as e:
        st.error(f"Error durante la simulación: {str(e)}")
        st.write("Full error details:", e)

st.markdown("---")
with st.expander("Detalles y extensiones sugeridas"):
    st.markdown(
        """
        - **Acciones enmascaradas** (MaskablePPO + ActionMasker) → evita acciones inválidas.
        - **Duraciones variables** (p.ej., 2 slots) y **especialidades** por doctor.
        - **Penalidad por cambios** entre doctores / back-to-back sin buffer.
        - **No-shows** y **overbooking** probabilístico.
        - **Reward shaping**: penalizar latencia de urgencias y huecos grandes.
        - **Baseline ILP/CP-SAT** (OR-Tools) para óptimos exactos en instancias pequeñas.
        """
    )
