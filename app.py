import streamlit as st
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from src.envs.clinic_scheduling_env import ClinicSchedulingEnv
from src.baselines import greedy_schedule
from src.utils.metrics import summarize, assignments_to_df
from src.utils.visualize import timeline_figure


st.set_page_config(page_title="RL Clinic Scheduling", layout="wide")
st.title("Scheduling inteligente para clínica (RL + PPO)")

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
    assignments: lista de tu entorno (pid, d, s, urgent, avail_start)
    """
    times = [slot_label(s) for s in range(n_slots)]
    cal = pd.DataFrame("—", index=times, columns=[f"Doctor_{d}" for d in range(n_doctors)])
    for (pid, d, s, urgent, _avail_start) in assignments:
        marker = "🟥" if int(urgent) == 1 else "🟩"
        cal.loc[slot_label(int(s)), f"Doctor_{int(d)}"] = f"{marker} Paciente_{int(pid)}"
    return cal
# ========================


with st.sidebar:
    st.header("Parámetros del día")
    n_doctors = st.slider("Doctores", 1, 6, 3)
    n_patients = st.slider("Pacientes", 0, 120, 60, step=5)
    n_slots = 32
    seed = st.number_input("Seed", value=42)
    timesteps = st.number_input("Entrenamiento (timesteps)", min_value=1000, value=500000, step=1000)
    st.caption("💡 Más timesteps = mejor aprendizaje. Recomendado: 500K+ para buenos resultados.")

col1, col2 = st.columns(2)

if "model" not in st.session_state:
    st.session_state.model = None

# Botón para generar y mostrar datos de ejemplo
if st.button("📊 Generar datos de ejemplo"):
    from src.data.generate_data import gen_doctors, gen_patients
    
    # Generar datos con el seed actual
    np.random.seed(int(seed))
    doctors_df = gen_doctors(n_doctors)
    patients_df = gen_patients(n_patients)
    
    st.subheader("📋 Doctores generados")
    st.dataframe(doctors_df, use_container_width=True)
    
    st.subheader("🏥 Pacientes generados")
    st.dataframe(patients_df, use_container_width=True)
    
    # Mostrar estadísticas
    st.subheader("📈 Estadísticas")
    urgent_count = patients_df['urgencia'].sum()
    st.metric("Pacientes urgentes", int(urgent_count))
    st.metric("Total pacientes", len(patients_df))
    st.metric("Total doctores", len(doctors_df))

if st.button("Entrenar / Reentrenar PPO"):
    with st.spinner("Entrenando modelo PPO..."):
        try:
            env = ClinicSchedulingEnv(n_doctors=n_doctors, n_slots=n_slots, n_patients=n_patients, seed=int(seed))
            
            # Debug: Test environment first
            st.info("🔍 Probando ambiente...")
            obs, _ = env.reset()
            st.write(f"Observación inicial: Grid shape {obs['grid'].shape}, Patient vector length {obs['patient'].shape}")
            
            # Test a few steps
            total_reward = 0
            steps = 0
            for _ in range(5):
                action = env.action_space.sample()  # Random action
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                if terminated:
                    break
            
            st.write(f"Test: {steps} pasos, reward total: {total_reward:.2f}")
            st.write(f"Pacientes asignados: {len(env.assigned)}")
            
            # Mejor configuración de PPO para aprendizaje más efectivo
            # Ajustar parámetros basados en el tamaño del problema
            n_steps = max(32, min(128, n_patients // 2))  # Adaptar a número de pacientes
            batch_size = max(16, min(64, n_steps // 2))    # Adaptar al batch size
            
            st.info(f"📊 Configuración PPO: n_steps={n_steps}, batch_size={batch_size}, n_epochs={10}")
            st.info(f"📈 Esto creará aproximadamente {(timesteps // n_steps) * 10} épocas de aprendizaje")
            
            model = PPO(
                "MultiInputPolicy", 
                env, 
                verbose=1,  # Mostrar progreso de entrenamiento
                seed=int(seed),
                learning_rate=1e-3,      # Learning rate más alto para aprendizaje más rápido
                ent_coef=0.05,           # Más exploración
                n_steps=n_steps,         # Adaptado al número de pacientes
                batch_size=batch_size,   # Adaptado al batch size
                n_epochs=10,             # Más épocas por batch
                gamma=0.99,              # Factor de descuento
                gae_lambda=0.95,         # GAE lambda
                clip_range=0.2,          # Clipping del ratio de políticas
                tensorboard_log=f"./logs/PPO_{n_doctors}docs_{n_patients}patients"  # Logs con parámetros
            )
            
            # Barra de progreso para el entrenamiento
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Callback personalizado para monitorear el progreso
            class ProgressCallback:
                def __init__(self, total_timesteps):
                    self.total_timesteps = total_timesteps
                    self.current_step = 0
                    self.episode_count = 0
                    self.last_step = 0
                
                def __call__(self, locals, globals):
                    self.current_step = locals['self'].num_timesteps
                    
                    # Count episodes
                    if self.current_step > self.last_step:
                        self.episode_count += 1
                        self.last_step = self.current_step
                    
                    progress = min(self.current_step / self.total_timesteps, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Entrenando... {self.current_step:,}/{self.total_timesteps:,} timesteps | Episodios: {self.episode_count}")
                    return True
            
            callback = ProgressCallback(int(timesteps))
            
            # Entrenar con callback de progreso
            model.learn(total_timesteps=int(timesteps), callback=callback)
            
            st.session_state.model = model
            st.success(f"Modelo entrenado exitosamente en {timesteps:,} timesteps!")
            
            # Mostrar estadísticas del entrenamiento
            if hasattr(model, 'logger') and model.logger is not None:
                st.info("📊 Estadísticas de entrenamiento disponibles en logs/")
            
        except Exception as e:
            st.error(f"Error durante el entrenamiento: {str(e)}")
            st.write("Detalles del error:", e)

# Generar un día y evaluar
# Generar un día y evaluar
if st.button("Simular día y comparar"):
    try:
        # Creamos un env fijo para este día
        env = ClinicSchedulingEnv(n_doctors=n_doctors, n_slots=n_slots, n_patients=n_patients, seed=int(seed))
        env.max_days = 1  # ← Fuerza comparación de UN solo día
        obs, _ = env.reset()

        # Congelar el set de pacientes de hoy para que RL y baseline vean EXACTAMENTE lo mismo
        from copy import deepcopy
        patients_today = deepcopy(env.patients)

        # RL
        if st.session_state.model is None:
            st.warning("Primero entrena el modelo (sidebar). Usaré un modelo recién entrenado rápido.")
            model = PPO("MultiInputPolicy", env, verbose=1, seed=int(seed), learning_rate=3e-4, ent_coef=0.01, n_steps=1024, batch_size=256)
            model.learn(total_timesteps=20_000)
        else:
            model = st.session_state.model

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))

        ass_rl = env.assigned
        sum_rl = summarize(ass_rl, env.n_doctors, env.n_slots)
        df_rl = assignments_to_df(ass_rl)

        # Debug opcional
        st.write(f"RL assignments count: {len(ass_rl)}")
        st.write(f"RL dataframe shape: {df_rl.shape}")
        if len(df_rl) > 0:
            st.write("RL dataframe preview:")
            st.dataframe(df_rl.head(), use_container_width=True)

        # Baseline greedy sobre los MISMOS pacientes congelados
        ass_bl, grid_bl = greedy_schedule(patients_today, env.n_doctors, env.n_slots)
        sum_bl = summarize(ass_bl, env.n_doctors, env.n_slots)
        df_bl = assignments_to_df(ass_bl)

        # Debug opcional
        st.write(f"Greedy assignments count: {len(ass_bl)}")
        st.write(f"Greedy dataframe shape: {df_bl.shape}")
        if len(df_bl) > 0:
            st.write("Greedy dataframe preview:")
            st.dataframe(df_bl.head(), use_container_width=True)

        # ====== Visualización por columnas ======
        with col1:
            st.subheader("RL (PPO)")
            st.metric("Utilización", f"{sum_rl['utilization']*100:.1f}%")
            st.metric("Huecos >30min", f"{sum_rl['large_gaps']}")
            st.metric("Latencia urgencias (slots)", f"{sum_rl['avg_urgent_latency']:.2f}")
            if len(ass_rl) > 0:
                cal_rl = make_calendar(ass_rl, env.n_doctors, env.n_slots)
                st.subheader("Calendario — PPO (RL)")
                st.dataframe(cal_rl, use_container_width=True)

        with col2:
            st.subheader("Greedy (baseline)")
            st.metric("Utilización", f"{sum_bl['utilization']*100:.1f}%")
            st.metric("Huecos >30min", f"{sum_bl['large_gaps']}")
            st.metric("Latencia urgencias (slots)", f"{sum_bl['avg_urgent_latency']:.2f}")
            if len(ass_bl) > 0:
                cal_bl = make_calendar(ass_bl, env.n_doctors, env.n_slots)
                st.subheader("Calendario — Greedy (baseline)")
                st.dataframe(cal_bl, use_container_width=True)

        st.caption("Leyenda: 🟩 Normal • 🟥 Urgencia • — Libre")
        # ========================================

    except Exception as e:
        st.error(f"Error during simulation: {str(e)}")
        st.write("Full error details:", e)


st.markdown("---")
with st.expander("Detalles y extensiones sugeridas"):
    st.markdown(
        """
        - **Duraciones variables** (p.ej., 2 slots) y **especialidades** por doctor.
        - **Penalidad por cambios** entre doctores / back-to-back sin buffer.
        - **No-shows** y **overbooking** probabilístico.
        - **Reward shaping**: ponderar más la latencia de urgencias.
        - **Baseline ILP/CP-SAT** (OR-Tools) para un óptimo exacto en instancias pequeñas.
        - **RAG** o reglas clínicas para restricciones avanzadas.
        """
    )
