import argparse
import numpy as np
import pandas as pd
from pathlib import Path


SPECIALTIES = ["General", "Pediatría", "Cardiología"]
TYPES = ["General", "Revisión", "Urgencia"]


SLOT_MIN = 15
DAY_SLOTS = 32


rng = np.random.default_rng()


def minutes_to_hhmm(start_min, slot_idx):
    m = start_min + slot_idx * SLOT_MIN
    h = m // 60
    mm = m % 60
    return f"{int(h):02d}:{int(mm):02d}"    


def gen_doctors(n_doctors: int, day_start_min=9*60):
    docs = []
    for i in range(n_doctors):
        name = f"Dr. {chr(65+i)}"
        spec = rng.choice(SPECIALTIES)
        docs.append({
            "doctor_id": i,
            "doctor": name,
            "especialidad": spec,
            "day_start_min": day_start_min,
            "n_slots": DAY_SLOTS,
        })
    return pd.DataFrame(docs)


def gen_patients(n_patients: int):
    pts = []
    for i in range(n_patients):
        name = f"Paciente_{i:03d}"
        tipo = rng.choice(TYPES, p=[0.6, 0.3, 0.1])
        urg = 1 if tipo == "Urgencia" else 0
        # Ventana de disponibilidad
        win = rng.integers(low=8, high=DAY_SLOTS+1)
        start = rng.integers(low=0, high=max(1, DAY_SLOTS - win + 1))
        end = start + win - 1
        pts.append({
            "paciente_id": i,
            "paciente": name,
            "tipo": tipo,
            "urgencia": urg,
            "avail_start_slot": start,
            "avail_end_slot": end,
        })
    return pd.DataFrame(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_doctors", type=int, default=3)
    ap.add_argument("--n_patients", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="input")
    args = ap.parse_args()


    global rng
    rng = np.random.default_rng(args.seed)


    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)


    df_d = gen_doctors(args.n_doctors)
    df_p = gen_patients(args.n_patients)


    df_d.to_csv(outdir / "doctors.csv", index=False)
    df_p.to_csv(outdir / "patients.csv", index=False)


    print("Generados:", outdir / "doctors.csv", outdir / "patients.csv")


if __name__ == "__main__":
    main()