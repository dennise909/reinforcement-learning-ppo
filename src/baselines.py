import numpy as np


# Greedy: urgencias primero, asigna primer slot libre dentro de la disponibilidad


def greedy_schedule(patients, n_doctors, n_slots):
    grid = np.zeros((n_doctors, n_slots), dtype=int)
    assignments = []

    # Debug: Print original patient order
    print("🔍 Original patients:")
    for i, p in enumerate(patients):
        print(f"  Patient {i}: urgent={p['urgencia']}, avail_start={p['avail_start_slot']}")

    # ordenar: urgencias primero, luego por disponibilidad más temprana
    order = sorted(range(len(patients)), key=lambda i: (-(patients[i]["urgencia"]), patients[i]["avail_start_slot"]))
    
    # Debug: Print sorted order
    print("📋 Sorted order:")
    for i in order:
        p = patients[i]
        print(f"  Position {i}: Patient {p['paciente_id']}, urgent={p['urgencia']}, avail_start={p['avail_start_slot']}")

    for i in order:
        p = patients[i]
        placed = False
        for s in range(p["avail_start_slot"], p["avail_end_slot"] + 1):
            for d in range(n_doctors):
                if grid[d, s] == 0:
                    grid[d, s] = 1
                    assignments.append((p["paciente_id"], d, s, p["urgencia"], p["avail_start_slot"]))
                    placed = True
                    break
            if placed:
                break
        # si no cabe, se queda sin asignar
    
    # Debug: Print final assignments
    print("✅ Final assignments:")
    for ass in assignments:
        print(f"  Patient {ass[0]}, Doctor {ass[1]}, Slot {ass[2]}, Urgent: {ass[3]}")
    
    return assignments, grid