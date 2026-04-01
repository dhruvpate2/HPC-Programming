import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Define the expected file names
files = {
    "lab_serial": "exp3_serial_lab.csv",
    "lab_parallel": "exp3_parallel_lab.csv",
    "hpc_serial": "exp3_serial_hpc.csv",
    "hpc_parallel": "exp3_parallel_hpc.csv"
}

# 2. Check if all files exist before proceeding
for key, filename in files.items():
    if not os.path.exists(filename):
        print(f"Error: Could not find '{filename}'. Please ensure you have generated and renamed all 4 CSV files correctly.")
        exit(1)

# 3. Load the CSV data into Pandas DataFrames
df_lab_ser = pd.read_csv(files["lab_serial"])
df_lab_par = pd.read_csv(files["lab_parallel"])
df_hpc_ser = pd.read_csv(files["hpc_serial"])
df_hpc_par = pd.read_csv(files["hpc_parallel"])

# ==========================================
# PLOT 1: Iteration vs. Times 
# (Plotting the Parallel runs as the primary operational data)
# ==========================================
def plot_iteration_vs_time(df, title, filename):
    plt.figure(figsize=(8, 6))
    
    plt.plot(df['Iteration'], df['Interpolation_Time'], marker='o', linestyle='-', label='Interpolation Time', color='blue')
    plt.plot(df['Iteration'], df['Mover_Time'], marker='s', linestyle='-', label='Mover Time', color='green')
    plt.plot(df['Iteration'], df['Total_Time'], marker='^', linestyle='--', label='Total Time', color='red')
    
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Execution Time [s]', fontsize=12)
    plt.title(title, fontsize=14)
    
    plt.xticks(df['Iteration'])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved plot: {filename}")
    plt.close()

plot_iteration_vs_time(df_lab_par, 'Exp 03: Iteration vs Time (Lab PC - Parallel)', 'Exp3_Iter_vs_Time_Lab_Parallel.png')
plot_iteration_vs_time(df_hpc_par, 'Exp 03: Iteration vs Time (HPC Cluster - Parallel)', 'Exp3_Iter_vs_Time_HPC_Parallel.png')
plot_iteration_vs_time(df_lab_ser, 'Exp 03: Iteration vs Time (Lab PC - Serial)', 'Exp3_Iter_vs_Time_Lab_Serial.png')
plot_iteration_vs_time(df_hpc_ser, 'Exp 03: Iteration vs Time (HPC Cluster - Serial)', 'Exp3_Iter_vs_Time_HPC_Serial.png')

# ==========================================
# PLOT 2: Mover Serial vs. Parallel Execution Times
# ==========================================
# Calculate total sum of the Mover times for each configuration
t_mover_lab_ser = df_lab_ser['Mover_Time'].sum()
t_mover_lab_par = df_lab_par['Mover_Time'].sum()
t_mover_hpc_ser = df_hpc_ser['Mover_Time'].sum()
t_mover_hpc_par = df_hpc_par['Mover_Time'].sum()

plt.figure(figsize=(8, 6))
bar_width = 0.35
index = np.arange(2)

serial_times = [t_mover_lab_ser, t_mover_hpc_ser]
parallel_times = [t_mover_lab_par, t_mover_hpc_par]

plt.bar(index, serial_times, bar_width, label='Serial', color='orange', alpha=0.8)
plt.bar(index + bar_width, parallel_times, bar_width, label='Parallel (4 Threads)', color='teal', alpha=0.8)

plt.xlabel('Computing System', fontsize=12)
plt.ylabel('Total Mover Time [s]', fontsize=12)
plt.title('Mover Execution Time: Serial vs Parallel', fontsize=14)
plt.xticks(index + bar_width / 2, ['Lab PC', 'HPC Cluster'], fontsize=12)

plt.legend(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('Exp3_Mover_Comparison.png', dpi=300)
print(f"Saved plot: Exp3_Mover_Comparison.png")
plt.close()

# ==========================================
# PLOT 3: Parallel Speedup 
# ==========================================
speedup_lab = t_mover_lab_ser / t_mover_lab_par
speedup_hpc = t_mover_hpc_ser / t_mover_hpc_par

plt.figure(figsize=(6, 5))
systems = ['Lab PC', 'HPC Cluster']
speedups = [speedup_lab, speedup_hpc]

bars = plt.bar(systems, speedups, color=['cornflowerblue', 'indianred'], width=0.5)
plt.axhline(y=1.0, color='black', linestyle='--', label='Baseline (No Speedup)')

plt.ylabel('Speedup Factor (S = T_serial / T_parallel)', fontsize=12)
plt.title('Speedup of OpenMP Mover Implementation', fontsize=14)

# Add some headroom to the Y-axis for the text labels
plt.ylim(0, max(max(speedups) * 1.3, 2.0)) 
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

# Annotate bars with the exact speedup value
for bar, speedup in zip(bars, speedups):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{speedup:.2f}x", ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('Exp3_Speedup.png', dpi=300)
print(f"Saved plot: Exp3_Speedup.png")
plt.close()
