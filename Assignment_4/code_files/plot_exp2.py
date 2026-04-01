import matplotlib.pyplot as plt
import numpy as np

problem_indices = [1, 2, 3]

time_lab_pc = [8.840356, 9.266865, 13.172626]  
time_hpc_cluster = [23.998892, 32.240073, 32.896622]  

def plot_consistency_bar(indices, lab_times, hpc_times, filename):
    plt.figure(figsize=(8, 6))
    
    bar_width = 0.35
    x = np.arange(len(indices))
    
    plt.bar(x - bar_width/2, lab_times, width=bar_width, label='Lab PC', color='blue', alpha=0.7)
    plt.bar(x + bar_width/2, hpc_times, width=bar_width, label='HPC Cluster', color='red', alpha=0.7)
    
    plt.xlabel('Problem Index', fontsize=12)
    plt.ylabel('Total Interpolation Time [s]', fontsize=12)
    plt.title('Experiment 02: Consistency Across Configurations\n(Fixed at 10^8 Particles)', fontsize=14)
    
    plt.xticks(x, [f'Config {i}' for i in indices], fontsize=11)
    
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved bar plot: {filename}")
    plt.close()

plot_consistency_bar(problem_indices, time_lab_pc, time_hpc_cluster, "Exp2_Consistency_BarPlot.png")
