import matplotlib.pyplot as plt

particles = [10**2, 10**4, 10**6, 10**8]

# Configuration 1: Nx=250, Ny=100
time_config1_lab = [0.000097, 0.001010, 0.088943, 8.897167]
time_config1_hpc = [0.000218, 0.005806, 0.252421, 23.482544]

# Configuration 2: Nx=500, Ny=200
time_config2_lab = [0.000363, 0.001302, 0.094582, 9.172498]
time_config2_hpc = [0.000812, 0.009237, 0.340396, 31.923247]

# Configuration 3: Nx=1000, Ny=400
time_config3_lab = [0.001475, 0.002696, 0.131655, 12.512121] 
time_config3_hpc = [0.004496, 0.020666, 0.379751, 32.890718] 

def plot_scaling(particles, time_lab, time_hpc, config_title, filename):
    plt.figure(figsize=(8, 6))
    
    plt.plot(particles, time_lab, marker='o', linestyle='-', label='Lab PC', color='blue')
    plt.plot(particles, time_hpc, marker='s', linestyle='-', label='HPC Cluster', color='red')
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel('Number of Particles (log scale)', fontsize=12)
    plt.ylabel('Interpolation Execution Time [s] (log scale)', fontsize=12)
    plt.title(f'Experiment 01: Scaling with Particles\n({config_title})', fontsize=14)
    
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved plot: {filename}")
    plt.close()

plot_scaling(particles, time_config1_lab, time_config1_hpc, "Config 1: Nx=250, Ny=100", "Exp1_Config1.png")
plot_scaling(particles, time_config2_lab, time_config2_hpc, "Config 2: Nx=500, Ny=200", "Exp1_Config2.png")
plot_scaling(particles, time_config3_lab, time_config3_hpc, "Config 3: Nx=1000, Ny=400", "Exp1_Config3.png")

