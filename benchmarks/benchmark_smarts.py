import os
import time
import psutil
import multiprocessing
import pandas as pd
from pathlib import Path

# Set OMP_NUM_THREADS to use all cores
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())

from parallel_rdkit import screen_smarts

# SMARTS patterns for drug classification
SMARTS_PATTERNS = {
    "amphetamine": {
        "general": "[NH,NH2][CH1;R0]([CH3;R0,CH2CH3])[CH2;R0]c1ccccc1",
    },
    "aminoindane": {
        "general": "[CH2]1[CH]([NH,NH2])[CH2](c2ccccc21)",
    },
    "cathinone": {
        "general": "N[CH1;R0]([CH3;R0,CH2CH3,CH2CH2CH3])[C;R0](=O)c1ccccc1",
    },
    "cathines": {
        "general": "N[CH1;R0]([CH3;R0,CH2CH3,CH2CH2CH3])[C;R0]([OH])c1ccccc1",
    },
    "tryptamine": {
        "general": "c1ccc2c(c1)c(c[nH]2)[CH2;R0][CH2;R0][N;R0]",
    },
    "phenethylamine": {
        "general": "[NH,NH2][CH2;R0][CH2;R0]c1ccccc1",
    },
    "lysergamide": {
        "general": "CN(C)C(=O)[CH]1[CH2]N([CH]2[CH2]c3c[n]c4c3c(ccc4)C2=C1)[C;H2+]",
    },
    "phenmetrazine": {
        "common": "C1COC([H])(C([H,CH3,CH2CH3])N1)c2ccccc2",
        "phenmetrazol_like": "C1COC([OH])(C([H,CH3,CH2CH3])N1)c2ccccc2",
    },
    "piprazine": {
        "benzyl": "[CH2](N2[CH2][CH2]N[CH2][CH2]2)c1ccccc1",
        "phenyl": "c1ccccc1N2[CH2][CH2]N[CH2][CH2]2",
        "benzoyl": "C(=O)(N2[CH2][CH2]N[CH2][CH2]2)c1ccccc1",
    },
    "phenidate": {
        "common": "COC(=O)[CH](c1ccccc1)[CH]1[CH2][CH2][CH2][CH2][NH]1",
        "piparadol_like": "[OH,H]C(c1ccccc1)(c2ccccc2)[CH1]3[NH][CH2][CH2,CH2CH2][CH2]3",
    },
    "phencyclidine": {"general": "c1ccccc1C2(CCCCC2)N"},
    "orphine": {
        "common": "CC(c1ccccc1)N2CCC(CC2)N3c4ccccc4NC3=O",
        "spirorphine": "CC(c1ccccc1)N2CCC3(CC2)C(=O)NCN3c4ccccc4",
    },
    "nitazene": {"general": "CCN([H,C])[CH2][CH2]N1c2ccccc2N=C1[CH,CH2]c3ccccc3"},
    "fentanyl": {
        "common": "CC(N(c1ccccc1)[CH]2CCN([CH2,CH][CH2,CH](c3ccccc3))CC2)=O",
        "all_tails": "CC(N(c1ccccc1)[CH]2CCN([CH2,CH][CH2,CH])CC2)=O",
        "car_oct_remi": "CC(N(c1ccccc1)C([C])2CCN([CH2,CH][CH2,CH])CC2)=O",
        "benzyl": "CC(N(c1ccccc1)C2CCN([CH2,CH](c3ccccc3))CC2)=O",
    },
    "cannabinoid": {
        "classical": "CCCCCc1cc(c2c(c1)OC([CH]3[CH]2C=C(CC3)C)(C)C)",
        "non_classical": "CCCCCc1ccc(c(c1)O)[CH]2CCC[CH](C2)O",
        "aminoalkylindole": "CCCn3cc(cc3)C(=O)c1cccc2ccccc12",
    },
}

def get_all_smarts():
    """Extract all SMARTS patterns from the dictionary."""
    all_smarts = []
    for category, patterns in SMARTS_PATTERNS.items():
        for pattern_name, smarts in patterns.items():
            all_smarts.append(smarts)
    return all_smarts

def extract_smiles_from_dsstox(n_mols):
    """Extract SMILES from DSSTox CSV files."""
    data_dir = Path("/gpfs01/work/nircoh/parallel_rdkit/data/DSSTox_CCD_dump_12092025")
    smiles_list = []
    
    # Read files in order
    csv_files = sorted(data_dir.glob("DSSToxCCDdump*.csv"))
    
    for csv_file in csv_files:
        if len(smiles_list) >= n_mols:
            break
        
        try:
            df = pd.read_csv(csv_file)
            if 'SMILES' in df.columns:
                file_smiles = df['SMILES'].dropna().tolist()
                smiles_list.extend(file_smiles)
                print(f"  Loaded {len(file_smiles)} SMILES from {csv_file.name}")
        except Exception as e:
            print(f"  Warning: Could not read {csv_file.name}: {e}")
            continue
    
    return smiles_list[:n_mols]

def create_smiles_file(smiles_list, output_path):
    """Create a SMILES file from a list of SMILES strings."""
    with open(output_path, 'w') as f:
        for smiles in smiles_list:
            f.write(f"{smiles}\n")
    return output_path

def monitor_resources(target_pid, stop_event, cpu_list, mem_list):
    try:
        process = psutil.Process(target_pid)
        process.cpu_percent(interval=None)  # Initialize
        while not stop_event.is_set():
            cpu_list.append(process.cpu_percent(interval=0.1))
            mem_list.append(process.memory_info().rss / 1024 / 1024)  # MB
    except Exception as e:
        pass

def run_benchmark(n_mols, smarts_list):
    print(f"\n{'='*60}")
    print(f"Running SMARTS matching benchmark for {n_mols} molecules...")
    print(f"{'='*60}")
    print(f"Number of SMARTS patterns: {len(smarts_list)}")
    
    # Extract SMILES from DSSTox
    print(f"\nExtracting {n_mols} SMILES from DSSTox...")
    smiles_list = extract_smiles_from_dsstox(n_mols)
    
    if len(smiles_list) < n_mols:
        print(f"Warning: Only {len(smiles_list)} molecules available, replicating to reach {n_mols}")
        # Replicate to reach desired count
        smiles_list = (smiles_list * ((n_mols // len(smiles_list)) + 1))[:n_mols]
    
    # Create temporary SMILES file
    temp_file = f"/tmp/dsstox_smiles_{n_mols}.smi"
    create_smiles_file(smiles_list, temp_file)
    print(f"Created temporary SMILES file: {temp_file}")
    
    # Benchmark direct mode
    print(f"\nBenchmarking direct mode...", flush=True)
    
    manager = multiprocessing.Manager()
    cpu_list = manager.list()
    mem_list = manager.list()
    stop_event = manager.Event()
    
    target_pid = os.getpid()
    monitor_proc = multiprocessing.Process(target=monitor_resources, args=(target_pid, stop_event, cpu_list, mem_list))
    monitor_proc.start()
    
    start_time = time.time()
    result = screen_smarts(
        smarts_list=smarts_list,
        smiles_file=temp_file,
        mode="direct"
    )
    end_time = time.time()
    
    stop_event.set()
    monitor_proc.join()
    
    duration = end_time - start_time
    max_cpu = max(list(cpu_list)) if len(cpu_list) > 0 else 0
    max_mem = max(list(mem_list)) if len(mem_list) > 0 else 0
    num_cores = multiprocessing.cpu_count()
    
    print(f"\n  Results:")
    print(f"  --------")
    print(f"  Time taken: {duration:.2f} seconds", flush=True)
    print(f"  Max CPU usage: {max_cpu:.1f}% (Total Cores: {num_cores}, Max Possible: {num_cores * 100}%)", flush=True)
    print(f"  Max Memory usage: {max_mem:.1f} MB", flush=True)
    print(f"  Result shape: {result.shape}", flush=True)
    print(f"  Molecules matching at least one pattern: {result.any(axis=1).sum()}", flush=True)
    
    if max_cpu > 150:
        print("  [PASS] CPU usage indicates parallel execution.", flush=True)
    elif n_mols <= 1000 and max_cpu >= 0:
        print("  [INFO] Short run might not hit high CPU sample.", flush=True)
    else:
        print("  [WARN] CPU usage low. Ensure multi-threading is active.", flush=True)
    
    # Cleanup
    os.remove(temp_file)
    
    return duration, max_cpu, max_mem

if __name__ == '__main__':
    import sys
    
    # Get all SMARTS patterns
    all_smarts = get_all_smarts()
    print(f"Total SMARTS patterns to test: {len(all_smarts)}")
    print("\nPatterns:")
    for i, smarts in enumerate(all_smarts, 1):
        print(f"  {i}. {smarts}")
    
    # If arguments are passed, use them (for fast testing)
    if len(sys.argv) > 1:
        sizes = [int(x) for x in sys.argv[1:]]
    else:
        sizes = [1000, 10000, 100000]
    
    results = []
    for n in sizes:
        duration, max_cpu, max_mem = run_benchmark(n, all_smarts)
        results.append({
            'n_molecules': n,
            'n_patterns': len(all_smarts),
            'duration_sec': duration,
            'max_cpu_percent': max_cpu,
            'max_memory_mb': max_mem
        })
    
    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Molecules':<12} {'Patterns':<10} {'Time (s)':<12} {'Max CPU %':<12} {'Max Mem (MB)':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r['n_molecules']:<12} {r['n_patterns']:<10} {r['duration_sec']:<12.2f} {r['max_cpu_percent']:<12.1f} {r['max_memory_mb']:<12.1f}")
