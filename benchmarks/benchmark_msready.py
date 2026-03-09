import os
import time
import psutil
import multiprocessing

# Set OMP_NUM_THREADS to use all cores, otherwise OpenMP might default to 1 in some conda envs
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())

from parallel_rdkit import msready_smiles_parallel, msready_inchi_inchikey_parallel

complex_smiles = [
    "CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
    "CC1=CC2=C(C=C1C)N(C=N2)C3C(C(C(O3)CO)OP(=O)(O)O)O",
    "C1=CC=C(C=C1)CC(C(=O)O)N",
    "CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2=C(CCCC2(C)C)C)C)C",
    "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)F)N4CCNCC4)C(=O)O",
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CN1CCC[C@H]1C2=CN=CC=C2",
    "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C",
]

def monitor_resources(target_pid, stop_event, cpu_list, mem_list):
    try:
        process = psutil.Process(target_pid)
        process.cpu_percent(interval=None) # Initialize
        while not stop_event.is_set():
            cpu_list.append(process.cpu_percent(interval=0.1))
            mem_list.append(process.memory_info().rss / 1024 / 1024) # MB
    except Exception as e:
        pass

def run_benchmark(n_smiles):
    print(f"\n==========================================")
    print(f"Running benchmark for {n_smiles} SMILES...")
    print(f"==========================================")
    
    smiles_list = (complex_smiles * ((n_smiles // len(complex_smiles)) + 1))[:n_smiles]
    
    for func_name, func in [
        ("msready_smiles_parallel", msready_smiles_parallel),
        ("msready_inchi_inchikey_parallel", msready_inchi_inchikey_parallel)
    ]:
        print(f"\nBenchmarking {func_name}...", flush=True)
        
        manager = multiprocessing.Manager()
        cpu_list = manager.list()
        mem_list = manager.list()
        stop_event = manager.Event()
        
        target_pid = os.getpid()
        monitor_proc = multiprocessing.Process(target=monitor_resources, args=(target_pid, stop_event, cpu_list, mem_list))
        monitor_proc.start()
        
        start_time = time.time()
        results = func(smiles_list)
        end_time = time.time()
        
        stop_event.set()
        monitor_proc.join()
        
        duration = end_time - start_time
        max_cpu = max(list(cpu_list)) if len(cpu_list) > 0 else 0
        max_mem = max(list(mem_list)) if len(mem_list) > 0 else 0
        num_cores = multiprocessing.cpu_count()
        
        print(f"  Time taken: {duration:.2f} seconds", flush=True)
        print(f"  Max CPU usage: {max_cpu:.1f}% (Total Cores: {num_cores}, Max Possible: {num_cores * 100}%)", flush=True)
        print(f"  Max Memory usage: {max_mem:.1f} MB", flush=True)
        
        if max_cpu > 150:
            print("  [PASS] CPU usage indicates parallel execution.", flush=True)
        elif n_smiles <= 1000 and max_cpu >= 0:
             print("  [INFO] Short run might not hit high CPU sample.", flush=True)
        else:
            print("  [WARN] CPU usage low. Ensure multi-threading is active.", flush=True)

if __name__ == '__main__':
    import sys
    # If arguments are passed, use them (for fast testing)
    if len(sys.argv) > 1:
        sizes = [int(x) for x in sys.argv[1:]]
    else:
        sizes = [100, 1000, 10000, 100000]
        
    for n in sizes:
        run_benchmark(n)

