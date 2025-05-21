# run_all_trials.jl

# Define your parameter arrays
Ns = [10, 12]
Ms = [4096, 8192]
sites = [1, 2, 3]
trials = 1:5
num_steps = 10

# Path to your run_pauli_trial script
run_script = "./run_pauli_trial.jl"

# Loop over all combinations and launch separate Julia processes
for N in Ns, M in Ms, site in sites, trial in trials
    cmd = `julia --project $run_script --N $N --num_steps $num_steps --M $M --site $site --trial $trial`
    println("Running command: ", string(cmd))
    run(cmd)
end
