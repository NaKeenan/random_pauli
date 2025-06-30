# run_all_trials.jl

# Define your parameter arrays
Ns = [8]
Ms = [4^8]
sites = 1:8
trials = 1:100
num_steps = 10
calc_weight_dist = false
save_coeffs = true
manual = true
if manual
    save_dir = "pauli_results_manual"
else
    save_dir = "pauli_results"
end

# Path to your run_pauli_trial script
run_script = "./run_pauli_trial.jl"

# Loop over all combinations and launch separate Julia processes
for N in Ns, M in Ms, trial in trials, site in sites
    cmd = `julia --project $run_script --N $N --num_steps $num_steps --M $M --site $site --trial $trial --calc_weight_dist $calc_weight_dist --save_coeffs $save_coeffs --manual $manual --save_dir $save_dir`
    println("Running command: ", string(cmd))
    run(cmd)
end