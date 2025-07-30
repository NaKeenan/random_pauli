using Pkg
Pkg.activate(".")

using ArgParse

"""
BATCH PAULI PROPAGATION SIMULATION LAUNCHER

This script launches multiple Pauli propagation simulations with different truncation
parameters to study convergence behavior. It supports two truncation methods:

1. M-based truncation: Keep the M largest Pauli terms (ps.trim method)
2. Alpha-based pruning: Remove terms with |coefficient| < alpha (ps.prune method)

The script can run systematic parameter sweeps to analyze how different truncation
strategies affect the accuracy of correlation functions in U(1) symmetric quantum circuits.

Usage examples:
- M-based sweep: julia batch_pauli_simulation.jl --M_values "256,512,1024"
- Alpha-based sweep: julia batch_pauli_simulation.jl --use_alpha --alpha_values "1e-12,1e-10,1e-8"
"""

function parse_arguments()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--N"
            help = "Number of qubits"
            arg_type = Int
            default = 22
        "--trial"
            help = "Trial number"
            arg_type = Int
            default = 1
        "--site"
            help = "Initial site for Z operator"
            arg_type = Int
            default = 1
        "--num_steps"
            help = "Number of time steps"
            arg_type = Int
            default = 10
        "--M_values"
            help = "Comma-separated list of M values (e.g., '256,512,1024')"
            arg_type = String
            default = "256,512,1024,2048,4096"
        "--alpha_values"
            help = "Comma-separated list of alpha values (e.g., '1e-10,1e-12,1e-14')"
            arg_type = String
            default = ""
        "--use_alpha"
            help = "Use alpha pruning instead of M truncation"
            arg_type = Bool
            default = false
        "--save_dir"
            help = "Directory to save results"
            arg_type = String
            default = "pauli_results"
        "--log_dir"
            help = "Directory to save logs"
            arg_type = String
            default = "logs"
        "--calc_weight_dist"
            help = "Calculate weight distribution"
            arg_type = Bool
            default = true
        "--calc_entropy"
            help = "Calculate Pauli entropy"
            arg_type = Bool
            default = true
        "--save_coeffs"
            help = "Save full coefficients"
            arg_type = Bool
            default = true
        "--manual"
            help = "Use manual circuit parameters"
            arg_type = Bool
            default = true
        "--parallel"
            help = "Run simulations in parallel (if available)"
            arg_type = Bool
            default = false
        "--skip_existing"
            help = "Skip simulations if output file already exists"
            arg_type = Bool
            default = true
    end
    return parse_args(s)
end

function parse_M_values(M_string::String)
    """Parse comma-separated M values string into array of integers"""
    M_strings = split(M_string, ",")
    M_values = Int[]
    for M_str in M_strings
        M_str = strip(M_str)
        if !isempty(M_str)
            push!(M_values, parse(Int, M_str))
        end
    end
    return sort(M_values)
end

function parse_alpha_values(alpha_string::String)
    """Parse comma-separated alpha values string into array of floats"""
    if isempty(strip(alpha_string))
        return Float64[]
    end
    alpha_strings = split(alpha_string, ",")
    alpha_values = Float64[]
    for alpha_str in alpha_strings
        alpha_str = strip(alpha_str)
        if !isempty(alpha_str)
            push!(alpha_values, parse(Float64, alpha_str))
        end
    end
    return sort(alpha_values, rev=true)  # Sort from largest to smallest
end

function check_file_exists(save_dir, M, alpha, site, trial, num_steps, N)
    """Check if output file already exists"""
    if alpha !== nothing
        filename = joinpath(save_dir, "pauli_alpha$(alpha)_site$(site)_trial$(trial)_T$(num_steps)_N$(N).jld2")
    else
        filename = joinpath(save_dir, "pauli_M$(M)_site$(site)_trial$(trial)_T$(num_steps)_N$(N).jld2")
    end
    return isfile(filename)
end

function run_single_simulation(N, num_steps, M, alpha, site, trial, save_dir, log_dir, calc_weight_dist, calc_entropy, save_coeffs, manual, skip_existing)
    """Run a single simulation with given parameters"""
    
    # Check if file already exists
    if skip_existing && check_file_exists(save_dir, M, alpha, site, trial, num_steps, N)
        if alpha !== nothing
            println("SKIPPING alpha=$alpha: Output file already exists")
        else
            println("SKIPPING M=$M: Output file already exists")
        end
        return true
    end
    
    # Construct command
    cmd_args = [
        "julia", "run_pauli_trial.jl",
        "--trial", string(trial),
        "--site", string(site),
        "--num_steps", string(num_steps),
        "--N", string(N),
        "--save_dir", save_dir,
        "--log_dir", log_dir,
        "--calc_weight_dist", string(calc_weight_dist),
        "--calc_entropy", string(calc_entropy),
        "--save_coeffs", string(save_coeffs),
        "--manual", string(manual)
    ]
    
    # Add either M or alpha parameter
    if alpha !== nothing
        push!(cmd_args, "--alpha", string(alpha))
        println("Running simulation with alpha=$alpha...")
    else
        push!(cmd_args, "--M", string(M))
        println("Running simulation with M=$M...")
    end
    
    println("Command: $(join(cmd_args, " "))")
    
    try
        # Run the command
        result = run(`$cmd_args`)
        if result.exitcode == 0
            if alpha !== nothing
                println("SUCCESS: alpha=$alpha completed successfully")
            else
                println("SUCCESS: M=$M completed successfully")
            end
            return true
        else
            if alpha !== nothing
                println("ERROR: alpha=$alpha failed with exit code $(result.exitcode)")
            else
                println("ERROR: M=$M failed with exit code $(result.exitcode)")
            end
            return false
        end
    catch e
        if alpha !== nothing
            println("ERROR: alpha=$alpha failed with exception: $e")
        else
            println("ERROR: M=$M failed with exception: $e")
        end
        return false
    end
end

function main()
    println("=== BATCH SIMULATION LAUNCHER ===")
    
    # Parse arguments
    args = parse_arguments()
    
    # Extract parameters
    N = args["N"]
    trial = args["trial"]
    site = args["site"]
    num_steps = args["num_steps"]
    M_values = parse_M_values(args["M_values"])
    alpha_values = parse_alpha_values(args["alpha_values"])
    use_alpha = args["use_alpha"]
    save_dir = args["save_dir"]
    log_dir = args["log_dir"]
    calc_weight_dist = args["calc_weight_dist"]
    calc_entropy = args["calc_entropy"]
    save_coeffs = args["save_coeffs"]
    manual = args["manual"]
    parallel = args["parallel"]
    skip_existing = args["skip_existing"]
    
    # Determine which parameter array to use
    if use_alpha && !isempty(alpha_values)
        param_values = alpha_values
        param_type = "alpha"
    else
        param_values = M_values
        param_type = "M"
    end
    
    # Print configuration
    println("Configuration:")
    println("  N = $N")
    println("  Trial = $trial")
    println("  Site = $site")
    println("  Time steps = $num_steps")
    if param_type == "alpha"
        println("  Alpha values = $alpha_values")
        println("  Using alpha pruning method")
    else
        println("  M values = $M_values")
        println("  Using M truncation method")
    end
    println("  Save directory = $save_dir")
    println("  Log directory = $log_dir")
    println("  Manual mode = $manual")
    println("  Skip existing = $skip_existing")
    println("  Parallel = $parallel")
    println()
    
    # Create directories
    mkpath(save_dir)
    mkpath(log_dir)
    
    # Track results
    successful_runs = []
    failed_runs = []
    skipped_runs = []
    
    # Run simulations
    println("Starting simulations...")
    start_time = time()
    
    if parallel
        println("Running in parallel mode...")
        # For parallel execution, we could use Distributed.jl, but for now we'll run sequentially
        # This is a placeholder for future parallel implementation
        println("Note: Parallel execution not yet implemented, running sequentially")
    end
    
    for param_val in param_values
        println("\n" * "="^50)
        if param_type == "alpha"
            println("Processing alpha = $param_val")
            M = nothing
            alpha = param_val
        else
            println("Processing M = $param_val")
            M = param_val
            alpha = nothing
        end
        println("="^50)
        
        # Check if already exists
        if skip_existing && check_file_exists(save_dir, M, alpha, site, trial, num_steps, N)
            if param_type == "alpha"
                println("SKIPPING alpha=$param_val: Output file already exists")
            else
                println("SKIPPING M=$param_val: Output file already exists")
            end
            push!(skipped_runs, param_val)
            continue
        end
        
        # Run simulation
        success = run_single_simulation(N, num_steps, M, alpha, site, trial, save_dir, log_dir, calc_weight_dist, calc_entropy, save_coeffs, manual, skip_existing)
        
        if success
            push!(successful_runs, param_val)
        else
            push!(failed_runs, param_val)
        end
    end
    
    # Summary
    end_time = time()
    total_time = end_time - start_time
    
    println("\n" * "="^50)
    println("BATCH SIMULATION SUMMARY")
    println("="^50)
    println("Total time: $(round(total_time, digits=2)) seconds")
    println("Successful runs ($(length(successful_runs))): $successful_runs")
    println("Failed runs ($(length(failed_runs))): $failed_runs")
    println("Skipped runs ($(length(skipped_runs))): $skipped_runs")
    
    if length(failed_runs) > 0
        println("\nWARNING: Some simulations failed. Check log files for details.")
    end
    
    # Suggest next steps
    if length(successful_runs) > 1
        println("\nSuggested next steps:")
        println("1. Run comparison analysis:")
        println("   julia compare_M_values.jl")
        println("2. Plot results:")
        println("   julia plot_results.jl")
    end
    
    println("\nBatch simulation completed!")
end

# Run the main function
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
