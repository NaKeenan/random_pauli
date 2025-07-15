using Pkg
Pkg.activate(".")

using ArgParse

"""
BATCH SIMULATION LAUNCHER FOR DIFFERENT M VALUES

This script launches multiple Pauli propagation simulations with different truncation
parameters M to study convergence behavior. It's designed to systematically explore
how the truncation parameter affects the accuracy of correlation functions.
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

function check_file_exists(save_dir, M, site, trial, N)
    """Check if output file already exists"""
    filename = joinpath(save_dir, "pauli_M$(M)_site$(site)_trial$(trial)_N$(N).jld2")
    return isfile(filename)
end

function run_single_simulation(N, num_steps, M, site, trial, save_dir, log_dir, calc_weight_dist, save_coeffs, manual, skip_existing)
    """Run a single simulation with given parameters"""
    
    # Check if file already exists
    if skip_existing && check_file_exists(save_dir, M, site, trial, N)
        println("SKIPPING M=$M: Output file already exists")
        return true
    end
    
    # Construct command
    cmd_args = [
        "julia", "run_pauli_trial.jl",
        "--trial", string(trial),
        "--site", string(site),
        "--M", string(M),
        "--num_steps", string(num_steps),
        "--N", string(N),
        "--save_dir", save_dir,
        "--log_dir", log_dir,
        "--calc_weight_dist", string(calc_weight_dist),
        "--save_coeffs", string(save_coeffs),
        "--manual", string(manual)
    ]
    
    println("Running simulation with M=$M...")
    println("Command: $(join(cmd_args, " "))")
    
    try
        # Run the command
        result = run(`$cmd_args`)
        if result.exitcode == 0
            println("SUCCESS: M=$M completed successfully")
            return true
        else
            println("ERROR: M=$M failed with exit code $(result.exitcode)")
            return false
        end
    catch e
        println("ERROR: M=$M failed with exception: $e")
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
    save_dir = args["save_dir"]
    log_dir = args["log_dir"]
    calc_weight_dist = args["calc_weight_dist"]
    save_coeffs = args["save_coeffs"]
    manual = args["manual"]
    parallel = args["parallel"]
    skip_existing = args["skip_existing"]
    
    # Print configuration
    println("Configuration:")
    println("  N = $N")
    println("  Trial = $trial")
    println("  Site = $site")
    println("  Time steps = $num_steps")
    println("  M values = $M_values")
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
    successful_runs = Int[]
    failed_runs = Int[]
    skipped_runs = Int[]
    
    # Run simulations
    println("Starting simulations...")
    start_time = time()
    
    if parallel
        println("Running in parallel mode...")
        # For parallel execution, we could use Distributed.jl, but for now we'll run sequentially
        # This is a placeholder for future parallel implementation
        println("Note: Parallel execution not yet implemented, running sequentially")
    end
    
    for M in M_values
        println("\n" * "="^50)
        println("Processing M = $M")
        println("="^50)
        
        # Check if already exists
        if skip_existing && check_file_exists(save_dir, M, site, trial, N)
            println("SKIPPING M=$M: Output file already exists")
            push!(skipped_runs, M)
            continue
        end
        
        # Run simulation
        success = run_single_simulation(N, num_steps, M, site, trial, save_dir, log_dir, calc_weight_dist, save_coeffs, manual, skip_existing)
        
        if success
            push!(successful_runs, M)
        else
            push!(failed_runs, M)
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
