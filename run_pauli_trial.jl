using Pkg
Pkg.activate(".")

using ArgParse
include("utils.jl")  # Ensure path is correct

parsed_args = ArgParseSettings()
@add_arg_table parsed_args begin
    "--trial"
        arg_type = Int
    "--site"
        arg_type = Int
    "--M"
        arg_type = Int
    "--alpha"
        arg_type = Float64
    "--num_steps"
        arg_type = Int
    "--N"
        arg_type = Int
    "--save_dir"
        arg_type = String
        default = "pauli_results"
    "--log_dir"
        arg_type = String
        default = "logs"
    "--calc_weight_dist"
        arg_type = Bool
        default = true
    "--calc_entropy"
        arg_type = Bool
        default = true
    "--save_coeffs"
        arg_type = Bool
        default = true
    "--manual"
        arg_type = Bool
        default = false
end
args = parse_args(parsed_args)

# Create log directory if it doesn't exist
mkpath(args["log_dir"])

# Construct file paths
if args["alpha"] !== nothing
    log_file = joinpath(args["log_dir"], "trial$(args["trial"])_site$(args["site"])_alpha$(args["alpha"])_T$(args["num_steps"])_N$(args["N"]).log")
    output_file = joinpath(args["save_dir"], "pauli_alpha$(args["alpha"])_site$(args["site"])_trial$(args["trial"])_T$(args["num_steps"])_N$(args["N"]).jld2")
else
    log_file = joinpath(args["log_dir"], "trial$(args["trial"])_site$(args["site"])_M$(args["M"])_T$(args["num_steps"])_N$(args["N"]).log")
    output_file = joinpath(args["save_dir"], "pauli_M$(args["M"])_site$(args["site"])_trial$(args["trial"])_T$(args["num_steps"])_N$(args["N"]).jld2")
end

# Run with error handling and logging
open(log_file, "w") do io
    if isfile(output_file)
        println(io, "SKIPPED: Output file already exists at $output_file")
    else
        try
            if args["manual"]
                runtime = run_pauli_manual(
                    args["N"], args["num_steps"], args["M"],
                    args["site"];
                    alpha=args["alpha"], save_dir=args["save_dir"], calc_weight_dist=args["calc_weight_dist"], calc_entropy=args["calc_entropy"], save_coeffs=args["save_coeffs"]
                )
            else
                runtime = run_pauli_trial(
                    args["N"], args["num_steps"], args["M"],
                    args["site"], args["trial"];
                    alpha=args["alpha"], save_dir=args["save_dir"], calc_weight_dist=args["calc_weight_dist"], calc_entropy=args["calc_entropy"], save_coeffs=args["save_coeffs"]
                )
            end
            println(io, "SUCCESS")
            println(io, "Runtime: $runtime seconds")
        catch err
            println(io, "ERROR: ", err)
            println(io, "Backtrace:")
            for (i, frame) in enumerate(stacktrace(catch_backtrace()))
                println(io, "[$i] $frame")
            end
        end
    end
end
