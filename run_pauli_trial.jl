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
end
args = parse_args(parsed_args)

# Create directories if they don't exist
mkpath(args["log_dir"])
mkpath(args["save_dir"])

# Construct file paths
log_file = joinpath(args["log_dir"], "trial$(args["trial"])_site$(args["site"])_M$(args["M"]).log")
output_file = joinpath(args["save_dir"], "pauli_M$(args["M"])_site$(args["site"])_trial$(args["trial"]).jld2")  # example output file name

open(log_file, "w") do io
    if isfile(output_file)
        println(io, "SKIPPED: Output file already exists at $output_file")
    else
        try
            runtime = run_pauli_trial(
                args["N"], args["num_steps"], args["M"],
                args["site"], args["trial"];
                save_dir=args["save_dir"], calc_weight_dist=args["calc_weight_dist"]
            )
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
