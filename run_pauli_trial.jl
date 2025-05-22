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
end
args = parse_args(parsed_args)

# Create log directory if it doesn't exist
mkpath(args["log_dir"])

# Log file path
log_file = joinpath(args["log_dir"], "trial$(args["trial"])_site$(args["site"])_M$(args["M"]).log")

# Run with error handling and logging
open(log_file, "w") do io
    try
        runtime = run_pauli_trial(
            args["N"], args["num_steps"], args["M"],
            args["site"], args["trial"];
            save_dir=args["save_dir"]
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
