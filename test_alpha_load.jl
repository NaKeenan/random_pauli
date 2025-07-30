using Pkg
Pkg.activate(".")

using JLD2
using Plots
using Statistics

# Check what alpha files we have for N=8
files = readdir("pauli_results")
alpha_files = filter(f -> occursin("alpha", f) && occursin("N8", f), files)
println("Found alpha files for N=8:")
for f in alpha_files
    println("  $f")
end

# Try to load the first available alpha file to test
if !isempty(alpha_files)
    test_file = "pauli_results/" * alpha_files[1]
    println("\nTesting file: $test_file")
    
    try
        jldopen(test_file, "r") do f
            results = read(f, "results")
            params = read(f, "params")
            
            println("Successfully loaded file!")
            println("Number of time steps: $(length(results))")
            println("Number of sites: $(length(results[1]))")
            println("Parameters: ")
            for (key, val) in params
                if key != "coeffs" && key != "angle_list"  # Skip large arrays
                    println("  $key: $val")
                end
            end
            
            # Create a simple plot
            correlation_data = [results[t][1] for t in 1:length(results)]
            
            plt = plot(1:length(results), correlation_data,
                      title="Correlation Function ⟨Z₁(t)⟩ (N=8)",
                      xlabel="Time step",
                      ylabel="⟨Z₁(t)⟩",
                      linewidth=2,
                      marker=:circle,
                      label="α = $(params["alpha"])")
            
            mkpath("plots")
            savefig(plt, "plots/test_alpha_N8.pdf")
            println("\nSaved test plot: plots/test_alpha_N8.pdf")
        end
    catch e
        println("Error loading file: $e")
    end
else
    println("No alpha files found for N=8!")
end
