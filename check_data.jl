using Pkg
Pkg.activate(".")

using JLD2

# Quick diagnostic script to check correlation function values
function check_correlation_data()
    filename = "pauli_results/pauli_M256_site1_trial1_N22.jld2"
    
    if !isfile(filename)
        println("File not found: $filename")
        return
    end
    
    jldopen(filename, "r") do f
        results = read(f, "results")
        params = read(f, "params")
        
        println("Parameters:")
        for (k, v) in params
            if k in ["N", "M", "site", "trial", "num_steps"]
                println("  $k: $v")
            end
        end
        
        println("\nCorrelation data inspection:")
        println("Number of time steps: $(length(results))")
        println("Number of sites: $(length(results[1]))")
        
        # Check first few time steps for sites 1-5
        for t in 1:min(5, length(results))
            println("\nTime step $t:")
            for site in 1:min(5, length(results[t]))
                val = results[t][site]
                println("  Site $site: $val (real: $(real(val)), imag: $(imag(val)))")
            end
        end
        
        # Check for any unusual values
        println("\nChecking for unusual values:")
        all_values = Float64[]
        for t in 1:length(results)
            for site in 1:length(results[t])
                val = real(results[t][site])
                push!(all_values, val)
            end
        end
        
        println("Min value: $(minimum(all_values))")
        println("Max value: $(maximum(all_values))")
        println("Mean value: $(sum(all_values)/length(all_values))")
        println("Number of negative values: $(sum(all_values .< 0))")
        println("Number of zero values: $(sum(all_values .== 0))")
        println("Number of very small values (|x| < 1e-10): $(sum(abs.(all_values) .< 1e-10))")
    end
end

check_correlation_data()
