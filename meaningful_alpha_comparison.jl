using Pkg
Pkg.activate(".")

using JLD2
using Plots
using Statistics

# Load results with meaningful alpha values for N=8
function load_meaningful_alpha_results()
    results_dict = Dict()
    
    # Include both small and large alpha values for comparison
    alpha_files = [
        ("1.0e-10", 2),  # Very small alpha (minimal pruning)
        ("0.01", 5),     # Moderate pruning
        ("0.1", 6),      # More aggressive pruning  
        ("1.0", 7),      # Strong pruning
        ("10.0", 8),     # Very aggressive pruning
    ]
    
    for (alpha_str, trial) in alpha_files
        alpha_val = parse(Float64, alpha_str)
        filename = "pauli_results/pauli_alpha$(alpha_str)_site1_trial$(trial)_T5_N8.jld2"
        
        if isfile(filename)
            try
                jldopen(filename, "r") do f
                    results = read(f, "results")
                    params = read(f, "params")
                    results_dict[alpha_val] = (results=results, params=params, file=filename)
                end
                println("✓ Loaded α=$alpha_val from trial $trial")
            catch e
                println("✗ Error loading $filename: $e")
            end
        else
            println("✗ File not found: $filename")
        end
    end
    
    return results_dict
end

# Create comprehensive comparison showing pruning effects
function create_pruning_comparison(results_dict)
    if isempty(results_dict)
        println("No results to plot!")
        return
    end
    
    mkpath("plots")
    
    # Sort alpha values (log scale for better visualization)
    alpha_values = sort(collect(keys(results_dict)))
    println("Comparing α values: $alpha_values")
    
    # Colors for different alpha values
    colors = [:blue, :green, :orange, :red, :purple]
    
    # Plot 1: Correlation function comparison (linear scale)
    plt1 = plot(title="Effect of Alpha Pruning on ⟨Z₁(t)⟩ (N=8)",
                xlabel="Time step",
                ylabel="⟨Z₁(t)⟩",
                legend=:topright,
                size=(800, 600))
    
    for (i, alpha) in enumerate(alpha_values)
        results = results_dict[alpha][:results]
        num_steps = length(results)
        correlation_data = [real(results[t][1]) for t in 1:num_steps]
        
        if alpha < 1e-5
            label_str = "α = $(alpha) (minimal pruning)"
        else
            label_str = "α = $(alpha)"
        end
        
        plot!(plt1, 1:num_steps, correlation_data,
              label=label_str,
              linewidth=3,
              marker=:circle,
              markersize=5,
              color=colors[i])
    end
    
    savefig(plt1, "plots/alpha_pruning_effect_N8.pdf")
    println("Saved: plots/alpha_pruning_effect_N8.pdf")
    
    # Plot 2: Log scale to see differences better
    plt2 = plot(title="Alpha Pruning Effect (Log Scale)",
                xlabel="Time step", 
                ylabel="|⟨Z₁(t)⟩|",
                yscale=:log10,
                legend=:topright,
                size=(800, 600))
    
    for (i, alpha) in enumerate(alpha_values)
        results = results_dict[alpha][:results]
        num_steps = length(results)
        correlation_data = [abs(real(results[t][1])) for t in 1:num_steps]
        
        # Filter out very small values for log plot
        valid_mask = correlation_data .> 1e-12
        if any(valid_mask)
            times = (1:num_steps)[valid_mask]
            data = correlation_data[valid_mask]
            
            if alpha < 1e-5
                label_str = "α = $(alpha) (minimal)"
            else
                label_str = "α = $(alpha)"
            end
            
            plot!(plt2, times, data,
                  label=label_str,
                  linewidth=3,
                  marker=:circle,
                  markersize=5,
                  color=colors[i])
        end
    end
    
    savefig(plt2, "plots/alpha_pruning_effect_log_N8.pdf")
    println("Saved: plots/alpha_pruning_effect_log_N8.pdf")
    
    # Plot 3: Runtime vs Alpha (if we have runtime data)
    plt3 = plot(title="Runtime vs Alpha Parameter",
                xlabel="α (pruning parameter)",
                ylabel="Runtime (seconds)",
                xscale=:log10,
                legend=false,
                size=(600, 400))
    
    runtimes = []
    for alpha in alpha_values
        runtime = results_dict[alpha][:params]["runtime"]
        push!(runtimes, runtime)
    end
    
    plot!(plt3, alpha_values, runtimes,
          linewidth=3,
          marker=:circle,
          markersize=6,
          color=:red)
    
    savefig(plt3, "plots/alpha_runtime_N8.pdf")
    println("Saved: plots/alpha_runtime_N8.pdf")
    
    # Summary table
    println("\n" * "="^60)
    println("ALPHA PRUNING COMPARISON SUMMARY (N=8)")
    println("="^60)
    println("Alpha Value | Runtime (s) | Final ⟨Z₁⟩ | Interpretation")
    println("-"^60)
    
    for alpha in alpha_values
        params = results_dict[alpha][:params]
        results = results_dict[alpha][:results]
        
        runtime = params["runtime"]
        final_corr = real(results[end][1])
        
        if alpha < 1e-5
            interpretation = "Minimal pruning - most accurate"
        elseif alpha < 0.1
            interpretation = "Moderate pruning"
        elseif alpha < 1.0
            interpretation = "Aggressive pruning"
        else
            interpretation = "Very aggressive pruning"
        end
        
        println("$(lpad(string(alpha), 11)) | $(lpad(round(runtime, digits=3), 11)) | $(lpad(round(final_corr, digits=6), 11)) | $interpretation")
    end
    println("="^60)
    
    return plt1, plt2, plt3
end

# Main execution
println("Loading meaningful alpha results for N=8...")
results_dict = load_meaningful_alpha_results()

if !isempty(results_dict)
    create_pruning_comparison(results_dict)
    println("\n🎯 Analysis complete! The plots now show meaningful differences in pruning behavior.")
    println("📊 Check plots/ directory for:")
    println("   • alpha_pruning_effect_N8.pdf - Linear scale comparison")
    println("   • alpha_pruning_effect_log_N8.pdf - Log scale comparison") 
    println("   • alpha_runtime_N8.pdf - Runtime vs alpha")
else
    println("❌ No meaningful alpha results found!")
end
