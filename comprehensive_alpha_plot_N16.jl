using Pkg
Pkg.activate(".")

using JLD2
using Plots
using Statistics

# Load all available alpha results for N=16
function load_all_alpha_results_N16()
    results_dict = Dict()
    
    files = readdir("pauli_results")
    alpha_files = filter(f -> occursin("alpha", f) && occursin("N16", f), files)
    
    for file in alpha_files
        # Extract alpha value from filename
        m = match(r"pauli_alpha([^_]+)_", file)
        if m !== nothing
            alpha_str = m.captures[1]
            alpha_val = parse(Float64, alpha_str)
            
            filepath = "pauli_results/" * file
            
            try
                jldopen(filepath, "r") do f
                    results = read(f, "results")
                    params = read(f, "params")
                    results_dict[alpha_val] = (results=results, params=params, file=file)
                end
                println("Loaded α=$alpha_val from $file")
            catch e
                println("Error loading $file: $e")
            end
        end
    end
    
    return results_dict
end

# Create comprehensive comparison plots for N=16
function create_comparison_plots_N16(results_dict)
    if isempty(results_dict)
        println("No results to plot!")
        return
    end
    
    mkpath("plots")
    
    # Sort alpha values
    alpha_values = sort(collect(keys(results_dict)))
    println("Plotting α values: $alpha_values")
    
    # Colors for different alpha values
    colors = [:blue, :green, :orange, :red, :purple]
    
    # Plot 1: Site 1 correlation function comparison
    plt1 = plot(title="Correlation Function ⟨Z₁(t)⟩ for Different α Values (N=16)",
                xlabel="Time step",
                ylabel="⟨Z₁(t)⟩",
                legend=:topright,
                size=(800, 600))
    
    for (i, alpha) in enumerate(alpha_values)
        results = results_dict[alpha][:results]
        num_steps = length(results)
        correlation_data = [real(results[t][1]) for t in 1:num_steps]
        
        plot!(plt1, 1:num_steps, correlation_data,
              label="α = $alpha",
              linewidth=3,
              marker=:circle,
              markersize=5,
              color=colors[i])
    end
    
    savefig(plt1, "plots/alpha_comparison_all_N16.pdf")
    println("Saved: plots/alpha_comparison_all_N16.pdf")
    
    # Plot 2: Multiple sites for smallest alpha value (most accurate)
    smallest_alpha = minimum(alpha_values)
    results = results_dict[smallest_alpha][:results]
    num_sites = length(results[1])
    num_steps = length(results)
    
    sites_to_plot = min(6, num_sites)
    plots = []
    
    for site in 1:sites_to_plot
        plt = plot(title="Site $site",
                   xlabel="Time step",
                   ylabel="⟨Z_$(site)(t)⟩")
        
        correlation_data = [real(results[t][site]) for t in 1:num_steps]
        plot!(plt, 1:num_steps, correlation_data,
              label="α = $smallest_alpha",
              linewidth=2,
              marker=:circle,
              markersize=3,
              color=:blue)
        
        push!(plots, plt)
    end
    
    plt2 = plot(plots..., layout=(2, 3), size=(1200, 800))
    plot!(plt2, plot_title="Correlation Functions for Multiple Sites (α = $smallest_alpha, N=16)")
    
    savefig(plt2, "plots/alpha_multiple_sites_N16.pdf")
    println("Saved: plots/alpha_multiple_sites_N16.pdf")
    
    # Plot 3: Log-scale comparison
    plt3 = plot(title="Correlation Function ⟨Z₁(t)⟩ (Log Scale, N=16)",
                xlabel="Time step",
                ylabel="|⟨Z₁(t)⟩|",
                yscale=:log10,
                legend=:topright,
                size=(800, 600))
    
    for (i, alpha) in enumerate(alpha_values)
        results = results_dict[alpha][:results]
        num_steps = length(results)
        correlation_data = [abs(real(results[t][1])) for t in 1:num_steps]
        
        # Only plot non-zero values
        non_zero_mask = correlation_data .> 1e-15
        if any(non_zero_mask)
            times = (1:num_steps)[non_zero_mask]
            data = correlation_data[non_zero_mask]
            
            plot!(plt3, times, data,
                  label="α = $alpha",
                  linewidth=3,
                  marker=:circle,
                  markersize=5,
                  color=colors[i])
        end
    end
    
    savefig(plt3, "plots/alpha_comparison_log_N16.pdf")
    println("Saved: plots/alpha_comparison_log_N16.pdf")
    
    # Plot 4: Runtime vs Alpha
    plt4 = plot(title="Runtime vs Alpha Parameter (N=16)",
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
    
    plot!(plt4, alpha_values, runtimes,
          linewidth=3,
          marker=:circle,
          markersize=6,
          color=:red)
    
    savefig(plt4, "plots/alpha_runtime_N16.pdf")
    println("Saved: plots/alpha_runtime_N16.pdf")
    
    # Plot 5: Comparison of final correlation values
    plt5 = plot(title="Final Correlation ⟨Z₁(T=10)⟩ vs Alpha (N=16)",
                xlabel="α (pruning parameter)",
                ylabel="Final ⟨Z₁(T)⟩",
                xscale=:log10,
                legend=false,
                size=(600, 400))
    
    final_correlations = []
    for alpha in alpha_values
        results = results_dict[alpha][:results]
        final_corr = real(results[end][1])
        push!(final_correlations, final_corr)
    end
    
    plot!(plt5, alpha_values, final_correlations,
          linewidth=3,
          marker=:circle,
          markersize=6,
          color=:blue)
    
    savefig(plt5, "plots/alpha_final_correlation_N16.pdf")
    println("Saved: plots/alpha_final_correlation_N16.pdf")
    
    # Summary statistics
    println("\n" * "="^70)
    println("ALPHA PRUNING ANALYSIS FOR N=16 SYSTEM")
    println("="^70)
    println("Alpha Value | Runtime (s) | Final ⟨Z₁⟩ | Interpretation")
    println("-"^70)
    
    for alpha in alpha_values
        results = results_dict[alpha][:results]
        params = results_dict[alpha][:params]
        final_corr = real(results[end][1])
        runtime = params["runtime"]
        
        if alpha >= 0.1
            interpretation = "Very aggressive pruning"
        elseif alpha >= 0.01
            interpretation = "Aggressive pruning"
        elseif alpha >= 0.001
            interpretation = "Moderate pruning"
        else
            interpretation = "Conservative pruning"
        end
        
        println("$(lpad(string(alpha), 11)) | $(lpad(round(runtime, digits=3), 11)) | $(lpad(round(final_corr, digits=6), 11)) | $interpretation")
    end
    println("="^70)
    
    # Additional analysis
    println("\nAdditional Analysis:")
    println("• System size: N = 16 qubits")
    println("• Time evolution: T = 10 steps")
    println("• Pruning range: α ∈ [$(minimum(alpha_values)), $(maximum(alpha_values))]")
    
    # Check correlation decay
    smallest_alpha_results = results_dict[minimum(alpha_values)][:results]
    largest_alpha_results = results_dict[maximum(alpha_values)][:results]
    
    initial_corr_min = real(smallest_alpha_results[1][1])
    final_corr_min = real(smallest_alpha_results[end][1])
    
    initial_corr_max = real(largest_alpha_results[1][1])
    final_corr_max = real(largest_alpha_results[end][1])
    
    println("• Correlation decay (α = $(minimum(alpha_values))): $(initial_corr_min) → $(final_corr_min)")
    println("• Correlation decay (α = $(maximum(alpha_values))): $(initial_corr_max) → $(final_corr_max)")
    
    return plt1, plt2, plt3, plt4, plt5
end

# Main execution
println("Loading all alpha-based results for N=16...")
results_dict = load_all_alpha_results_N16()

if !isempty(results_dict)
    create_comparison_plots_N16(results_dict)
    println("\n🎯 Analysis complete for N=16! Check the plots/ directory for results.")
    println("📊 Generated plots:")
    println("   • alpha_comparison_all_N16.pdf - Linear scale comparison")
    println("   • alpha_multiple_sites_N16.pdf - Multiple sites analysis")
    println("   • alpha_comparison_log_N16.pdf - Log scale comparison")
    println("   • alpha_runtime_N16.pdf - Runtime vs alpha")
    println("   • alpha_final_correlation_N16.pdf - Final correlation vs alpha")
else
    println("❌ No alpha-based results found for N=16!")
end
