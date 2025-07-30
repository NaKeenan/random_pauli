using Pkg
Pkg.activate(".")

using JLD2
using Plots
using Statistics

# Load all available alpha results for N=8
function load_all_alpha_results()
    results_dict = Dict()
    
    files = readdir("pauli_results")
    alpha_files = filter(f -> occursin("alpha", f) && occursin("N8", f), files)
    
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

# Create comprehensive comparison plots
function create_comparison_plots(results_dict)
    if isempty(results_dict)
        println("No results to plot!")
        return
    end
    
    mkpath("plots")
    
    # Sort alpha values
    alpha_values = sort(collect(keys(results_dict)))
    println("Plotting α values: $alpha_values")
    
    # Plot 1: Site 1 correlation function comparison
    plt1 = plot(title="Correlation Function ⟨Z₁(t)⟩ for Different α Values (N=8)",
                xlabel="Time step",
                ylabel="⟨Z₁(t)⟩",
                legend=:topright)
    
    for alpha in alpha_values
        results = results_dict[alpha][:results]
        num_steps = length(results)
        correlation_data = [real(results[t][1]) for t in 1:num_steps]
        
        plot!(plt1, 1:num_steps, correlation_data,
              label="α = $alpha",
              linewidth=2,
              marker=:circle,
              markersize=4)
    end
    
    savefig(plt1, "plots/alpha_comparison_all_N8.pdf")
    println("Saved: plots/alpha_comparison_all_N8.pdf")
    
    # Plot 2: Multiple sites for largest alpha value (most accurate)
    largest_alpha = maximum(alpha_values)
    results = results_dict[largest_alpha][:results]
    num_sites = length(results[1])
    num_steps = length(results)
    
    sites_to_plot = min(4, num_sites)
    plots = []
    
    for site in 1:sites_to_plot
        plt = plot(title="Site $site",
                   xlabel="Time step",
                   ylabel="⟨Z_$(site)(t)⟩")
        
        correlation_data = [real(results[t][site]) for t in 1:num_steps]
        plot!(plt, 1:num_steps, correlation_data,
              label="α = $largest_alpha",
              linewidth=2,
              marker=:circle,
              markersize=3,
              color=:blue)
        
        push!(plots, plt)
    end
    
    plt2 = plot(plots..., layout=(2, 2), size=(800, 600))
    plot!(plt2, plot_title="Correlation Functions for Multiple Sites (α = $largest_alpha, N=8)")
    
    savefig(plt2, "plots/alpha_multiple_sites_N8.pdf")
    println("Saved: plots/alpha_multiple_sites_N8.pdf")
    
    # Plot 3: Log-scale comparison
    plt3 = plot(title="Correlation Function ⟨Z₁(t)⟩ (Log Scale, N=8)",
                xlabel="Time step",
                ylabel="|⟨Z₁(t)⟩|",
                yscale=:log10,
                legend=:topright)
    
    for alpha in alpha_values
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
                  linewidth=2,
                  marker=:circle,
                  markersize=4)
        end
    end
    
    savefig(plt3, "plots/alpha_comparison_log_N8.pdf")
    println("Saved: plots/alpha_comparison_log_N8.pdf")
    
    # Summary statistics
    println("\n=== Summary Statistics ===")
    for alpha in alpha_values
        results = results_dict[alpha][:results]
        params = results_dict[alpha][:params]
        final_corr = real(results[end][1])
        
        println("α = $alpha:")
        println("  Runtime: $(params["runtime"]) seconds")
        println("  Final ⟨Z₁(T)⟩: $final_corr")
        println("  File: $(results_dict[alpha][:file])")
        println()
    end
end

# Main execution
println("Loading all alpha-based results for N=8...")
results_dict = load_all_alpha_results()

if !isempty(results_dict)
    create_comparison_plots(results_dict)
    println("\nAnalysis complete! Check the plots/ directory for results.")
else
    println("No alpha-based results found for N=8!")
end
