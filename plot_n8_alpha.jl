using Pkg
Pkg.activate(".")

using JLD2
using Plots
using Statistics

# Configure plotting backend
ENV["GKSwstype"] = "nul"
gr()

# Load the alpha-based results for N=8
function load_alpha_results()
    results_dict = Dict()
    
    # Load different alpha values
    alpha_values = [1.0e-10, 1.0e-9, 1.0e-8]
    trial_mapping = Dict(1.0e-10 => 2, 1.0e-9 => 4, 1.0e-8 => 3)  # Map alpha to trial number
    
    for alpha in alpha_values
        trial = trial_mapping[alpha]
        filename = "pauli_results/pauli_alpha$(alpha)_site1_trial$(trial)_T5_N8.jld2"
        
        if isfile(filename)
            jldopen(filename, "r") do f
                results = read(f, "results")
                params = read(f, "params")
                results_dict[alpha] = (results=results, params=params)
            end
            println("Loaded results for α=$alpha (trial $trial)")
        else
            println("Warning: File not found: $filename")
        end
    end
    
    return results_dict
end

# Plot correlation functions for different alpha values
function plot_alpha_comparison(results_dict)
    plt = plot()
    
    # Sort alpha values
    alpha_values = sort(collect(keys(results_dict)))
    
    for alpha in alpha_values
        results = results_dict[alpha][:results]
        num_steps = length(results)
        
        # Extract correlation function for site 1
        correlation_data = [results[t][1] for t in 1:num_steps]
        
        plot!(plt, 1:num_steps, correlation_data, 
              label="α = $alpha", 
              linewidth=2,
              marker=:circle,
              markersize=4)
    end
    
    xlabel!(plt, "Time step")
    ylabel!(plt, "⟨Z₁(t)⟩")
    title!(plt, "Correlation Function for Different α Values (N=8)")
    
    return plt
end

# Plot multiple sites
function plot_multiple_sites(results_dict, max_sites=4)
    alpha_values = sort(collect(keys(results_dict)))
    largest_alpha = maximum(alpha_values)
    
    # Get number of sites from the largest alpha result
    num_sites = length(results_dict[largest_alpha][:results][1])
    sites_to_plot = min(max_sites, num_sites)
    
    plots = []
    
    for site in 1:sites_to_plot
        plt = plot()
        
        for alpha in alpha_values
            results = results_dict[alpha][:results]
            num_steps = length(results)
            
            # Extract correlation function for this site
            correlation_data = [results[t][site] for t in 1:num_steps]
            
            plot!(plt, 1:num_steps, correlation_data, 
                  label="α = $alpha", 
                  linewidth=2,
                  marker=:circle,
                  markersize=3)
        end
        
        xlabel!(plt, "Time step")
        ylabel!(plt, "⟨Z_$(site)(t)⟩")
        title!(plt, "Site $site")
        
        if site == 1
            plot!(plt, legend=:topright)
        else
            plot!(plt, legend=false)
        end
        
        push!(plots, plt)
    end
    
    # Combine all plots
    combined_plot = plot(plots..., layout=(2, 2), size=(800, 600))
    
    return combined_plot
end

# Main execution
println("Loading alpha-based results for N=8...")
results_dict = load_alpha_results()

if !isempty(results_dict)
    println("Creating plots...")
    
    # Create plots directory
    mkpath("plots")
    
    # Plot 1: Single site comparison
    plt1 = plot_alpha_comparison(results_dict)
    savefig(plt1, "plots/correlation_alpha_comparison_site1_N8.pdf")
    println("Saved single site comparison: plots/correlation_alpha_comparison_site1_N8.pdf")
    
    # Plot 2: Multiple sites comparison
    plt2 = plot_multiple_sites(results_dict, 4)
    savefig(plt2, "plots/correlation_alpha_comparison_multiple_sites_N8.pdf")
    println("Saved multiple sites comparison: plots/correlation_alpha_comparison_multiple_sites_N8.pdf")
    
    println("Analysis complete!")
else
    println("No results found!")
end
