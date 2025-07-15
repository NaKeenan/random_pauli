using Pkg
Pkg.activate(".")

using JLD2
using Plots
using Statistics

# Configure plotting backend to avoid GKS server issues
ENV["GKSwstype"] = "nul"  # Use null workstation to avoid display issues
gr()  # Use GR backend in headless mode

"""
Compare Pauli propagation results for different truncation parameters M.
This script loads results from multiple M values and plots the correlation functions
to show how truncation affects the accuracy of the simulation.
"""

function load_and_compare_M_values(N, site, trial, Ms)
    """
    Load results for different M values and compare correlation functions.
    
    Arguments:
    - N: Number of qubits
    - site: Initial site (should be consistent across all M values)
    - trial: Trial number (should be consistent across all M values)
    - Ms: Array of M values to compare
    
    Returns:
    - results_dict: Dictionary with M values as keys and results as values
    """
    results_dict = Dict()
    
    for M in Ms
        filename = "pauli_results/pauli_M$(M)_site$(site)_trial$(trial)_N$(N).jld2"
        if isfile(filename)
            jldopen(filename, "r") do f
                results = read(f, "results")
                params = read(f, "params")
                results_dict[M] = (results=results, params=params)
            end
            println("Loaded results for M=$M")
        else
            println("Warning: File $filename not found")
        end
    end
    
    return results_dict
end

function plot_correlation_comparison(results_dict, site)
    """
    Plot correlation functions for different M values.
    
    Arguments:
    - results_dict: Dictionary with M values as keys and results as values
    - site: Site index to plot (1-indexed)
    """
    plt = plot()
    
    # Sort M values for consistent plotting
    Ms = sort(collect(keys(results_dict)))
    
    for M in Ms
        results = results_dict[M][:results]
        num_steps = length(results)
        
        # Extract correlation function for the specified site
        correlation_data = [results[t][site] for t in 1:num_steps]
        
        plot!(plt, 1:num_steps, correlation_data, 
              label="M = $M", 
              linewidth=2)
    end
    
    xlabel!(plt, "Time step")
    ylabel!(plt, "⟨Z_$(site)(t)⟩")
    title!(plt, "Correlation Function Convergence vs Truncation Parameter M")
    
    return plt
end

function plot_correlation_comparison_loglog(results_dict, site)
    """
    Plot correlation functions for different M values in log-log scale.
    Handles negative values by plotting both positive and negative parts separately.
    
    Arguments:
    - results_dict: Dictionary with M values as keys and results as values
    - site: Site index to plot (1-indexed)
    """
    plt = plot()
    
    # Sort M values for consistent plotting
    Ms = sort(collect(keys(results_dict)))
    
    for M in Ms
        results = results_dict[M][:results]
        num_steps = length(results)
        
        # Extract correlation function for the specified site
        correlation_data = [results[t][site] for t in 1:num_steps]
        
        # Split into positive and negative parts
        times = 1:num_steps
        pos_mask = correlation_data .> 0
        neg_mask = correlation_data .< 0
        
        # Plot positive values
        if any(pos_mask)
            pos_times = times[pos_mask]
            pos_data = correlation_data[pos_mask]
            plot!(plt, pos_times, pos_data, 
                  label="M = $M (pos)", 
                  linewidth=2,
                  xscale=:log10,
                  yscale=:log10,
                  linestyle=:solid)
        end
        
        # Plot negative values (absolute value with different style)
        if any(neg_mask)
            neg_times = times[neg_mask]
            neg_data = abs.(correlation_data[neg_mask])
            plot!(plt, neg_times, neg_data, 
                  label="M = $M (neg)", 
                  linewidth=2,
                  xscale=:log10,
                  yscale=:log10,
                  linestyle=:dash)
        end
    end
    
    xlabel!(plt, "Time step")
    ylabel!(plt, "|⟨Z_$(site)(t)⟩|")
    title!(plt, "Correlation Function Log-Log (Positive=solid, Negative=dash)")
    
    return plt
end

function plot_all_sites_comparison(results_dict, max_sites_to_plot=5)
    """
    Plot correlation functions for multiple sites and M values.
    
    Arguments:
    - results_dict: Dictionary with M values as keys and results as values
    - max_sites_to_plot: Maximum number of sites to plot
    """
    Ms = sort(collect(keys(results_dict)))
    largest_M = maximum(Ms)
    
    # Get the number of sites from the largest M result (most accurate)
    num_sites = length(results_dict[largest_M][:results][1])
    sites_to_plot = min(max_sites_to_plot, num_sites)
    
    # Create subplots
    plots = []
    
    for site in 1:sites_to_plot
        plt = plot()
        
        for M in Ms
            results = results_dict[M][:results]
            num_steps = length(results)
            
            # Extract correlation function for this site
            correlation_data = [results[t][site] for t in 1:num_steps]
            
            plot!(plt, 1:num_steps, correlation_data, 
                  label="M = $M", 
                  xscale=:log10,
                  yscale=:log10,
                  linewidth=2)
        end
        
        xlabel!(plt, "Time step")
        ylabel!(plt, "⟨Z_$(site)(t)⟩")
        title!(plt, "Site $site")
        
        if site == 1
            # Only show legend for the first plot
            plot!(plt, legend=:topright)
        else
            plot!(plt, legend=false)
        end
        
        push!(plots, plt)
    end
    
    # Combine all plots
    combined_plot = plot(plots..., layout=(1, sites_to_plot), size=(300*sites_to_plot, 400))
    
    return combined_plot
end

function plot_convergence_analysis(results_dict)
    """
    Analyze how results converge as M increases.
    """
    Ms = sort(collect(keys(results_dict)))
    
    if length(Ms) < 2
        println("Need at least 2 M values for convergence analysis")
        return nothing
    end
    
    # Use the largest M as reference
    reference_M = maximum(Ms)
    reference_results = results_dict[reference_M][:results]
    
    # Calculate differences from reference
    differences = []
    M_values = []
    
    for M in Ms[1:end-1]  # Exclude the reference M
        results = results_dict[M][:results]
        num_steps = length(results)
        
        # Calculate RMS difference from reference
        total_diff = 0.0
        total_points = 0
        
        for t in 1:num_steps
            for site in 1:length(results[t])
                diff = abs(results[t][site] - reference_results[t][site])
                total_diff += diff^2
                total_points += 1
            end
        end
        
        rms_diff = sqrt(total_diff / total_points)
        push!(differences, rms_diff)
        push!(M_values, M)
    end
    
    # Plot convergence
    plt = plot(M_values, differences, 
               xlabel="M (truncation parameter)",
               ylabel="RMS difference from M=$(reference_M)",
               title="Convergence Analysis: Error vs Truncation Parameter",
               marker=:circle,
               markersize=6,
               linewidth=2,
               xscale=:log10,
               yscale=:log10)
    
    return plt
end

function plot_weight_distributions_comparison(results_dict, plots_dir=".")
    """
    Create an animated video comparing weight distributions for different M values.
    
    Arguments:
    - results_dict: Dictionary with M values as keys and results as values
    - plots_dir: Directory to save the video
    """
    Ms = sort(collect(keys(results_dict)))
    
    # Check if weight distributions are available
    weight_dist_available = false
    for M in Ms
        params = results_dict[M][:params]
        if haskey(params, "weight_dist_array_fix") && !isnothing(params["weight_dist_array_fix"])
            weight_dist_available = true
            break
        end
    end
    
    if !weight_dist_available
        println("Warning: No weight distribution data found. Skipping weight distribution video.")
        return nothing
    end
    
    # Get the maximum number of time steps and maximum weight
    max_time_steps = 0
    max_weight = 0
    
    for M in Ms
        params = results_dict[M][:params]
        if haskey(params, "weight_dist_array_fix") && !isnothing(params["weight_dist_array_fix"])
            weight_dist_array = params["weight_dist_array_fix"]
            max_time_steps = max(max_time_steps, length(weight_dist_array))
            
            for dist in weight_dist_array
                if !isempty(dist)
                    max_weight = max(max_weight, maximum(collect(keys(dist))))
                end
            end
        end
    end
    
    if max_time_steps == 0
        println("Warning: No valid weight distribution data found.")
        return nothing
    end
    
    # Create weight range
    x = 0:max_weight
    
    # Find global maximum amplitude for consistent y-axis
    global_max_amp = 0.0
    for M in Ms
        params = results_dict[M][:params]
        if haskey(params, "weight_dist_array_fix") && !isnothing(params["weight_dist_array_fix"])
            weight_dist_array = params["weight_dist_array_fix"]
            for dist in weight_dist_array
                if !isempty(dist)
                    max_amp = maximum(real(collect(values(dist))))
                    global_max_amp = max(global_max_amp, max_amp)
                end
            end
        end
    end
    
    # Create animation
    anim = Plots.Animation()
    
    for t in 1:max_time_steps
        plt = plot()
        
        for M in Ms
            params = results_dict[M][:params]
            if haskey(params, "weight_dist_array_fix") && !isnothing(params["weight_dist_array_fix"])
                weight_dist_array = params["weight_dist_array_fix"]
                
                if t <= length(weight_dist_array)
                    dist = weight_dist_array[t]
                    
                    # Convert to dense array
                    dense_dist = [real(get(dist, xi, 0.0)) for xi in x]
                    
                    plot!(plt, x, dense_dist,
                          label="M = $M",
                          linewidth=2,
                          marker=:circle,
                          markersize=3)
                end
            end
        end
        
        # Set consistent axes and labels
        ylims!(plt, (0, global_max_amp * 1.1))
        xlabel!(plt, "Weight")
        ylabel!(plt, "Amplitude")
        title!(plt, "Weight Distribution Comparison - Time Step $t")
        
        # Add frame to animation
        frame(anim, plt)
    end
    
    # Save animation
    video_filename = joinpath(plots_dir, "weight_distributions_comparison_N$(results_dict[Ms[1]][:params]["N"]).mp4")
    mp4(anim, video_filename, fps=120)
    println("Saved weight distribution comparison video: $video_filename")
    
    return video_filename
end

# Main execution
function main()
    # Parameters
    N = 22
    site = 1
    trial = 1
    Ms = [256, 512, 1024, 2048, 4096, 8192, 16384]  # Example M values to compare
    
    # Create plots directory
    plots_dir = "plots"
    mkpath(plots_dir)
    
    println("Loading results for N=$N, site=$site, trial=$trial")
    results_dict = load_and_compare_M_values(N, site, trial, Ms)
    
    if isempty(results_dict)
        println("No results found!")
        return
    end
    
    println("Creating plots...")
    
    # Plot 1: Correlation function for the initial site (linear scale)
    plt1 = plot_correlation_comparison(results_dict, site)
    savefig(plt1, joinpath(plots_dir, "correlation_M_comparison_site$(site)_N$(N).pdf"))
    println("Saved correlation comparison plot")
    
    # Plot 1b: Correlation function for the initial site (log-log scale)
    plt1b = plot_correlation_comparison_loglog(results_dict, site)
    savefig(plt1b, joinpath(plots_dir, "correlation_M_comparison_site$(site)_N$(N)_loglog.pdf"))
    println("Saved correlation comparison log-log plot")
    
    # Plot 2: Multiple sites comparison (linear scale)
    plt2 = plot_all_sites_comparison(results_dict, 5)
    savefig(plt2, joinpath(plots_dir, "correlation_M_comparison_multiple_sites_N$(N).pdf"))
    println("Saved multiple sites comparison plot")
    
    # Plot 3: Convergence analysis
    plt3 = plot_convergence_analysis(results_dict)
    if plt3 !== nothing
        savefig(plt3, joinpath(plots_dir, "convergence_analysis_N$(N).pdf"))
        println("Saved convergence analysis plot")
    end
    
    # Plot 4: Weight distributions comparison video
    video_filename = plot_weight_distributions_comparison(results_dict, plots_dir)
    if video_filename !== nothing
        println("Saved weight distribution comparison video")
    end
    
    println("All plots and videos saved to $(plots_dir)/ directory!")
end

# Run the analysis
main()
