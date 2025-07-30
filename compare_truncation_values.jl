using Pkg
Pkg.activate(".")

using JLD2
using Plots
using Statistics

# Configure plotting backend to avoid GKS server issues
ENV["GKSwstype"] = "nul"  # Use null workstation to avoid display issues
gr()  # Use GR backend in headless mode

"""
Compare Pauli propagation results for different truncation parameters (M or alpha).
This script automatically detects whether M-based or alpha-based truncation was used,
loads results from multiple truncation values and plots the correlation functions
to show how truncation affects the accuracy of the simulation.
"""

function detect_truncation_type(pauli_results_dir="pauli_results")
    """
    Automatically detect whether we have M-based or alpha-based truncation files.
    
    Returns:
    - (:M, M_values) if M-based files are found
    - (:alpha, alpha_values) if alpha-based files are found
    - (nothing, nothing) if no recognized files are found
    """
    if !isdir(pauli_results_dir)
        return (nothing, nothing)
    end
    
    files = readdir(pauli_results_dir)
    
    # Look for M-based files
    M_files = filter(f -> occursin(r"pauli_M\d+_", f), files)
    
    # Look for alpha-based files  
    alpha_files = filter(f -> occursin(r"pauli_alpha", f), files)
    
    if !isempty(M_files)
        # Extract M values from filenames
        M_values = []
        for file in M_files
            m = match(r"pauli_M(\d+)_", file)
            if m !== nothing
                push!(M_values, parse(Int, m.captures[1]))
            end
        end
        return (:M, sort(unique(M_values)))
    elseif !isempty(alpha_files)
        # Extract alpha values from filenames
        alpha_values = []
        for file in alpha_files
            m = match(r"pauli_alpha([^_]+)_", file)
            if m !== nothing
                alpha_str = m.captures[1]
                # Handle scientific notation (e.g., "1.0e-10" or "1e-10")
                try
                    alpha_val = parse(Float64, alpha_str)
                    push!(alpha_values, alpha_val)
                catch
                    # If parsing fails, keep as string for now
                    push!(alpha_values, alpha_str)
                end
            end
        end
        return (:alpha, sort(unique(alpha_values)))
    else
        return (nothing, nothing)
    end
end

function load_and_compare_truncation_values(N, site, trial, truncation_type, truncation_values)
function load_and_compare_truncation_values(N, site, trial, truncation_type, truncation_values)
    """
    Load results for different truncation values and compare correlation functions.
    
    Arguments:
    - N: Number of qubits
    - site: Initial site (should be consistent across all truncation values)
    - trial: Trial number (should be consistent across all truncation values)
    - truncation_type: :M or :alpha
    - truncation_values: Array of M values or alpha values to compare
    
    Returns:
    - results_dict: Dictionary with truncation values as keys and results as values
    """
    results_dict = Dict()
    
    for val in truncation_values
        if truncation_type == :M
            # Look for files with M-based naming
            pattern = "pauli_M$(val)_site$(site)_trial$(trial)"
        else  # truncation_type == :alpha
            # Look for files with alpha-based naming
            pattern = "pauli_alpha$(val)_site$(site)_trial$(trial)"
        end
        
        # Find matching files (there might be different T values)
        files = readdir("pauli_results")
        matching_files = filter(f -> occursin(pattern, f) && endswith(f, "_N$(N).jld2"), files)
        
        if !isempty(matching_files)
            # Take the first matching file (or you could ask user to specify T)
            filename = "pauli_results/" * matching_files[1]
            
            jldopen(filename, "r") do f
                results = read(f, "results")
                params = read(f, "params")
                results_dict[val] = (results=results, params=params)
            end
            
            if truncation_type == :M
                println("Loaded results for M=$val")
            else
                println("Loaded results for α=$val") 
            end
        else
            if truncation_type == :M
                println("Warning: No files found for M=$val, site=$site, trial=$trial, N=$N")
            else
                println("Warning: No files found for α=$val, site=$site, trial=$trial, N=$N")
            end
        end
    end
    
    return results_dict
end
end

function plot_correlation_comparison(results_dict, site, truncation_type)
    """
    Plot correlation functions for different truncation values.
    
    Arguments:
    - results_dict: Dictionary with truncation values as keys and results as values
    - site: Site index to plot (1-indexed)
    - truncation_type: :M or :alpha
    """
    plt = plot()
    
    # Sort truncation values for consistent plotting
    trunc_values = sort(collect(keys(results_dict)))
    
    for val in trunc_values
        results = results_dict[val][:results]
        num_steps = length(results)
        
        # Extract correlation function for the specified site
        correlation_data = [results[t][site] for t in 1:num_steps]
        
        if truncation_type == :M
            label_str = "M = $val"
        else
            label_str = "α = $val"
        end
        
        plot!(plt, 1:num_steps, correlation_data, 
              label=label_str, 
              linewidth=2)
    end
    
    xlabel!(plt, "Time step")
    ylabel!(plt, "⟨Z_$(site)(t)⟩")
    
    if truncation_type == :M
        title!(plt, "Correlation Function Convergence vs Truncation Parameter M")
    else
        title!(plt, "Correlation Function Convergence vs Pruning Parameter α")
    end
    
    return plt
end

function plot_correlation_comparison_loglog(results_dict, site, truncation_type)
    """
    Plot correlation functions for different truncation values in log-log scale.
    Handles negative values by plotting both positive and negative parts separately.
    
    Arguments:
    - results_dict: Dictionary with truncation values as keys and results as values
    - site: Site index to plot (1-indexed)
    - truncation_type: :M or :alpha
    """
    plt = plot()
    
    # Sort truncation values for consistent plotting
    trunc_values = sort(collect(keys(results_dict)))
    
    for val in trunc_values
        results = results_dict[val][:results]
        num_steps = length(results)
        
        # Extract correlation function for the specified site
        correlation_data = [results[t][site] for t in 1:num_steps]
        
        # Split into positive and negative parts
        times = 1:num_steps
        pos_mask = correlation_data .> 0
        neg_mask = correlation_data .< 0
        
        if truncation_type == :M
            label_prefix = "M = $val"
        else
            label_prefix = "α = $val"
        end
        
        # Plot positive values
        if any(pos_mask)
            pos_times = times[pos_mask]
            pos_data = correlation_data[pos_mask]
            plot!(plt, pos_times, pos_data, 
                  label="$(label_prefix) (pos)", 
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
                  label="$(label_prefix) (neg)", 
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

function plot_all_sites_comparison(results_dict, truncation_type, max_sites_to_plot=5)
    """
    Plot correlation functions for multiple sites and truncation values.
    
    Arguments:
    - results_dict: Dictionary with truncation values as keys and results as values
    - truncation_type: :M or :alpha
    - max_sites_to_plot: Maximum number of sites to plot
    """
    trunc_values = sort(collect(keys(results_dict)))
    largest_val = maximum(trunc_values)
    
    # Get the number of sites from the largest truncation value result (most accurate)
    num_sites = length(results_dict[largest_val][:results][1])
    sites_to_plot = min(max_sites_to_plot, num_sites)
    
    # Create subplots
    plots = []
    
    for site in 1:sites_to_plot
        plt = plot()
        
        for val in trunc_values
            results = results_dict[val][:results]
            num_steps = length(results)
            
            # Extract correlation function for this site
            correlation_data = [results[t][site] for t in 1:num_steps]
            
            if truncation_type == :M
                label_str = "M = $val"
            else
                label_str = "α = $val"
            end
            
            plot!(plt, 1:num_steps, correlation_data, 
                  label=label_str, 
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

function plot_convergence_analysis(results_dict, truncation_type)
    """
    Analyze how results converge as truncation parameter increases.
    """
    trunc_values = sort(collect(keys(results_dict)))
    
    if length(trunc_values) < 2
        println("Need at least 2 truncation values for convergence analysis")
        return nothing
    end
    
    # Use the largest truncation value as reference
    reference_val = maximum(trunc_values)
    reference_results = results_dict[reference_val][:results]
    
    # Calculate differences from reference
    differences = []
    values_for_plot = []
    
    for val in trunc_values[1:end-1]  # Exclude the reference value
        results = results_dict[val][:results]
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
        push!(values_for_plot, val)
    end
    
    # Plot convergence
    if truncation_type == :M
        xlabel_str = "M (truncation parameter)"
        ylabel_str = "RMS difference from M=$(reference_val)"
        title_str = "Convergence Analysis: Error vs Truncation Parameter M"
    else
        xlabel_str = "α (pruning parameter)"
        ylabel_str = "RMS difference from α=$(reference_val)"
        title_str = "Convergence Analysis: Error vs Pruning Parameter α"
    end
    
    plt = plot(values_for_plot, differences, 
               xlabel=xlabel_str,
               ylabel=ylabel_str,
               title=title_str,
               marker=:circle,
               markersize=6,
               linewidth=2,
               xscale=:log10,
               yscale=:log10)
    
    return plt
end

function plot_weight_distributions_comparison(results_dict, truncation_type, plots_dir=".")
    """
    Create an animated video comparing weight distributions for different truncation values.
    
    Arguments:
    - results_dict: Dictionary with truncation values as keys and results as values
    - truncation_type: :M or :alpha
    - plots_dir: Directory to save the video
    """
    trunc_values = sort(collect(keys(results_dict)))
    
    # Check if weight distributions are available
    weight_dist_available = false
    for val in trunc_values
        params = results_dict[val][:params]
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
    
    for val in trunc_values
        params = results_dict[val][:params]
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
    for val in trunc_values
        params = results_dict[val][:params]
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
        
        for val in trunc_values
            params = results_dict[val][:params]
            if haskey(params, "weight_dist_array_fix") && !isnothing(params["weight_dist_array_fix"])
                weight_dist_array = params["weight_dist_array_fix"]
                
                if t <= length(weight_dist_array)
                    dist = weight_dist_array[t]
                    
                    # Convert to dense array
                    dense_dist = [real(get(dist, xi, 0.0)) for xi in x]
                    
                    if truncation_type == :M
                        label_str = "M = $val"
                    else
                        label_str = "α = $val"
                    end
                    
                    plot!(plt, x, dense_dist,
                          label=label_str,
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
        
        if truncation_type == :M
            title!(plt, "Weight Distribution Comparison (M-based) - Time Step $t")
        else
            title!(plt, "Weight Distribution Comparison (α-based) - Time Step $t")
        end
        
        # Add frame to animation
        frame(anim, plt)
    end
    
    # Save animation
    if truncation_type == :M
        video_filename = joinpath(plots_dir, "weight_distributions_comparison_M_N$(results_dict[trunc_values[1]][:params]["N"]).mp4")
    else
        video_filename = joinpath(plots_dir, "weight_distributions_comparison_alpha_N$(results_dict[trunc_values[1]][:params]["N"]).mp4")
    end
    
    mp4(anim, video_filename, fps=120)
    println("Saved weight distribution comparison video: $video_filename")
    
    return video_filename
end

# Main execution
function main()
    # Parameters
    N = 8
    site = 1
    trial = 1
    
    # Create plots directory
    plots_dir = "plots"
    mkpath(plots_dir)
    
    # Automatically detect truncation type and values
    println("Detecting truncation type and values...")
    truncation_type, truncation_values = detect_truncation_type()
    
    if truncation_type === nothing
        println("Error: No M-based or alpha-based result files found!")
        println("Please ensure that result files are in the pauli_results/ directory")
        println("Expected filenames:")
        println("  - For M-based: pauli_M<value>_site<site>_trial<trial>_T<steps>_N<N>.jld2")
        println("  - For alpha-based: pauli_alpha<value>_site<site>_trial<trial>_T<steps>_N<N>.jld2")
        return
    end
    
    if truncation_type == :M
        println("Detected M-based truncation with values: $truncation_values")
    else
        println("Detected α-based pruning with values: $truncation_values")
    end
    
    println("Loading results for N=$N, site=$site, trial=$trial")
    results_dict = load_and_compare_truncation_values(N, site, trial, truncation_type, truncation_values)
    
    if isempty(results_dict)
        println("No results found!")
        return
    end
    
    println("Creating plots...")
    
    # Generate filename prefix for plots
    if truncation_type == :M
        prefix = "correlation_M_comparison"
    else
        prefix = "correlation_alpha_comparison"
    end
    
    # Plot 1: Correlation function for the initial site (linear scale)
    plt1 = plot_correlation_comparison(results_dict, site, truncation_type)
    savefig(plt1, joinpath(plots_dir, "$(prefix)_site$(site)_N$(N).pdf"))
    println("Saved correlation comparison plot")
    
    # Plot 1b: Correlation function for the initial site (log-log scale)
    plt1b = plot_correlation_comparison_loglog(results_dict, site, truncation_type)
    savefig(plt1b, joinpath(plots_dir, "$(prefix)_site$(site)_N$(N)_loglog.pdf"))
    println("Saved correlation comparison log-log plot")
    
    # Plot 2: Multiple sites comparison (linear scale)
    plt2 = plot_all_sites_comparison(results_dict, truncation_type, 5)
    savefig(plt2, joinpath(plots_dir, "$(prefix)_multiple_sites_N$(N).pdf"))
    println("Saved multiple sites comparison plot")
    
    # Plot 3: Convergence analysis
    plt3 = plot_convergence_analysis(results_dict, truncation_type)
    if plt3 !== nothing
        if truncation_type == :M
            convergence_filename = "convergence_analysis_M_N$(N).pdf"
        else
            convergence_filename = "convergence_analysis_alpha_N$(N).pdf"
        end
        savefig(plt3, joinpath(plots_dir, convergence_filename))
        println("Saved convergence analysis plot")
    end
    
    # Plot 4: Weight distributions comparison video
    video_filename = plot_weight_distributions_comparison(results_dict, truncation_type, plots_dir)
    if video_filename !== nothing
        println("Saved weight distribution comparison video")
    end
    
    println("All plots and videos saved to $(plots_dir)/ directory!")
    
    if truncation_type == :M
        println("Analyzed M-based truncation results")
    else
        println("Analyzed α-based pruning results")
    end
end

# Run the analysis
main()
