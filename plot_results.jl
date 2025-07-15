using Pkg
Pkg.activate(".")

using JLD2
using Glob
using Plots
using Statistics
gr()
ENV["GKSwstype"] = "nul"
function load_results_for_N_grouped_by_M(N_target::Int, results_dir::String="pauli_results")
    files = Glob.glob(joinpath(results_dir, "*.jld2"))
    results_by_M = Dict{Int, Vector{Vector{Float64}}}()

    for file in files
        jldopen(file, "r") do f
            params = read(f, "params")
            if params["N"] == N_target
                M = params["M"]
                site = params["site"]
                num_steps = params["num_steps"]
                data = read(f, "results")  # may be Vector{BigFloat}
                data = [data[t][site] for t in 1:num_steps]
                # Convert to Vector{Float64}
                data_float64 = Float64.(data)
                if !haskey(results_by_M, M)
                    results_by_M[M] = Vector{Vector{Float64}}()
                end
                push!(results_by_M[M], data_float64)
            end
        end
    end

    return results_by_M
end

function average_results(results_array)
    mat = hcat(results_array...)
    mean_result = mean(mat, dims=2)
    return vec(mean_result)
end

function plot_average_for_N_multiple_M(N_target::Int; results_dir="pauli_results", savepath="average_plot_Ms.png")
    results_by_M = load_results_for_N_grouped_by_M(N_target, results_dir)

    if isempty(results_by_M)
        println("No results found for N = $N_target in $results_dir")
        return
    end

    plt = plot()
    for (M, results_array) in sort(collect(results_by_M))
        avg_result = average_results(results_array)
        num_steps = length(avg_result)

        plot!(plt, 1:num_steps, avg_result, label="M = $M")
    end

    xlabel!(plt, "Time step")
    ylabel!(plt, "Average result")
    title!(plt, "Average results for N = $N_target, separated by M")
    savefig(plt, savepath)
    println("Plot saved to $savepath")
end

function plot_weight_dist(N_target)
    plots_dir = "plots"
    mkpath(plots_dir)
    
    files = Glob.glob(joinpath("pauli_results", "*N$N_target.jld2"))

    for file in files
        weight_dist_array = nothing
        jldopen(file, "r") do f
            params = read(f, "params")
            weight_dist_array = get(params, "weight_dist_array_fix", nothing)
        end
        if isnothing(weight_dist_array)
            continue
        end

        # --- Animation over full weight distribution ---
        x_max = maximum([maximum(collect(keys(d))) for d in weight_dist_array])
        x = 0:x_max
        dense_dists = [real([get(d, xi, 0.0) for xi in x]) for d in weight_dist_array]
        y_max = maximum([maximum(dd) for dd in dense_dists])

        anim = Plots.Animation()
        for i in 1:length(dense_dists)
            p = plot(x, dense_dists[i],
                     ylim=(0, y_max),
                     title="Distribution $i",
                     xlabel="Weight", ylabel="Amplitude")
            frame(anim, p)
        end

        video_filename = joinpath(plots_dir, "weight_distributions_N$N_target.mp4")
        mp4(anim, video_filename, fps=120)
        println("Saved animation to $video_filename")

        # --- Line plot of weight 1 over time ---
        w1_vals = [get(d, 1, 0.0) |> real for d in weight_dist_array]
        println(w1_vals)
        p2 = plot(1:length(w1_vals), w1_vals,
                  xlabel="Time step", ylabel="Amplitude at weight 1",
                  title="Weight 1 amplitude over time for N=$N_target", xscale=:log10, yscale=:log10)
        savefig(p2, joinpath(plots_dir, "weight1_vs_time_N$N_target.pdf"))
        println("Saved weight-1 plot to $(joinpath(plots_dir, "weight1_vs_time_N$N_target.pdf"))")

        return
    end
end



# Run example:
N=22

# Create plots directory
plots_dir = "plots"
mkpath(plots_dir)

println(plot_weight_dist(N))
plot_average_for_N_multiple_M(N, savepath=joinpath(plots_dir, "pauli_avg_N$(N)_multiple_M.pdf"))
