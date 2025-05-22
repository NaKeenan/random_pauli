using JLD2
using Glob
using Plots
using Statistics

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
        plot!(plt, 1:num_steps, avg_result, label="M = $M", yaxis=:log, xaxis=:log)
    end

    xlabel!(plt, "Time step")
    ylabel!(plt, "Average result")
    title!(plt, "Average results for N = $N_target, separated by M")
    savefig(plt, savepath)
    println("Plot saved to $savepath")
end

# Run example:
N=16
plot_average_for_N_multiple_M(N, savepath="pauli_avg_N$(N)_multiple_M.png")
