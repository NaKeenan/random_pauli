using JSON

# Parameters
N = 14
filename = "params.json"

# Angle keys
angle_keys = ["θ1", "θ2", "Jx", "Jz", "θ3", "θ4"]

θ1, θ2 = 0, 0
Jx, Jz = 1, 1
θ3, θ4 = 0, 0

data = Dict("N" => N,
            "θ1" => θ1,
            "θ2" => θ2,
            "Jx" => Jx,
            "Jz" => Jz,
            "θ3" => θ3,
            "θ4" => θ4)

# Save to JSON file
open(filename, "w") do io
    JSON.print(io, data)  # no indent argument!
end

println("Data saved to $filename")
