import PauliStrings as ps
include("utils.jl")

function operator_test(N, trial, M=2^12)
    O = ps.Operator(N)
    O += "XZ"
    println("initial operator:")
    println(O)
    gate_list, angle_list = random_gate(N, trial)
    O = apply_gate_list(O, gate_list, angle_list; M=M)

    println("final operator:")
    return O 
end

println(operator_test(2, 1))