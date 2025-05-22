import PauliStrings as ps
include("utils.jl")

function operator_test(N, trial, M=2^12)
    O = ps.Operator(N)
    O += "Z", 1
    gate_list, angle_list = random_circuit(N, trial)
    O = apply_gate_list(O, gate_list, angle_list; M=M)
    return O 
end
