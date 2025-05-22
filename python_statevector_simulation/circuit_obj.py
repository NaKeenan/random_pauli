import numpy as np
import functions as fn
from tqdm import tqdm


class Circuit:
    def __init__(self, N, gates, order):
        """
        Initialize the circuit object.
        - N is the number of qubits.
        - gates is either the list of gates or of parameters: 
            [circuit_type, geometry, Js] or [circuit_type, geometry]
            where if u1 is used, Js = [J, Jz] and if su2 is used, Js = J = Jz.
          if no Js is reported, the gates are set randomly.
        - initial_state is either the state or the parameters:
            [state_type, p, state_phases, theta]
        - order is the order of the gates.
        """
        self.N = N
        self.order = order
        self.gates = gates

    def run(self, masks_dict, state, T):
        """
        Run the circuit and calculate the mangetization
        """
        magn_profile = np.zeros((T+1, self.N), dtype=np.float64)
                
        magn_profile[0] = fn.get_magnetization(state, self.N)
        
        for t in (range(1,T+1)):
            
            state = fn.apply_U(state, self.gates, self.order, masks_dict, None)   
            
            magn_profile[t] = fn.get_magnetization(state, self.N)         

                
        return magn_profile
