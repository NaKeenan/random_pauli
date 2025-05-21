import numpy as np
import functions as fn
import time

def print_time(t, T, start_time):
    end_time = time.time()
    if t % 100 == 0:
        elapsed =  - start_time
        avg_per_iter = elapsed / t
        remaining = avg_per_iter * (T - t)

        # format ETA as H:MM:SS
        eta_h = int(remaining) // 3600
        eta_m = (int(remaining) % 3600) // 60
        eta_s = int(remaining) % 60

        print(
            f"[{t}/{T}] "
            f"Elapsed: {elapsed:.1f}s, "
            f"ETA: {eta_h:d}:{eta_m:02d}:{eta_s:02d}"
        )
    return end_time


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
        magn_profile = np.zeros((T, self.N), dtype=np.float64)
        
        start_time = time.time()
        
        magn_profile[0] = fn.get_magnetization(state, self.N, self.operators)
        
        for t in range(1,T+1):
            
            state = fn.apply_U(state, self.gates, self.order, masks_dict, None)   
            
            magn_profile[t] = fn.get_magnetization(state, self.N, self.operators)         

            start_time = print_time(t, T, start_time)
                
        return magn_profile
