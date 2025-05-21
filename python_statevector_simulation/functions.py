import numpy as np
import scipy.linalg as la
import datetime, timeit
import random as rd
import numpy.random as nprd
import shutil
terminal_width, _ = shutil.get_terminal_size()
from functools import reduce
import pickle
from scipy.stats import unitary_group
from tqdm import tqdm
from scipy.linalg import fractional_matrix_power as fmp
from math import comb

from scipy.linalg import eigh
import warnings
from itertools import combinations
from sympy.physics.quantum.cg import CG
from sympy import S
from sympy import sympify
from scipy.linalg import expm
from functools import reduce


from numba import njit, prange#, config,

# set the threading layer before any parallel target compilation
# config.THREADING_LAYER = 'threadsafe'

def print_matrix(matr, precision=4):
    s = [[str(e) if abs(e) > 1e-15 else '.' for e in row] for row in np.round(matr,precision)]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens if x != 0) or '.'
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))

##########################################################################
# Timekeeping functions
##########################################################################

def begin(verbosity=1):
    if verbosity==2:
        print('\n')
        print('*' * terminal_width)
        print(' '*4, 'Started:', datetime.datetime.now())
    return None

def finish(verbosity=1):
    if verbosity==2:
        print(' '*4, 'Finished:', datetime.datetime.now())
        print('*' * terminal_width)
        print('\n')
    return None

def tic(task_description_string, verbosity=1):
    if verbosity==2:
        print('-' * terminal_width)
        print('---> ' + task_description_string)
    return timeit.default_timer()

def toc(tic_time, verbosity=1):
    toc_time = timeit.default_timer()
    if verbosity==2:
        print(' '*4, round(toc_time - tic_time, 6), 'seconds')
    return toc_time

#########################################################################
I = np.diag([1, 
               1])
X = np.array([[0,1],
              [1,0]])
Y = np.array([[0,-1j],
              [1j,0]])
Z = np.array([[1, 0],
              [0,-1]])
M = np.array([[0,1],
              [0,0]])
P = np.array([[0,0],
              [1,0]])
UP = np.array([1,0])
DOWN = np.array([0,1])

II = np.kron(I,I)
IX = np.kron(I,X)
XI = np.kron(X,I)
IZ = np.kron(I,Z)
ZI = np.kron(Z,I)
XX = np.kron(X,X)
YY = np.kron(Y,Y)
ZZ = np.kron(Z,Z)
PM = np.kron(P,M)
MP = np.kron(M,P)

#########################################################################

def get_masks(N, first_qubit, K=2):
    """
    Return an array of shape (2**K,) of index-arrays (masks), each selecting those
    computational-basis states (0..2**N-1) whose bits at positions
    first_qubit, first_qubit+1, ..., first_qubit+K-1 (mod N) equal one of the
    2**k possible patterns.

    Qubits are numbered from 0 (LSB) to N-1 (MSB).  A mask is the sorted list of
    integers whose binary representation has the specified K-bit pattern in those
    positions.
    
    K is the locality of the mask, i.e. the number of qubits in the window.
    """
    comp_basis = np.arange(2**N)
    masks = []

    # generate all k‐bit patterns 0..2**k-1
    for pattern in range(2**K):
        idx = comp_basis
        # for each qubit in the window, filter idx by whether that bit matches
        for offset in range(K):
            q = (first_qubit + offset) % N # qubit index
            want = (pattern >> offset) & 1 # get the bit of pattern at offset
            idx = idx[(idx // 2**q) % 2 == want]
        masks.append(idx)

    return np.array(masks, dtype=object)


def get_masks_typed(N, first_qubit, K):
    comp = np.arange(2**N, dtype=np.int64)
    D = 2**K
    M = (2**N) // D
    masks = np.empty((D, M), dtype=np.int64)

    for pat in prange(D):
        idx = comp
        for offset in range(K):
            q   = (first_qubit + offset) % N
            bit = (pat >> offset) & 1
            idx = idx[(idx // (1 << q)) % 2 == bit]
        masks[pat, :] = idx

    return masks

def apply_gate(state, gate, masks):
    '''
    Apply a gate to the state

    Parameters:
    - state: state vector on full Hilbert space
    - gate: 4x4 matrix corresponding to a 2-qubit gate
    - masks: list of 4 elements
        . first element contains the indices to treat as |00>
        . second element contains the indices to treat as |01>
        . third element contains the indices to treat as |10>
        . fourth element contains the indices to treat as |11>
    '''
    state_fin = np.zeros_like(state, dtype=complex)

    # Split the state in its four components
    state_split = state[masks]

    # Apply gate to state
    state_fin[masks] =  np.matmul(gate, state_split)[:,]

    return state_fin
    

@njit(parallel=True, fastmath=True, cache=True)
def apply_gate(state, gate, masks):
    '''
    Apply a gate to the state

    Parameters:
    - state: state vector on full Hilbert space
    - gate: 4x4 matrix corresponding to a 2-qubit gate
    - masks: list of 4 elements
        . first element contains the indices to treat as |00>
        . second element contains the indices to treat as |01>
        . third element contains the indices to treat as |10>
        . fourth element contains the indices to treat as |11>
    '''
    state_fin = np.zeros_like(state, dtype=np.complex128)
    num_elements = len(masks[0]) # 2^N/4
    for idx in prange(num_elements):
        i0, i1, i2, i3 = masks[0][idx], masks[1][idx], masks[2][idx], masks[3][idx]
        s0, s1, s2, s3 = state[i0], state[i1], state[i2], state[i3]
        t0 = gate[0,0]*s0 + gate[0,1]*s1 + gate[0,2]*s2 + gate[0,3]*s3
        t1 = gate[1,0]*s0 + gate[1,1]*s1 + gate[1,2]*s2 + gate[1,3]*s3
        t2 = gate[2,0]*s0 + gate[2,1]*s1 + gate[2,2]*s2 + gate[2,3]*s3
        t3 = gate[3,0]*s0 + gate[3,1]*s1 + gate[3,2]*s2 + gate[3,3]*s3
        state_fin[i0], state_fin[i1], state_fin[i2], state_fin[i3] = t0, t1, t2, t3
    return state_fin

@njit(parallel=True, fastmath=True, cache=True)
def apply_gate_k(state, gate, masks):
    """
    Apply a K-local gate to an N-qubit state vector.

    Parameters
    ----------
    state : complex128[2**N]
        The input state vector.
    gate : complex128[2**K, 2**K]
        The K-qubit gate to apply.
    masks : int64[2**K, M], M = 2**N / 2**K
        masks[c] is the array of all basis-state indices whose
        local K-bit pattern equals the integer c (0 ≤ c < 2**K).

    Returns
    -------
    state_out : 1D complex128 array, length = 2**N
        The output state, with the K-qubit gate applied in place.
    """
    D, M = masks.shape         # D = 2**K,  M = 2**N / 2**K
    out = np.zeros_like(state)

    for b in prange(M):
        # gather
        amp = np.empty(D, dtype=np.complex128)
        for c in range(D):
            amp[c] = state[masks[c, b]]

        # apply
        res = gate.dot(amp)

        # scatter
        for c in range(D):
            out[masks[c, b]] = res[c]

    return out

def apply_U(state, gates, gate_ordering_idx_list, masks_dict, K=None):
    '''
    Apply the Floquet operator to the state psi 2-qubit gate at a time

    Parameters:
    - state: state vector on full Hilbert space
    - gates: list of matrices. each is a 2-qubit gate
    - gate_ordering_idx_list: list of indeces correspoding to the
                              order the gates wil be applied:
                              eg. i -> gate_{i,i+1}
    - masks_dict: dictionary containing N masks defining how a gate on
                  2 consecutive sites needs to be applied
    '''
    for gate_idx, order_idx in enumerate(gate_ordering_idx_list):
        if K is not None:
            try:
                state = apply_gate_k(state, gates[gate_idx], masks_dict[order_idx])
            except:
                print('Error applying gate')
                print('len gates:', len(gates))
                print('gate:', gate_idx)
                print('list:', gate_ordering_idx_list)
                print('order:', order_idx)
                print('masks:', masks_dict[order_idx])
                raise
        else:
            state = apply_gate(state, gates[gate_idx], masks_dict[order_idx])
    return state

#########################################################################

def get_magn(state):
    '''
    Calculate the magnetization of the state
    '''
    N = int(np.log2(len(state)))
    M = 0
    for n in range(N):
        paulis = [I]*N
        paulis[n] = Z
        s_z_n = reduce(np.kron, paulis)
        M += (state.conj().T @ s_z_n @ state).real
    return M

def r_y(theta):
    return np.cos(theta/2)*I - 1j*np.sin(theta/2)*Y

def initial_entangled_state(N, theta, state_phases):
    """
    Generate the state |\psi> = \bigotimes_{i=0}^{N/2-1} |\Psi^+>_{(N/2 - i), (N/2 + i)}
    for an N-qubit system, where N is even.

    Parameters:
        N (int): Total number of qubits, must be even.

    Returns:
        numpy.ndarray: State vector of the entangled system.
    """
    def generate_initial_order(N):
        """Generate the initial qubit order [0, N-1, 1, N-2, ...]."""
        order = []
        for i in range(N // 2):
            order.append(i)
            order.append(N - 1 - i)
        if N % 2 == 1:
            order.append(N // 2)
        return order

    def resort_state_vector_general(state_vector, N):
        """Resort a state vector with N qubits from [0, N-1, 1, N-2, ...] to canonical [0, 1, 2, ...]."""
        num_states = len(state_vector)
        num_qubits = int(np.log2(num_states))
        
        assert num_qubits == N, "State vector size does not match qubit count."
        assert 2**N == num_states, "State vector size is not a power of 2."

        # Generate initial qubit order and its inverse mapping
        initial_order = generate_initial_order(N)

        # New state vector
        new_state_vector = np.zeros_like(state_vector, dtype=complex)

        # Permute indices
        for i in range(num_states):
            # Convert index to binary, rearrange bits, then convert back to integer
            binary = f"{i:0{N}b}"  # Binary representation with padding
            rearranged_binary = "".join(binary[initial_order.index(j)] for j in range(N))
            new_index = int(rearranged_binary, 2)
            new_state_vector[new_index] = state_vector[i]

        return new_state_vector

    states = []
    for n in range(N//2):
        if state_phases == 'homogenous':
            phase = 1
        elif state_phases == 'staggered':
            phase = (-1)**n
        st = np.sin(theta * phase)
        ct = np.cos(theta * phase)
        st_1 = np.sin(theta * phase/2)
        st_2 = np.sin(theta * phase/2)
        ct_2 = np.cos(theta * phase/2)
        # if n != 4:
        #     state = np.array([-st, ct, ct, st])/np.sqrt(2) # ry(theta)ry(theta) |Psi+>
        # else:
        state = np.array([ct, st, st, -ct])/np.sqrt(2) # ry(theta)ry(theta) |Phi->, not preserving M
            # state = np.array([ct_2**2, st_1/2, st_1/2, st_2**2]) # ry(theta)ry(theta) |00>

        states.append(state)

    unsorted_state = reduce(np.kron, states)
    return resort_state_vector_general(unsorted_state, N)

def initial_state(N, qkeep, theta, state_phases):
    '''
    Create the initial state as (exp(-i theta/2 sigma_y) |0>)^\otimes N
    '''

    states_n = []
    for n in range(N):
        if n not in qkeep:
            theta_ = .0 * np.pi #np.pi/2
        else:
            theta_ = theta
        if state_phases == 'homogenous':
            state_n = r_y(theta_) @ np.array([1, 0])
        elif state_phases == 'staggered':
            state_n = r_y(theta_) @ (np.array([1, 0]) if n%2==0 else np.array([0, 1]))
        
        states_n.append(state_n)

    state = reduce(np.kron, states_n)

    return state

def ptrace(rho, qkeep):
    N = int(np.log2(rho.shape[0]))
    rd = [2,] * N
    qkeep = list(np.sort(qkeep))
    dkeep = list(np.array(rd)[qkeep])
    qtrace = list(set(np.arange(N))-set(qkeep))
    dtrace = list(np.array(rd)[qtrace])
    if len(rho.shape) == 1: # if rho is ket
        temp = (rho
                .reshape(rd) # convert it to 2x2x2x...x2
                .transpose(qkeep+qtrace) # leave sites to trace as last
                .reshape([np.prod(dkeep),np.prod(dtrace)])) # dkeep x dtrace 
        partial_rho = temp.dot(temp.conj().T) 
    else : # if rho is density matrix
        partial_rho = np.trace(rho
                      .reshape(rd+rd)
                      .transpose(qtrace+[N+q for q in qtrace]+qkeep+[N+q for q in qkeep])
                      .reshape([np.prod(dtrace),np.prod(dtrace),
                                np.prod(dkeep),np.prod(dkeep)]))
    return partial_rho

def gen_u1(params=None):
    if params is not None:
        if len(params) == 2: params += [0, 0, 0]
        return gate_xxz_disordered(*params)
    gate = np.zeros((4,4), dtype=complex)
    gate[0,0] = np.exp(1j*np.random.rand()*2*np.pi)
    gate[3,3] = np.exp(1j*np.random.rand()*2*np.pi)
    gate[1:3,1:3] = unitary_group.rvs(2)
    return gate

def gen_su2(J=None):
    if J is None:
        J = np.random.rand()*np.pi
    swap = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]])
    
    gate = np.eye(4) * np.cos(J/2) - 1j * np.sin(J/2) * swap
    return gate

def gen_gates_order(N, geometry='brickwork', boundary_conditions='PBC', eo_first='True'):
    # Generate the order the gates will be applied
    if geometry == 'random':
        if boundary_conditions == 'PBC':
            return rd.sample([n for n in range(N)],N)
        elif boundary_conditions == 'OBC':
            return rd.sample([n for n in range(N-1)],N-1)
    if geometry != 'brickwork':
        raise ValueError('Only random and brickwork geometries are supported')
    gate_ordering_idx_list = []
    if eo_first:
        for n in range(N):
            if n % 2 == 0:
                if n == N-1 and boundary_conditions == 'OBC':
                    continue
                else:
                    gate_ordering_idx_list.append(n)
    for n in range(N):
        if n % 2 == 1:
            if n == N-1 and boundary_conditions == 'OBC':
                continue
            else:
                gate_ordering_idx_list.append(n)
    if not eo_first:
        for n in range(N):
            if n % 2 == 0:
                if n == N-1 and boundary_conditions == 'OBC':
                    continue
                else:
                    gate_ordering_idx_list.append(n)

    return np.array(gate_ordering_idx_list, dtype=int)

def vNE(rho):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-10]


    return -np.sum(eigvals*np.log2(eigvals))


def gen_Ls(Ns, circuit_type):
    if circuit_type == 'u1':
        L = np.zeros((2**Ns, 2**Ns), dtype=complex)
        for n in range(Ns):
            paulis = [I]*Ns
            paulis[n] = Z
            L += reduce(np.kron, paulis)
        return [L]
    elif circuit_type == 'su2':
        Ls = []
        for PAULI in [X, Y, Z]:
            for n in range(Ns):
                L = np.zeros((2**Ns, 2**Ns), dtype=complex)
                paulis = [I]*Ns
                paulis[n] = PAULI
                L += reduce(np.kron, paulis)
            Ls.append(L)
        return Ls

def WY(rho_A, Ls):
    # compute Wigner-Yanase skew information
    rho_A_sqrt = la.sqrtm(rho_A)
    WYs = []
    for L in Ls:
        WYs.append(
            np.trace(L @ rho_A @ L) - np.trace(rho_A_sqrt @ L @ rho_A_sqrt @ L))
    if len(WYs) == 1:
        return WYs[0].real
    return np.array(WYs).real

def gen_QFI(rho_A, Ls, ss):
    # compute QFI information
    # ss is a list of parameters s, where s=0 for SLD and s=1 for RLD and s=.5 for WY
    rho_A_sqrt = la.sqrtm(rho_A)
    QFIs = np.zeros((len(Ls), len(ss)))
    for Lidx, L in enumerate(Ls):
        LrhoL = np.trace(L @ rho_A @ L)
        for sidx, s in enumerate(ss):
            rhoLrhoL = np.trace(fmp(rho_A, s) @ L @ fmp(rho_A, 1-s) @ L)
            QFIs[Lidx, sidx] = (LrhoL - rhoLrhoL).real
    return QFIs

def load_mask_memory(N, K=2):
    '''
    Load the mask memory for a given N and K
    '''
    mask_dict = {}
    for n in range(N):
        mask_dict[n] = get_masks_typed(N, n, K)
    return mask_dict

def gen_Q(N, Ns=None):
    '''
    Generate the Q matrix composed by the projectors on the sectors of different
    magnetization values
    '''
    # if f'N{N}.pkl' in os.listdir('mask_memory'):
    #     DB = pickle.load(open(f'mask_memory/N{N}.pkl', 'rb'))
    #     mask_dict = DB['mask_dict']
    #     states_per_sector = DB['states_per_sector']
    #     qs = DB['qs']
    #     if Ns is None:
    #         return mask_dict
    #     Q = DB['Q']
    #     return mask_dict, qs, states_per_sector, Q
    
    if Ns is None:
        print('Ns must be specified if the mask memory is not available yet')
        mask_dict = {}
        for n in range(N):
            mask_dict[n] = get_masks(N, n)
        return mask_dict
    
    mask_dict = {}
    for n in range(N):
        mask_dict[n] = get_masks(N, n)
        
    computational_basis = np.arange(2**Ns)
    basis = np.array([bin(i).count('1') for i in computational_basis], dtype=int)

    states_per_sector = {}
    Q = np.zeros((2**Ns, 2**Ns), dtype=complex)    
    qs = []
    for M_A in range(Ns+1):
        temp_comp_states = computational_basis[basis == M_A]
        states_per_sector[M_A] = temp_comp_states
        vector = np.zeros(2**Ns)
        vector[temp_comp_states] = 1
        qs.append(np.outer(vector, np.conj(vector)))

    Q = reduce(np.add, qs)

    data_to_save = {'mask_dict': mask_dict,
                    'qs': qs,
                    'states_per_sector': states_per_sector,
                    'Q': Q}   

    with open(f'mask_memory/N{N}.pkl', 'wb') as file:
        pickle.dump(data_to_save, file)

    return mask_dict, qs, states_per_sector, Q

@njit(parallel=True, fastmath=True, cache=True)
def apply_u1(state, gate, masks):
    '''
    Apply a gate to the state

    Parameters:
    - state: state vector on full Hilbert space
    - gate: 4x4(1x1,2x2,1x1) matrix corresponding to a 2-qubit gate
    - masks: list of 4 elements
        . first element contains the indices to treat as |00>
        . second element contains the indices to treat as |01>
        . third element contains the indices to treat as |10>
        . fourth element contains the indices to treat as |11>
    '''
    state_fin = np.zeros_like(state, dtype=np.complex128)
    num_elements = len(masks[0]) # 2^N/4
    for idx in prange(num_elements):
        i0, i1, i2, i3 = masks[0][idx], masks[1][idx], masks[2][idx], masks[3][idx]
        s0, s1, s2, s3 = state[i0], state[i1], state[i2], state[i3]
        state_fin[i0] = gate[0,0]*s0
        state_fin[i1] = gate[1,1]*s1 + gate[1,2]*s2
        state_fin[i2] = gate[2,1]*s1 + gate[2,2]*s2
        state_fin[i3] = gate[3,3]*s3
    return state_fin

@njit(parallel=True, fastmath=True, cache=True)
def apply_su2(state, gate, masks):
    '''
    Apply a gate to the state

    Parameters:
    - state: state vector on full Hilbert space
    - gate: 4x4(1x1,1x1,1x1,1x1) matrix corresponding to a 2-qubit gate
    - masks: list of 4 elements
        . first element contains the indices to treat as |00>
        . second element contains the indices to treat as |01>
        . third element contains the indices to treat as |10>
        . fourth element contains the indices to treat as |11>
    '''
    state_fin = np.zeros_like(state, dtype=np.complex128)
    num_elements = len(masks[0]) # 2^N/4
    for idx in prange(num_elements):
        i0, i1, i2, i3 = masks[0][idx], masks[1][idx], masks[2][idx], masks[3][idx]
        state_fin[i0] = gate[0,0]*state[i0]
        state_fin[i1] = gate[1,2]*state[i1]
        state_fin[i2] = gate[2,1]*state[i2]
        state_fin[i3] = gate[3,3]*state[i3]
    return state_fin

def gate_xyz_disordered(h1, h2, Jx, Jy, Jz, h3, h4):
    """Return the unitary matrix for the disordered XXZ model."""
    U_H1 = np.diag(np.exp(-.5j*np.array([h1+h2, h1-h2, h2-h1, -h1-h2])))
    U_H2 = np.diag(np.exp(-.5j*np.array([h3+h4, h3-h4, h4-h3, -h3-h4])))
    U_XX = II * np.cos(Jx) - 1.0j * XX * np.sin(J)
    U_YY = II * np.cos(Jy) - 1.0j * YY * np.sin(J)
    U_ZZ = II * np.cos(Jz) - 1.0j * ZZ * np.sin(Jz)
    U_XXZ = U_XX @ U_YY @ U_ZZ
    return U_H1 @ U_XXZ @ U_H2

def gate_xxz_disordered(J, Jz, h1, h2, phi):
    ''' phase diagram [0,Pi] x [0,Pi]
    SWAP at J = pi
    '''
    U_H1 = np.diag(np.exp(-.5j*np.array([h1+h2, h1-h2, h2-h1, -h1-h2])))
    U_PM_MP = la.expm(-1j * J/2 * (PM * np.exp(1.0j * phi) + \
                                   MP * np.exp(-1.0j * phi)))
    U_ZZ = II * np.cos(Jz/4) - 1j * ZZ * np.sin(Jz/4) 
    U_XXZ = U_PM_MP @ U_ZZ
    return U_H1 @ U_XXZ

def gen_MagMask(masks_dict):
    '''generate mask for magnetization calculation'''
    N = len(masks_dict)
    magn_mask_is = np.zeros((N, 2**N))
    for i in range(N):
        masks = masks_dict[N - i - 1]
        qubit_up_mask = masks[0].tolist() + masks[1].tolist()
        magn_mask_i = np.zeros(2**N)
        magn_mask_i[qubit_up_mask] = 1
        magn_mask_is[i] = 2*magn_mask_i-1
    return magn_mask_is


def gen_Jz(N):
    """
    Generate the Jz operator for N spins.
    """ 
    id_ = np.diag([1,1])
    sz = np.diag([1,-1])
    Jz = np.zeros((2**N, 2**N), dtype=np.complex128)
    for i in range(N):
        ops = [id_] * N
        ops[i] = sz
        op = reduce(np.kron, ops)
        Jz += op
    return Jz

def get_magnetization(st, N, operators=None):
    id_ = np.diag([1,1])
    sz = np.diag([1,-1])
    res = np.zeros(N, dtype=np.float64)
    for i in range(N):
        if operators is None:
            ops = [id_] * N
            ops[i] = sz
            op = reduce(np.kron, ops)
        else:
            op = operators[i]
        res[i] = st.conj().dot(op @ st)
    return np.array(res)


@njit(parallel=True, fastmath=True, cache=True)
def compute_magn(psi_2, magn_mask_is):
    N = int(np.log2(len(psi_2)))
    magn_is = np.zeros(N)
    for i in prange(N):
        magn_is[i] = np.dot(psi_2, magn_mask_is[i]).real
    return magn_is

def compute_single_trajectory(gates, order, psi_0, T, masks_dict, magn_mask_is):
    spin_densities_list = []
    psi_2 = psi_0.conj() * psi_0    
    magnetization = compute_magn(psi_2, magn_mask_is)
    spin_densities_list.append(magnetization)
    N = len(order)

    for t in range(T):
        if t < N:

            for order_idx, gate_idx in enumerate(order):
                psi_0 = apply_gate(psi_0, gates[order_idx], masks_dict[gate_idx])
                psi_2 = psi_0.conj() * psi_0 
                magnetization = compute_magn(psi_2, magn_mask_is)   
                spin_densities_list.append(magnetization)
        else:
            psi_0 = apply_U(psi_0, gates, order, masks_dict)
            psi_2 = psi_0.conj() * psi_0    
            magnetization = compute_magn(psi_2, magn_mask_is)
            spin_densities_list.append(magnetization)

    return np.array(spin_densities_list)

def gen_t_list(N, T):
    '''generate time list
    First N steps computed gate by gate
    Then stroboscopic steps computed at each whole time step up to T
    '''
    additional_steps = (np.arange(N).reshape(N, 1) + 
                    (np.arange(1, N + 1) / N)
                    ).flatten()
    stroboscobic = np.arange(N,T)
    return np.concatenate(([0], additional_steps, stroboscobic))

def gen_initial_state(masks_dict, rnd_seed=None):
    '''generate initial state
    |0>^{\otimes N} with the first qubit up

    verify it with 
        psi_2 = psi_0.conj() * psi_0    
        fn.compute_magn(psi_2, magn_mask_is)
    '''
    N = len(masks_dict)
    if rnd_seed is not None:
        nprd.seed(rnd_seed)
    psi_0 = nprd.uniform(0,1,2**N) + 1.0j*nprd.uniform(0,1,2**N)
    qubit_0_down_mask = masks_dict[0][2].tolist() + masks_dict[0][3].tolist()
    psi_0[qubit_0_down_mask] = 0
    return psi_0 / np.sqrt(np.dot(psi_0.conj(), psi_0))

def apply_single_z(psi_0, masks_dict):
    psi = psi_0.copy()
    N = len(masks_dict)
    for i in range(0, N, 2):
        h1, h2 = nprd.uniform(-np.pi, np.pi, 2)
        ZZ_random = np.diag(np.exp(-.5j*np.array([h1+h2, h1-h2, h2-h1, -h1-h2])))
        psi = apply_gate(psi, ZZ_random, masks_dict[i])
    return psi

# def compute_single_trajectory_with_trick(params, order, psi_0, T, 
#                                          disorder_realizations, masks_dict, magn_mask_is):
#     N = len(order)
#     numb_steps = (T-N) + N**2 + 1
#     spin_evol = np.zeros((disorder_realizations, numb_steps, N))

#     for realization in range(disorder_realizations):
#         psi_r = apply_single_z(psi_0, masks_dict)
#         psi_2 = psi_r.conj() * psi_r    
#         magnetization = compute_magn(psi_2, magn_mask_is)
#         spin_evol[realization, 0, :] = magnetization

#     J, Jz = params

#     for t in tqdm(range(T)):
#         if t < N:
#             phis = nprd.uniform(-np.pi, np.pi, N)
#             hs = nprd.uniform(-np.pi, np.pi, 2*N)
#             gates = np.array([gate_xxz_disordered(J, Jz, hs[2*i], 
#                                                   hs[2*1+1], phis[i]) for i in range(N)])
            
#             for order_idx, gate_idx in enumerate(order):
#                 psi_0 = apply_gate(psi_0, gates[order_idx], masks_dict[gate_idx])

#                 for realization in range(disorder_realizations):
#                     psi_r = apply_single_z(psi_0, masks_dict)
#                     psi_2 = psi_r.conj() * psi_r    
#                     magnetization = compute_magn(psi_2, magn_mask_is)
#                     spin_evol[realization, t*N + order_idx, :] = magnetization
#         else:
#             for realization in range(disorder_realizations):
#                 psi_r = apply_single_z(psi_0, masks_dict)
#                 psi_2 = psi_r.conj() * psi_r    
#                 magnetization = compute_magn(psi_2, magn_mask_is)
#                 spin_evol[realization, (t-N) + N**2, :] = magnetization

#     return spin_evol

def gen_PS():
    '''Generate a random product state on one qubit'''
    state = np.random.rand(2) + 1j * np.random.rand(2)
    return state / np.linalg.norm(state)

def initial_mixed_state(N, theta, state_phases, p):
    '''Generate a random product state on N qubits'''
    # Generate id + p Product State
    paulis = []
    for n in range(N):
        if np.random.rand() < p:
            if state_phases == 'homogenous':
                phase = 1
            elif state_phases == 'staggered':
                phase = (-1)**n
            state = r_y(phase*theta) @ np.array([1, 0])
            paulis.append(state)
        else:
            if np.random.rand() < .5:
                paulis.append(UP)
            else:
                paulis.append(DOWN)
    return reduce(np.kron, paulis)

def is_bad_value(coeff):
    return coeff == 0 or np.isnan(coeff) or np.isinf(coeff) or not np.isfinite(coeff)

def generic_coherence_measure(rho, eigvals, eigvecs, Ls, f):
    '''From eq.1.44 of
    Irénée Frerot. A quantum statistical approach to quantum correlations in many-body systems. 
    Statistical Mechanics [cond-mat.stat-mech]. Université de Lyon, 2017. 
    
    f is a standard monotone function.
    the mixed state is assumed to be 
                ρ = p |ψ⟩⟨ψ| + (1 - p) 𝕀/d
    where |psi> is the tilted state and Id/d is the maximally mixed state:
            |ψ(θ)⟩ = e^(-iθ/2 ∑ₖ σₖʸ) |000...0⟩
            
    ρ_A = Tr_{B} ρ
    '''
    Coherent = np.zeros(len(Ls))
    # Op = np.zeros(len(Ls))
    
    # if f == f_SLD:
    #     sigma = L @ rho @ L
    #     rho_1 = la.inv(rho)
    #     Op = np.trace(rho @  L @ rho_1 @ L @ rho - rho @ sigma @ rho_1 - rho + L @ rho @ rho @ L @ rho_1)
    
    # if f == f_WY:
    #     Op = WY(rho, Ls)
        
    # if f == f_rel_ent:
    #     sigma = L @ rho @ L
    #     log_rho = la.logm(rho) # stable_logm(rho)
    #     log_sigma = la.logm(sigma)
    #     Op = np.trace(sigma @ log_sigma) - np.trace(sigma @ log_rho)
    
    # if f == f_q_info_var:
    # ### TO BE COMPLETED
    
    # if f == f_geo_mean:
    # ### TO BE COMPLETED
    
    # if f == f_harm_mean:
    #     rho_1 = la.inv(rho)
    #     Op = (np.trace(L @ rho @ rho @ L @ rho_1) - np.trace(L @ rho @ L))/2
        
    eigvals[eigvals < 1e-12] = 0
        
    indexes = np.argsort(eigvals)
    eigvals = eigvals[indexes]
    eigvecs = eigvecs.T[indexes]
            
    for i, eigval_i in enumerate(eigvals): # Apply L on the state
        Ls_eigvec_i = [L @ eigvecs[i] for L in Ls]
        
        for j, eigval_j in enumerate(eigvals):
            if i==j: continue
            try:
                coeff = ((eigval_i - eigval_j)**2/(eigval_i*f((eigval_j/(eigval_i))))) 
            except ZeroDivisionError:
                pass
            if is_bad_value(coeff): 
                try:
                    coeff = ((eigval_j - eigval_i)**2/(eigval_j*f((eigval_i/(eigval_j))))) 
                except ZeroDivisionError:
                    pass
                if is_bad_value(coeff): 
                    continue
        
            Coherent += [coeff * np.abs(eigvecs[j].conj().dot(Li))**2 for Li in Ls_eigvec_i]
                
    if not (f(0) == 0 or np.isnan(f(0))): Coherent *= f(0)/2 # Removed for the QFIs with f(0) = 0
    else: Coherent *= 1/2
            
    return Coherent

def compute_incoherent_fisher_info(p_t, p_t_minus_dt, dt):
    epsilon = 1e-12  # Small constant to avoid division by zero
    # Step 1: Compute dp/dt using backward difference
    dp_dt = (p_t - p_t_minus_dt) / dt
    # Step 2: Compute d/dt(log(p)) = dp/dt / p
    log_p_derivative = dp_dt / (p_t + epsilon)
    # Step 3: Compute the weighted sum for F_Q^IC
    F_Q_IC = np.sum(p_t * log_p_derivative**2)
    return F_Q_IC


def f_SLD(x): # Bures metric: f(x) = (x + 1) / 2
    return (x + 1) / 2 # y * f(x/y) = y * (x/y + 1) / 2 = (x + y) / 2

def f_Heinz(x, r): # Heinz family:: f(x) = (x^r + x^(1-r)) / 2
    return (x**r + x**(1 - r)) / 2

def f_ALPHA(x, alpha): # Alpha-divergency: f(x) = α(α - 1)(x - 1)^2 / ((x - x^α)(x^α - 1))
    numerator = alpha * (alpha - 1) * (x - 1)**2
    denominator = (x - x**alpha) * (x**alpha - 1)
    return numerator / denominator

def f_WY(x): # Wigner-Yanase metric: f(x) = (1/4) * (1 + sqrt(x))^2
    return (1 / 4) * (1 + np.sqrt(x))**2 # (pi-pj)/((1/4) * (pi+pj)^2/pi) = 4pi(pi-pj)/(pi+pj)^2 != sqrt(pi*pj)

def f_rel_ent(x): # Relative entropy: f(x) = (x - 1) / log(x)
    return (x - 1) / np.log(x)

def f_q_info_var(x): # Quantum information variance: f(x) = (2 * (x - 1)^2) / ((x + 1) * (log(x))^2)
    numerator = 2 * (x - 1)**2
    denominator = (x + 1) * (np.log(x)**2)
    return numerator / denominator

def f_geo_mean(x): # Geometric mean: f(x) = sqrt(x)
    return np.sqrt(x)

def f_harm_mean(x): # Harmonic mean: f(x) = (2 * x) / (x + 1)
    return (2 * x) / (x + 1)


def find_crossing_times(x_vals, y_vals1, y_vals2):
    """
    Find the crossing times between y_vals1 and y_vals2 by interpolation.
    
    Parameters:
    x_vals (np.ndarray): Array of x values.
    y_vals1 (np.ndarray): Array of y values for the first function.
    y_vals2 (np.ndarray): Array of y values for the second function.
    
    Returns:
    np.ndarray: Array of x values where the two functions cross.
    """
    # Compute the difference between the y values
    diff = y_vals1 - y_vals2
    
    # Find the indices where the sign of the difference changes
    crossing_indices = np.where(np.diff(np.sign(diff)))[0]
    
    # Interpolate to find the exact crossing points
    crossing_times = []
    if len(crossing_indices) == 0:
        return np.array(crossing_times)
    for idx in crossing_indices:
        x1, x2 = x_vals[idx], x_vals[idx + 1]
        y1, y2 = diff[idx], diff[idx + 1]
        crossing_time = x1 - y1 * (x2 - x1) / (y2 - y1)
        crossing_times.append(crossing_time)
    
    return np.array(crossing_times)

def compute_unitary(gates, order, masks_dict, N):
    '''
    Compute the unitary of the circuit
    '''
    # since the gates are applied on states apply the gates to the identity matrix, column by column sending them to the apply_U function
    U = np.eye(2**N, dtype=complex)
    for i in tqdm(range(2**N)):
        U[:, i] = apply_U(U[:, i], gates, order, masks_dict)
    return U

def compute_hamiltonian(gates, order, masks_dict, N):
    '''
    Compute the Hamiltonian of the circuit
    '''
    U = compute_unitary(gates, order, masks_dict, N)
    # diagonalize U
    eigvals, eigvecs = np.linalg.eig(U)
    # compute the log of the eigenvalues
    log_eigvals = np.log(eigvals)
    return eigvecs.T.conj() @ log_eigvals @ eigvecs


@njit(parallel=True, fastmath=True, cache=True)
def compute_projector(Ns, states):
    """
    Computes the projector onto the subspace spanned by the computational basis 
    states in the list 'states'. Each state is assumed to be an integer corresponding 
    to the basis index.
    """
    dim = 2**Ns
    P = np.zeros((dim, dim), dtype=np.complex128)
    for state_1 in states:
        v_1 = np.zeros(dim, dtype=np.complex128)
        v_1[state_1] = 1.0
        for state_2 in states:
            v_2 = np.zeros(dim, dtype=np.complex128)
            v_2[state_2] = 1.0
            # Add the projector for this state
            P += np.outer(v_1, v_2)
    # normalize
    P /= np.linalg.norm(P)
    return P

@njit(parallel=True, fastmath=True, cache=True)
def compute_projector(Ns, states):
    """
    Computes the projector onto span{|s⟩ : s in states}.
    states should be a 1D np.int64 array of basis indices.
    """
    dim = 1 << Ns           # 2**Ns
    P   = np.zeros((dim, dim), dtype=np.complex128)
    n   = states.shape[0]   # number of basis states

    # Fill P[s1,s2] = 1 for all s1,s2 in states
    for i in prange(n):
        s1 = states[i]
        for j in prange(n):
            s2 = states[j]
            P[s1, s2] = 1.0

    # The Frobenius norm of this matrix is n, so normalize by n
    return P / n

def manual_U1_tw(rho, projectors):
    '''
    Apply the twirling operation to the density matrix rho.
    The twirling operation is a sum over the projectors, weighted by the density matrix.
    If ordered is True, the projectors are applied on the reordered basis.
    '''
    P = np.array([Pj / np.max(Pj) for Pj in projectors.values()])  # Shape (N, d, d)
    
    return np.sum(P * rho, axis=0)


''' Checking the consistency of S(rho || G(rho)) == S(G(rho)) - S(rho) 
<\

import scipy.special  # For binomial coefficient

state = fn.initial_state(N, sites_to_keep, .2 * np.pi, state_phases)
state /= np.linalg.norm(state)
# apply a U
h_list = np.random.uniform(-np.pi, np.pi, 5*N).reshape(N, 5) /alpha
gates = [fn.gen_u1([*h]) for h in h_list]
order = fn.gen_gates_order(N, geometry=geometry)
state = fn.apply_U(state, gates, order, masks_dict)
pstate = fn.ptrace(state, sites_to_keep)
##############################################################################
pstateQ = fn.twirling(pstate, projectors)    
    
reordered_pstate = basis_reordering.T @ pstate @ basis_reordering
reordered_pstateQ = basis_reordering.T @ pstateQ @ basis_reordering
    
from scipy.linalg import logm, expm

def vNentropy(x): return - x @ logm(x)
def idfunction(x): return x 

A = - basis_reordering.T @ pstateQ @ logm(pstateQ) @ basis_reordering
B = fn.operation_per_block(reordered_pstate, vNentropy, Ns)
C = fn.twirling(- basis_reordering.T @ pstate @ logm(pstateQ) @ basis_reordering, reordered_projectors)
C = (- basis_reordering.T @ pstate @ logm(pstateQ) @ basis_reordering)
C = (- pstate @ logm(pstateQ))

fn.print_matrix(A, 2)
fn.print_matrix(B, 2)
fn.print_matrix(C, 2)
np.trace(A), np.trace(B), np.trace(C)

>
'''


##### Relative entropy
warnings.simplefilter("ignore", category=UserWarning)

def _safe_logm(mat: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Compute log(mat) by eigen‑decomposition, clamping eigenvalues to [epsilon, ∞).
    """
    vals, vecs = eigh(mat)
    # clamp eigenvalues away from zero
    safe_vals = np.clip(vals, epsilon, None)
    log_vals  = np.log(safe_vals)
    return (vecs * log_vals) @ vecs.conj().T

def _safe_frac_power(mat: np.ndarray, power: float, epsilon: float) -> np.ndarray:
    """
    Compute mat**power by eigen‑decomposition, clamping eigenvalues to [epsilon, ∞).
    """
    vals, vecs = eigh(mat)
    safe_vals = np.clip(vals, epsilon, None)
    frac_vals = safe_vals**power
    return (vecs * frac_vals) @ vecs.conj().T

def renyi_divergence(
    rho: np.ndarray,
    sigma: np.ndarray,
    alpha: float = 1.0,
    epsilon: float = 1e-12,
) -> float:
    """
    Computes D_α(ρ || σ) with spectrum‑level regularization to avoid Infs/NaNs.

    Parameters
    ----------
    rho, sigma : np.ndarray
        Density matrices (Hermitian, trace 1).
    alpha : float
        Rényi parameter (α > 0, α ≠ 1 normally; α→1 gives KLD).
    epsilon : float
        Clamping floor for all eigenvalues.

    Returns
    -------
    float
        The Rényi divergence D_α(ρ || σ).
    """
    # basic checks
    if alpha <= 0:
        raise ValueError("α must be > 0.")
    t_rho = np.trace(rho)
    t_sig = np.trace(sigma)
    if not np.allclose(t_rho, 1, atol=1e-6) or not np.allclose(t_sig, 1, atol=1e-6):
        raise ValueError("Both ρ and σ must have trace 1: "
                         f"tr(ρ) = {t_rho}, tr(σ) = {t_sig}.")

    # α → 1 → Kullback-Leibler
    if np.isclose(alpha, 1.0):
        log_rho   = _safe_logm(rho,   epsilon)
        log_sigma = _safe_logm(sigma, epsilon)
        # D = tr[ρ (log ρ − log σ)]
        D = np.real_if_close(np.trace(rho @ (log_rho - log_sigma)))
        return float(D)

    # α → 0 limit
    if np.isclose(alpha, 0.0):
        # support projector of ρ
        vals_rho, _ = eigh(rho)
        support = (vals_rho > epsilon).astype(float)
        vals_sig, _ = eigh(sigma)
        return -np.log(np.sum(support * np.clip(vals_sig, epsilon, None)))

    # enforce α ≤ 2 if desired
    if alpha > 2:
        raise ValueError("α must be ≤ 2 for this implementation.")

    # general α ≠ 1
    rho_a = _safe_frac_power(rho,   alpha,     epsilon)
    sig_b = _safe_frac_power(sigma, 1 - alpha, epsilon)
    trace_term = np.trace(rho_a @ sig_b)

    # guard against tiny negatives or Infs
    trace_term = np.real_if_close(trace_term)
    trace_term = float(np.clip(trace_term, epsilon, None))

    return (1.0 / (alpha - 1.0)) * np.log(trace_term)

def renyi_divergence_sym(
    rho: np.ndarray,
    symmetry: str,
    alpha: float = 1.0,
    epsilon: float = 1e-12,
    K = None,
    Ubasis=None
) -> float:
    """
    Computes D_α(ρ || G(ρ)) with spectrum‑level regularization to avoid Infs/NaNs.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix (Hermitian, trace 1).
    symmetry : str
        Symmetry group for the twirling operation (e.g., 'U1', 'SU2').
    alpha : float
        Rényi parameter (α > 0, α ≠ 1 normally; α→1 gives KLD).
    epsilon : float
        Clamping floor for all eigenvalues.

    Returns
    -------
    float
        The Rényi divergence D_α(ρ || σ).
    """
    # basic checks
    if alpha <= 0:
        raise ValueError("α must be > 0.")
    if symmetry not in ['U1']:
        raise ValueError("symmetry must be 'U1'.")
    Ns = int(np.log2(rho.shape[0]))
    if Ubasis is not None:
        U_basis = Ubasis
    else:
        if symmetry == 'U1':
            projectors, U_basis = build_projectors(Ns)
        else:
            raise ValueError("Invalid symmetry group. Choose 'U1'.")
        
    if symmetry == 'U1':           
        sigma = manual_U1_tw(rho, projectors)
        
    rho = U_basis.conj().T @ rho @ U_basis
    sigma = U_basis.conj().T @ sigma @ U_basis
    t_rho = np.trace(rho)
    t_sig = np.trace(sigma)
    
    if not np.allclose(t_rho, 1, atol=1e-6) or not np.allclose(t_sig, 1, atol=1e-6):
        raise ValueError("Both ρ and σ must have trace 1: "
                         f"tr(ρ) = {t_rho}, tr(σ) = {t_sig}.")

    # α → 1 → Kullback-Leibler
    if np.isclose(alpha, 1.0):
        log_rho   = _safe_logm(rho,   epsilon)
        log_sigma = _safe_logm(sigma, epsilon)
        log_sigma_basis = U_basis @ log_sigma @ U_basis.conj().T
        if symmetry == 'U1':           
            log_sigma_basis_tw = manual_U1_tw(log_sigma_basis, projectors)
        log_sigma = U_basis.conj().T @ log_sigma_basis_tw @ U_basis
        # D = tr[ρ (log ρ − log σ)]
        D = np.real_if_close(np.trace(rho @ (log_rho - log_sigma)))
        return float(D)

    # α → 0 limit
    if np.isclose(alpha, 0.0):
        # support projector of ρ
        vals_rho, _ = eigh(rho)
        support = (vals_rho > epsilon).astype(float)
        vals_sig, _ = eigh(sigma)
        return -np.log(np.sum(support * np.clip(vals_sig, epsilon, None)))

    # enforce α ≤ 2 if desired
    if alpha > 2:
        raise ValueError("α must be ≤ 2 for this implementation.")

    # general α ≠ 1
    rho_a = _safe_frac_power(rho,   alpha,     epsilon)
    sig_b = _safe_frac_power(sigma, 1 - alpha, epsilon)
    trace_term = np.trace(rho_a @ sig_b)

    # guard against tiny negatives or Infs
    trace_term = np.real_if_close(trace_term)
    trace_term = float(np.clip(trace_term, epsilon, None))

    return (1.0 / (alpha - 1.0)) * np.log(trace_term)


def max_divergence(rho: np.ndarray, sigma: np.ndarray, epsilon=1e-10) -> float:
    """
    Computes the max divergence D_infty(rho || sigma).
    """
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.maximum(eigvals, epsilon)  # Regularization

    sqrt_sigma_inv = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
    sandwiched_matrix = sqrt_sigma_inv @ rho @ sqrt_sigma_inv
    lambda_max = np.max(np.linalg.eigvalsh(sandwiched_matrix))
    
    return np.log(lambda_max)



def decompose_rho_modes(rho, Ub, Op):
    """
    Decomposes the density matrix 'rho' into its frequency (charge difference) modes.
    
    Parameters:
    -----------
    rho : ndarray
        The density matrix in the original (computational) basis.
    Ub : ndarray
        The unitary transformation matrix that rotates the computational basis into the 
        eigenbasis of J_z or J^2 (sorted from the lowest to the highest eigenvalue).
    Op : ndarray
        The magnetization operator or the total angular momentum operator J^2.
        
    Returns:
    --------
    freq_modes : dict
        A dictionary where keys are integer frequency modes (charge differences) and 
        the values are matrices of the same shape as the rotated density matrix
        containing the part of rho associated with that frequency.
    """
    # Rotate the density matrix into the J^2-eigenbasis.
    # In this basis, U_b.T @ J^2 @ U_b is diagonal.
    rho_rot = Ub.conj().T @ rho @ Ub
    
    # The diagonal of the rotated J^2 is the sorted charge vector.
    # (We assume here that this product is exactly diagonal; in numerical code, you might
    # enforce a tolerance.)
    diag_Op = Ub.conj().T @ Op @ Ub
    eigs_Op = np.real(np.diag(diag_Op))
    
    # Initialize dictionary to store matrices for each frequency mode.
    freq_modes = {}
    d = rho_rot.shape[0]
    
    # Loop over each element of rho_rot and assign it to the appropriate frequency.
    # The frequency is defined as the difference between the different values of the 
    # eigval of the Operator of the row and column.
    for i in range(d):
        for j in range(d):
            # frequency mode associated with element (i,j)
            freq = int(eigs_Op[i] - eigs_Op[j])
            # Initialize the mode if it does not exist; same shape as rho_rot.
            if freq not in freq_modes:
                freq_modes[freq] = np.zeros_like(rho_rot, dtype=rho_rot.dtype)
            freq_modes[freq][i, j] = rho_rot[i, j]
            
    return freq_modes

'''
# Test cases
rho_pure = np.array([[1, 0], [0, 0]])  # Pure state
rho_mixed = np.array([[0.5, 0], [0, 0.5]])  # Maximally mixed state
rho_intermediate = np.array([[0.7, 0.3], [0.3, 0.3]])  # Intermediate case

sigma_mixed = np.array([[0.6, 0.4], [0.4, 0.4]])  # Reference state

# Test different values of alpha
alpha_values = [0.0001, .9999, 100]  # Alpha → 0, 1 (KL), and large alpha
test_pairs = {
    "Pure vs Mixed": (rho_pure, sigma_mixed),
    "Mixed vs Mixed": (rho_mixed, sigma_mixed),
    "Intermediate vs Mixed": (rho_intermediate, sigma_mixed)
}

for name, (rho, sigma) in test_pairs.items():
    print(f"\n### {name} ###")
    print(f"  Max Divergence = {max_divergence(rho, sigma)}")
    print(f"  KL-1 Divergence = {renyi_divergence(rho, sigma, 1)}")
    for alpha in alpha_values:
        renyi_val = renyi_divergence(rho, sigma, alpha)
        print(f"  Rényi Divergence (α={alpha}): {renyi_val}")

    print(f"  Large α Approx: {renyi_divergence(rho, sigma, 100)}")
    
'''

# U1 ############

def spin_basis_1d(N, m=0.0, dtype=np.uint32, tol=1e-8):
    """
    NumPy‐only re‑implementation of quspin.basis.spin_basis_1d
    valid for any N (even or odd) and allowing half‑integer total M.

    Parameters
    ----------
    N : int
        Number of spin‑1/2 sites.
    m : float
        Magnetization per site; 2*m*N must be (nearly) integer.
    dtype : np.dtype
        Output integer dtype for the bit‑strings.
    tol : float
        Tolerance for floating‑point checks.

    Returns
    -------
    Ns : int
        Number of basis states = binomial(N, n_up).
    states : ndarray[int]
        Unsigned ints (dtype) whose binary representation encodes the spin configuration,
        sorted descending exactly as QuSpin.spin_basis_1d does.
        
    TEST:
        Ns = np.arange(4, 21)
        for N in Ns:
            tot_states = 0
            ms = np.linspace(-N/2, N/2, N+1)/N
            for m in ms:
                numb_states = len(fn.spin_basis_1d(N, m=m))
                tot_states += numb_states
            assert tot_states == 2**N, f'numb states found for {N}: {tot_states} instead of {2**N}'
    """

    # 1) total 2M = 2*m*N must be integer
    twoM_float = 2 * m * N
    if abs(twoM_float - round(twoM_float)) > tol:
        raise ValueError(f"2*m*N = {twoM_float} not (nearly) integer")
    twoM = int(round(twoM_float))

    # 2) number of up‑spins: n_up = (2M + N) / 2
    if (twoM + N) % 2 != 0:
        # Should never happen if twoM is int and N is int, but just in case
        raise ValueError(
            f"(2*m*N + N)/2 = {(twoM+N)/2} not integer"
        )
    n_up = (twoM + N)//2
    if not (0 <= n_up <= N):
        raise ValueError(f"n_up = {n_up} out of range [0, {N}]")

    # 3) enumerate all combinations of n_up ones in N bits,
    #    in descending‐bit lex order to match QuSpin exactly.
    states_list = []
    # iterate over positions N-1, N-2, …, 0
    for comb_ in combinations(range(N-1, -1, -1), n_up):
        s = 0
        for i in comb_:
            s |= (1 << i)
        states_list.append(s)
        
    states = np.array(states_list, dtype=dtype)
    assert len(states) == comb(N, int(N/2 + m * N)), \
        f'Wrong basis for {N}, {m}: {len(states)} != {comb(N, int(N/2 + m * N))}'

    return states


# Create a dictionary to hold projectors for each magnetization subsector.
# Here we assume the magnetization m runs from -NA/2 to NA/2 in steps of 1.

def build_projectors(Ns):
    projectors = {}
    U_U1 = np.zeros((2**Ns, 2**Ns), dtype=np.complex128)
    row_index = 0
    old_basis = None
    for m in np.linspace(-Ns/2, Ns/2, Ns+1)/Ns:
        # Retrieve the list of computational basis states for this magnetization sector.
        # (Assuming spin_basis_1d(NA, m=m) returns an object with a member .states.)
        if old_basis is not None:
            it = 0
            while len(old_basis) == len(spin_basis_1d(Ns, m=m)):
                m += 1e-7
                it += 1
                if it > 1000:
                    print("Warning: too many iterations")
                    break
        
        basis_obj = spin_basis_1d(Ns, m=m)
        old_basis = basis_obj
                
        states_m = basis_obj
        # print(f"Magnetization m {m:.2f} has", len(states_m), "states: states_m =", states_m)
        for state in states_m[::-1]:
            U_U1[state, row_index] = 1
            row_index += 1
        # Compute the projector onto the subspace spanned by these states.
        projectors[m] = compute_projector(Ns, states_m)
        

        
    return projectors, U_U1
