First initialise the julia environment by navigating to the folder in terminal and running in julia the following
using Pkg; Pkg.activate("."); Pkg.instantiate()

To run simulation for parameters, enter them into batch.jl. Navigate to the folder in terminal and run julia batch.jl

This code was originally written for sampling U(1) gates randomly to build a layer that is to be repeated in time. For this, ensure that manual is set to false in batch.jl. 

Otherwise, untested version for manual setting of gate parameters. For this, set manual to true in batch.jl

In both cases, a 2-qubit gate is set up as (R_z(\theta_1/2) \otimes R_z(\theta_2/2)) R_xx(Jx/2) R_yy(Jx/2) R_zz(Jz/2) (R_z(\theta_3/2) \otimes R_z(\theta_4/2)). I am pretty sure, but it's been a while so need to double check that there isnt a factor of two floating around in the angles.

For random simulation (manual = false):
 - First run random_gen.jl for required params to generate files with random angles.
 - Then run batch.jl
 - Data will be saved in pauli_results/, errors will be saved in logs/

For manual simulation:
  - First run set_params.jl to save manual gate parameters. For the moment it is homogenous 2 qubit interactions across the chain, feel free to edit it to handle non-homogenous setups
  - Run batch.jl
  - Data will be saved in pauli_results_manual/, errors will be saved in logs_manual/

Other notes:
  - plot_results.jl needs fixing, use at own caution.
  - Used to save pauli weight distribution via the calc_weight_dist param. No longer in use, instead replaces with save_coeffs to have direct access to pauli coefficients and can construct more things in post processing.
  - However, this is costly, so might be worth setting to false if simulations are taking a while / too much space is being taken up in save files.

    
