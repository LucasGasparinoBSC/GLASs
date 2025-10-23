program AdvectionDiffusion1D
	use mod_AdvectionDiffusion1D

    implicit none
        
    class(AdvectionDiffusion1D_Base_t), pointer :: Diffusion1D

    call create_AdvectionDiffusionSolver('IMPLICIT', Diffusion1D)
    call Diffusion1D%initialize()
    call Diffusion1D%solve()
    call Diffusion1D%finalize()
end program