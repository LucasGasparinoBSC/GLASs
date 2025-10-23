program Diffusion1D
	use mod_Diffusion1D

    implicit none
        
    class(Diffusion1D_Base_t), pointer :: Diffusion1D

    call create_DiffusionSolver('IMPLICIT', Diffusion1D)
    call Diffusion1D%initialize()
    call Diffusion1D%solve()
    call Diffusion1D%finalize()
end program