program Diffusion1D
	use mod_Diffusion1D

    implicit none
        
    class(Diffusion1D_Base_t), pointer :: Diff1D

    call createDiffusionSolver('IMPLICIT', Diff1D)
    call Diff1D%initialize()
    call Diff1D%solve()
    call Diff1D%finalize()
end program