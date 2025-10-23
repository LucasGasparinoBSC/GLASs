module mod_AdvectionDiffusion1D
    use mod_AdvectionDiffusion1D_Jacobian
    use mod_AdvectionDiffusion1D_Base
 
	public

	contains

	subroutine createAdvectionDiffusionSolver(type, driver)

		character(len=*),                   		 intent(in)    :: type
		class(AdvectionDiffusion1D_Base_t), pointer, intent(inout) :: driver

		select case(trim(type))
			case( "IMPLICIT" ); allocate(AdvectionDiffusion1D_Base_t :: driver)
			case default;
				write(*,*) "ERROR : createAdvectionDiffusionSolver: Child class not implemented"
				stop 1
		end select
	end subroutine
end module
