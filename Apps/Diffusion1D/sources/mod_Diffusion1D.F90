module mod_Diffusion1D
    use mod_Diffusion1D_Jacobian
	use mod_Diffusion1D_Implicit
    use mod_Diffusion1D_Base
 
	public

	contains

	subroutine createDiffusionSolver(type, driver)

		character(len=*),                   intent(in)    :: type
		class(Diffusion1D_Base_t), pointer, intent(inout) :: driver

		select case(trim(type))
			case( "IMPLICIT" ); allocate(Diffusion1D_Implicit_t :: driver)
			case default;
				write(*,*) "ERROR : createDiffusionSolver: Child class not implemented"
				stop 1
		end select
	end subroutine
end module
