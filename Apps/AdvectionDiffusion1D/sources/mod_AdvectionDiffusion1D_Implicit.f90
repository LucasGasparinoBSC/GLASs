module mod_AdvectionDiffusion1D_Implicit

    use iso_c_binding, only: ip => c_int32_t, rp => c_float, dp => c_double
    use iso_c_binding, only: c_loc, c_funloc, c_ptr, c_funptr
    !use cg_wrapper_mod
    use mod_AdvectionDiffusion1D_Jacobian
    use mod_AdvectionDiffusion1D_Base

    implicit none

    type, extends(AdvectionDiffusion1D_Base_t) :: AdvectionDiffusion1D_Implicit_t
    contains
		procedure, pass :: initialize => AdvectionDiffusion1D_Implicit_initialize
		procedure, pass :: finalize => AdvectionDiffusion1D_Implicit_initialize

        procedure, pass :: matvec => AdvectionDiffusion1D_Implicit_matvec
        procedure, pass :: solve => AdvectionDiffusion1D_Implicit_solve

		procedure, pass :: getGlobalNodes => AdvectionDiffusion1D_Implicit_get_global_nodes
		procedure, pass :: setInitCond => AdvectionDiffusion1D_Implicit_set_init_cond

		procedure, pass :: writeNodes => AdvectionDiffusion1D_Implicit_write_nodes
		procedure, pass :: writeOutput => AdvectionDiffusion1D_Implicit_write_output	

    end type

    private

    public :: AdvectionDiffusion1D_Implicit_t

contains

	subroutine AdvectionDiffusion1D_Implicit_initialize(this)
	class(AdvectionDiffusion1D_Implicit_t) :: this
	class(AdvectionDiffusion1D_Jacobian_t), allocatable :: jac_object
	call this%initialize_parent()

	allocate (this%localOperator(this%p + 1, this%p + 1))

	jac_object = AdvectionDiffusion1D_Jacobian_t(p=this%p, nelem=this%nelem, &
													advectionVelocity=this%advectionVelocity, &
													viscosity=this%viscosity, &
													domainSize=this%domainSize &
													)
	call jac_object%getLocalImplicitOperator(this%deltaT, this%localOperator)

	call this%getGlobalNodes()
	call this%setInitCond()

	end subroutine

	subroutine AdvectionDiffusion1D_Implicit_finalize(this)
	class(AdvectionDiffusion1D_Implicit_t) :: this
		call this%finalize_parent()
	end subroutine

    ! Fortran concrete routine
    subroutine AdvectionDiffusion1D_Implicit_matvec(this, x_in, x_out)
        class(AdvectionDiffusion1D_Implicit_t), intent(inout) :: this
        real(rp), intent(in)    :: x_in(this%npoin)
        real(rp), intent(out)   :: x_out(this%npoin)
        integer(4) :: i, r, elemId, elemStartIdx, outIdx

        ! need to treat x_out differently if p divides current row
        !$acc parallel loop deviceptr(x_in, x_out) present(this%localOperator)
        do i = 2, ((this%npoin - 1)/this%p)
            outIdx = this%p*i + 1
            x_out(outIdx) = dot_product(this%localOperator(1, :), x_in(outIdx:outIdx + this%p)) &
                            + dot_product(this%localOperator(this%p + 1, :), x_in(outIdx - this%p:outIdx))
        end do
        !$acc end parallel loop

		! instead of *checking* whether p divides i-1, it's better to "impose" the remainder with p loops

        !$acc parallel loop deviceptr(x_in, x_out) present(this%localOperator) collapse(2)
        do i = 1, (this%npoin/this%p) - 1
            do r = 1, this%p - 1
                elemStartIdx = this%p*i + 1
                x_out(elemStartIdx + r) = dot_product(this%localOperator(r + 1,:),x_in(elemStartIdx:elemStartIdx + this%p))
            end do
        end do
        !$acc end parallel loop

		! apply Dirichlet boundary conditions
		x_out(1) = 0
		x_out(this%npoin) = 0

    end subroutine AdvectionDiffusion1D_Implicit_matvec

    ! C wrapper for the Fortran matvec
    subroutine AdvectionDiffusion1D_Implicit_matvec_c(x_in, x_out, user_data) bind(C)
        use iso_c_binding, only: rp => c_float
        implicit none
        real(rp), intent(in)  :: x_in(*)
        real(rp), intent(out) :: x_out(*)
        type(c_ptr), value       :: user_data

        type(AdvectionDiffusion1D_Implicit_t), pointer :: solverObject

        call c_f_pointer(user_data, solverObject)
        call solverObject%matvec(x_in, x_out)

    end subroutine AdvectionDiffusion1D_Implicit_matvec_c

    subroutine AdvectionDiffusion1D_Implicit_solve(this)
        class(AdvectionDiffusion1D_Implicit_t):: this

        type(c_ptr) :: cgSolver, opData
        type(c_funptr) :: matvec_funptr
        real(rp), allocatable :: x0(:), b(:), s(:)

        integer :: i

        call this%initialize()

		!$acc enter data copyin(this)

		select type (op => this)
        type is (AdvectionDiffusion1D_Implicit_t)
            opData = c_loc(op)
        end select

        ! Allocate and initialize vectors
        allocate (x0(this%npoin), source=0.001_rp)
        allocate (s(this%npoin), source=0.0_rp)

		do i = 1, this%nsteps
			this%time = this%time + this%deltaT
			!$acc enter data copyin(x0, b, s)

			! Create solver instance and setup
			cgSolver = cg_create_u32_f(this%npoin, this%maxIters, this%tol)

			matvec_funptr = c_funloc(AdvectionDiffusion1D_Implicit_matvec_c)

			call cg_setup_u32_f(cgSolver, x0, this%state)
			call cg_solve_u32_f(cgSolver, matvec_funptr, opData)
			call cg_get_solution_u32_f(cgSolver, s)
			this%state = s

			call this%writeOutput(i)
		end do

		call this%finalize()

    end subroutine

	
	subroutine AdvectionDiffusion1D_Implicit_set_init_cond(this)
		class(AdvectionDiffusion1D_Implicit_t), intent(in) :: this
		integer i

		!$acc parallel loop present(this%state)
		do i = 1, this%npoin
			! initialize b to Gaussian temperature spot
			this%state(i) = exp(-(this%nodes(i)**2)/(this%viscosity))
		enddo
		!$acc end parallel loop
		!$acc update host(this%state)
	end subroutine
	

	subroutine AdvectionDiffusion1D_Implicit_get_global_nodes(this)
		class(AdvectionDiffusion1D_Implicit_t), intent(inout) :: this
		real(rp) :: deltaX
		integer i, r
		real(rp) :: localNodes(this%p+1), localWeights(this%p+1)

		call LegendreGaussLobattoNodeEval(this%p+1, localNodes, localWeights)

		deltaX = this%domainSize/this%nelem

		! calculate coordinate at beginning of each spectral element

		!$acc parallel loop present(this%nodes)
		do i = 0, this%nelem - 1
			this%nodes(i*this%p) = -this%domainSize/2 +  deltaX*i
		end do
		!$acc end parallel loop

		!$acc parallel loop present(this%nodes) collapse(2)
		! fill rest of coordinates by adding transformed LGL nodes to beginning of each element
		do i = 0, this%nelem - 1
			do r = 1, this%p - 1
			this%nodes(i*this%p + r) = this%nodes(i*this%p) + ((1+localNodes(r+1))*deltaX/2)
			end do
		end do
		!$acc end parallel loop

		!$acc update host(this%nodes)


	end subroutine

	subroutine AdvectionDiffusion1D_Implicit_write_output(this, temporalStep)
		class(AdvectionDiffusion1D_Implicit_t), intent(inout) :: this
		integer temporalStep
		integer i
		character(len=:), allocatable :: numSpatialWrites

		write(numSpatialWrites, "(I10)") this%npoin/this%spatialWriteStep

		if (modulo(temporalStep,this%temporalWriteStep).ne.0) then
			!$acc update host(this%time, this%state)
			write(this%outUnit, "(1F22.15)", advance="no") this%time
			write(this%outUnit, '('//numSpatialWrites//'(E22.15,2x))',advance="no") this%state(1:this%npoin:this%spatialWriteStep)
			write(this%outUnit, "(A1)") " "
		endif

	end subroutine

		subroutine AdvectionDiffusion1D_Implicit_write_nodes(this)
		class(AdvectionDiffusion1D_Implicit_t), intent(inout) :: this
		integer i
		character(len=:), allocatable :: numSpatialWrites

		write(numSpatialWrites, "(I10)") this%npoin/this%spatialWriteStep

		!$acc update host(this%nodes)
		write(this%outUnit, "(1F22.15)", advance="no") 0_rp
		write(this%outUnit, '('//numSpatialWrites//'(E22.15,2x))',advance="no") this%nodes(1:this%npoin:this%spatialWriteStep)
		write(this%outUnit, "(A1)") " "

	end subroutine

end module
