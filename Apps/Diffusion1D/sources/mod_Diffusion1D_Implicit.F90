module mod_Diffusion1D_Implicit

    use iso_c_binding, only: ip => c_int32_t, rp => c_float, dp => c_double
    use iso_c_binding, only: c_loc, c_funloc, c_ptr, c_funptr
    use cg_wrapper_mod
    use mod_Diffusion1D_Operators
    use mod_Diffusion1D_Base

    implicit none

    type, extends(Diffusion1D_Base_t) :: Diffusion1D_Implicit_t

    contains
		procedure, pass :: initialize => Diffusion1D_Implicit_initialize
		procedure, pass :: finalize => Diffusion1D_Implicit_finalize

		procedure, pass :: multiplyMassMatrix => Diffusion1D_Implicit_mult_mass
        procedure, pass :: matvec => Diffusion1D_Implicit_matvec
        procedure, pass :: solve => Diffusion1D_Implicit_solve

		procedure, pass :: getGlobalNodes => Diffusion1D_Implicit_get_global_nodes
		procedure, pass :: setInitCond => Diffusion1D_Implicit_set_init_cond

		procedure, pass :: writeNodes => Diffusion1D_Implicit_write_nodes
		procedure, pass :: writeOutput => Diffusion1D_Implicit_write_output	

    end type

    private

    public :: Diffusion1D_Implicit_t

contains

	subroutine Diffusion1D_Implicit_initialize(this)
	class(Diffusion1D_Implicit_t), intent(inout) :: this
	class(Diffusion1D_Operators_t), allocatable   :: operators_object

    call this%free()
	call this%initialize_parent()

	operators_object = Diffusion1D_Operators_t(p=this%p, nelem=this%nelem, &
													advectionVelocity=this%advectionVelocity, &
													viscosity=this%viscosity, &
													domainSize=this%domainSize &
													)
	call operators_object%getLocalImplicitOperator(this%deltaT, this%localOperator, this%lglWeights)
	!!$acc update device(this%localOperator, this%lglWeights)

	call this%getGlobalNodes()
	call this%setInitCond()

	end subroutine

	subroutine Diffusion1D_Implicit_finalize(this)
	class(Diffusion1D_Implicit_t) :: this
		call this%finalize_parent()
	end subroutine

	subroutine Diffusion1D_Implicit_mult_mass(this, s)
	class(Diffusion1D_Implicit_t), intent(inout) :: this
	real(rp), contiguous, pointer :: s(:)
	integer(ip) elemStartIdx, outIdx
	integer(ip) i, r

	! multiply the input by the mass matrix
	! necessary to ensure the RHS operator given to the solver is symmetric positive definite

	! consider all nodes at element boundaries - indices are 1 more than a multiple of p
    !!$acc parallel loop
        do i = 2, ((this%npoin - 1)/this%p)
            outIdx = this%p*i + 1
            s(outIdx) = s(outIdx) * (this%lglWeights(1) + this%lglWeights(this%p+1))
        end do
    !!$acc end parallel loop

	! other nodes
	!!$acc parallel loop
        do i = 1, (this%npoin/this%p) - 1
            do r = 1, this%p - 1
                elemStartIdx = this%p*i + 1
                s(elemStartIdx + r) = this%lglWeights(r + 1)*s(elemStartIdx + r)
            end do
        end do
    !!$acc end parallel loop
	end subroutine 

    ! Fortran concrete routine
    subroutine Diffusion1D_Implicit_matvec(this, x_in, x_out)
        class(Diffusion1D_Implicit_t), intent(inout) :: this
        real(rp), intent(in)    :: x_in(this%npoin)
        real(rp), intent(out)   :: x_out(this%npoin)
        integer(4) :: i, r, elemId, elemStartIdx, outIdx

        ! need to treat x_out differently if p divides current row
        !!$acc parallel loop deviceptr(x_in, x_out) !present(this%localOperator)
        do i = 2, ((this%npoin - 1)/this%p)
            outIdx = this%p*i + 1
            x_out(outIdx) = dot_product(this%localOperator(1, :), x_in(outIdx:outIdx + this%p)) &
                            + dot_product(this%localOperator(this%p + 1, :), x_in(outIdx - this%p:outIdx))
        end do
        !!$acc end parallel loop

		! instead of *checking* whether p divides i-1, "impose" the remainder with p loops

        !!$acc parallel loop deviceptr(x_in, x_out) present(this%localOperator) collapse(2)
        do i = 1, (this%npoin/this%p) - 1
            do r = 1, this%p - 1
                elemStartIdx = this%p*i + 1
                x_out(elemStartIdx + r) = dot_product(this%localOperator(r + 1,:),x_in(elemStartIdx:elemStartIdx + this%p))
            end do
        end do
        !!$acc end parallel loop

		! apply Dirichlet boundary conditions
		x_out(1) = 0
		x_out(this%npoin) = 0

    end subroutine Diffusion1D_Implicit_matvec

    ! C wrapper for the Fortran matvec
    subroutine Diffusion1D_Implicit_matvec_c(x_in, x_out, user_data) bind(C)
        use iso_c_binding, only: rp => c_float
        implicit none
        real(rp), intent(in)  :: x_in(*)
        real(rp), intent(out) :: x_out(*)
        type(c_ptr), value       :: user_data

        type(Diffusion1D_Implicit_t), pointer :: solverObject

        call c_f_pointer(user_data, solverObject)
        call solverObject%matvec(x_in, x_out)

    end subroutine Diffusion1D_Implicit_matvec_c

    subroutine Diffusion1D_Implicit_solve(this)
        class(Diffusion1D_Implicit_t), intent(inout) :: this

        type(c_ptr) :: cgSolver, opData
        type(c_funptr) :: matvec_funptr
        real(rp), contiguous, pointer :: x0(:), state_p(:)

        integer :: i

		select type (op => this)
        type is (Diffusion1D_Implicit_t)
            opData = c_loc(op)
        end select

        ! Allocate and initialize vectors
        allocate (x0(this%npoin), source=0.001_rp)
        allocate (state_p(this%npoin), source=this%state)

        ! Alias variables & pointers
        !state_p => this%state

		!!$acc enter data copyin(x0, state_p, this%state)


		! Create solver instance and setup
		cgSolver = cg_create_u32_f(this%npoin, this%maxIters, this%tol)
		matvec_funptr = c_funloc(Diffusion1D_Implicit_matvec_c)

		call this%writeNodes()

		do i = 0, this%nsteps - 1
			this%time = this%time + this%deltaT
			call this%writeOutput(i,state_p)
			call this%multiplyMassMatrix(state_p)
			! ! pass pointers to variables on device
            ! !!$acc host_data use_device(x0, state_p)
			! call cg_setup_u32_f(cgSolver, x0, state_p)
			! call cg_solve_u32_f(cgSolver, matvec_funptr, opData)
			! call cg_get_solution_u32_f(cgSolver, state_p)
            ! !!$acc end host_data

		end do
    end subroutine

	
	subroutine Diffusion1D_Implicit_set_init_cond(this)
		class(Diffusion1D_Implicit_t), intent(in) :: this
		integer i

		!!$acc parallel loop present(this%nodes)
		do i = 1, this%npoin
			! initialize b to Gaussian temperature spot
			this%state(i) = exp(-(this%nodes(i)**2)/(this%viscosity))
		enddo
		!!$acc end parallel loop

	end subroutine
	

	subroutine Diffusion1D_Implicit_get_global_nodes(this)
		class(Diffusion1D_Implicit_t), intent(inout) :: this
		real(rp) :: deltaX
		integer i, r
		real(rp) :: localNodes(this%p+1), localWeights(this%p+1)

		call LegendreGaussLobattoNodeEval(this%p+1, localNodes, localWeights)

		deltaX = this%domainSize/this%nelem

		! calculate coordinate at beginning of each spectral element

		!!$acc parallel loop
		do i = 0, this%nelem - 1
			this%nodes(i*this%p + 1) = -this%domainSize/2 +  deltaX*i
		end do
		!!$acc end parallel loop

		!!$acc parallel loop collapse(2)
		! fill rest of coordinates by adding transformed LGL nodes to beginning of each element
		do i = 0, this%nelem - 1
			do r = 1, this%p - 1
			this%nodes(i*this%p + 1 + r) = this%nodes(i*this%p + 1) + ((1+localNodes(r+1))*deltaX/2)
			end do
		end do
		!!$acc end parallel loop

		!!$acc update host(this%nodes)



	end subroutine

	subroutine Diffusion1D_Implicit_write_output(this, temporalIdx, s)
		class(Diffusion1D_Implicit_t), intent(inout) :: this
		real(rp), contiguous, pointer :: s(:)

		integer temporalIdx, temporalStep
		integer i
		integer spatialStep
		character(len=30) :: numSpatialWrites

		write(numSpatialWrites, "(I10)") this%spatialWrites

		spatialStep = this%npoin/this%spatialWrites
		temporalStep = this%nsteps/this%temporalWrites

		if (modulo(temporalIdx,temporalStep).ne.0) then
			!!$acc update host(this%time, s)
			write(this%outUnit, "(1F22.15)", advance="no") this%time
			write(this%outUnit, '('//numSpatialWrites//'(E22.15,2x))',advance="no") s(1:this%npoin:spatialStep)
			write(this%outUnit, "(A1)") " "
		endif

	end subroutine

		subroutine Diffusion1D_Implicit_write_nodes(this)
		class(Diffusion1D_Implicit_t), intent(inout) :: this
		integer i
		integer spatialStep
		character(len=30) :: numSpatialWrites

		write(numSpatialWrites, "(I10)") this%spatialWrites

		spatialStep = this%npoin/this%spatialWrites
		!!$acc update host(this%nodes)
		! write a 0 so that all the time points (at start of each row) line up nicely
		write(this%outUnit, "(1F22.15)", advance="no") 0_rp
		write(this%outUnit, '('//trim(numSpatialWrites)//'(E22.15,2x))',advance="no") this%nodes(1:this%npoin:spatialStep)
		write(this%outUnit, "(A1)") " "

	end subroutine

end module
