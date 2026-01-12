module mod_Diffusion1D_Base

    use iso_c_binding, only: ip => c_int32_t, rp => c_float, dp => c_double
    use iso_c_binding, only: c_f_pointer, c_loc, c_funloc

    implicit none

    type Diffusion1D_Base_t
        real(rp) :: viscosity
        real(rp):: advectionVelocity
        real(rp):: domainSize
        integer(ip) :: p
        integer(ip) :: nelem
        integer(ip) :: npoin
        integer(ip) :: maxIters
        real(dp) :: tol
		integer(ip) :: nsteps
		real(rp) :: time
		real(rp) :: deltaT

		integer(ip) outUnit
		character(:), allocatable :: outFile
		integer(ip) :: spatialWrites
		integer(ip) :: temporalWrites

        real(rp), contiguous, pointer :: localOperator(:, :)
		real(rp), contiguous, pointer :: lglWeights(:)
		real(rp), contiguous, pointer :: nodes(:)
		real(rp), contiguous, pointer :: state(:)


    contains

		! Parent methods
		procedure, pass :: initialize_parent => Diffusion1D_initialize
		procedure, pass :: finalize_parent => Diffusion1D_finalize
        procedure, pass :: free => Diffusion1D_free

		!Child methods
		procedure, pass :: initialize => Diffusion1D_initialize
        procedure, pass :: finalize => Diffusion1D_finalize
        procedure, pass :: solve => Diffusion1D_solve

    end type

    interface Diffusion1D_Base_t
        procedure ::  this
    end interface

contains

    type(Diffusion1D_Base_t) function this()
    end function

    subroutine Diffusion1D_free(this)
        class(Diffusion1D_Base_t), intent(inout) :: this

		this%viscosity = 0_rp
        this%advectionVelocity = 0_rp
        this%domainSize = 0_rp

        ! solver parameters
        this%p = 0_ip   ! CG polynomial order
        this%nelem = 0_ip
        this%maxIters = 0_ip
        this%tol = 0_dp

		! time parameters
		this%nsteps = 0_ip
		this%time = 0_rp
        this%deltaT = 0_rp

		! output parameters
		this%spatialWrites = 0_ip
		this%temporalWrites = 0_ip
		this%outUnit = 0_ip
		this%outFile =  ""

        this%npoin = 0
        nullify (this%localOperator)
		nullify (this%nodes)
		nullify (this%state)
		nullify (this%lglWeights)

    end subroutine

	subroutine Diffusion1D_initialize(this)
        class(Diffusion1D_Base_t), intent(inout) :: this

		!$acc enter data copyin(this)

		! physical parameters
        this%viscosity = 0.01_rp
        this%advectionVelocity = 0.8_rp
        this%domainSize = 2._rp

        ! solver parameters
        this%p = 5_ip   ! CG polynomial order
        this%nelem = 20_ip
        this%maxIters = 100_ip
        this%tol = 1e-6_dp

		! time parameters
		this%nsteps = 1000_ip
		this%time = 0.0_rp
        this%deltaT = 0.01_rp

		! output parameters
		this%spatialWrites = 100_ip
		this%temporalWrites = 200_ip
		this%outUnit = 1192_ip
		this%outFile =  "GLASsAdvDiff1DOut.txt"

        this%npoin = this%p*this%nelem + 1

        allocate (this%localOperator(this%p+1, this%p+1), source=0.0_rp)
		allocate (this%nodes(this%npoin), source=0.0_rp)
		allocate (this%state(this%npoin), source=0.0_rp)
		allocate (this%lglWeights(this%p+1), source=0.0_rp)

		!$acc enter data copyin(this%localOperator, this%nodes, this%state, this%lglWeights)

		open (this%outUnit, file=this%outFile, action="write")

    end subroutine

    subroutine Diffusion1D_finalize(this)
        class(Diffusion1D_Base_t):: this
        if (associated(this%localOperator)) deallocate (this%localOperator)
		if (associated(this%nodes)) deallocate (this%nodes)
		if (associated(this%state)) deallocate (this%state)

		close(this%outUnit)
        stop 1

    end subroutine

    subroutine Diffusion1D_solve(this)
        class(Diffusion1D_Base_t), intent(inout) :: this

        write (*, *) "ERROR : Diffusion1D_solve : Method not implemented"
        stop 1

    end subroutine

end module
