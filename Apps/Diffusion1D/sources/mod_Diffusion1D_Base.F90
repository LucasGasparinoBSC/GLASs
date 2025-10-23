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
        real(rp) :: tol
		integer(ip) :: nsteps
		real(rp) :: time
		real(rp) :: deltaT

		integer(ip) outUnit
		character(:), allocatable :: outFile
		integer(ip) :: spatialWriteStep
		integer(ip) :: temporalWriteStep

        real(rp), contiguous, pointer :: localOperator(:, :)
		real(rp), contiguous, pointer :: nodes(:)
        real(rp), contiguous, pointer :: state(:)

    contains

		! Parent methods
		procedure, pass :: initialize_parent => Diffusion1D_initialize
		procedure, pass :: finalize_parent => Diffusion1D_finalize

		! Child methods
        procedure, pass :: free => Diffusion1D_free
		procedure, pass :: initialize => Diffusion1D_initialize
        procedure, pass :: finalize => Diffusion1D_finalize
        procedure, pass :: solve => Diffusion1D_solve

    end type

    interface Diffusion1D_Base_t
        procedure ::  this
    end interface

contains

    type(Diffusion1D_Base_t) function this()

        ! physical parameters
        real(rp), parameter :: viscosity = 0.01_rp
        real(rp), parameter :: advectionVelocity = 0.8_rp
        real(rp), parameter :: domainSize = 16._rp

        ! solver parameters
        integer(ip), parameter :: p = 5   ! CG polynomial order
        integer(ip), parameter :: nelem = 200_ip
        integer(ip), parameter :: maxIters = 100_ip
        real(rp), parameter :: tol = 1e-6_rp

		! time parameters
		integer(ip), parameter :: nsteps = 1000_ip
		real(rp) :: time
        real(rp), parameter :: deltaT = 0.01_rp

		! output parameters
		integer(ip), parameter :: spatialWrites = 100_ip
		integer(ip), parameter :: temporalWrites = 200_ip
		integer(ip), parameter :: outUnit = 1192_ip
		character, parameter :: outFile =  "GLASsAdvDiff1DOut.txt"

        integer(ip) :: npoin


        npoin = p*nelem + 1

        this%viscosity = viscosity
        this%advectionVelocity = advectionVelocity
        this%domainSize = domainSize

        this%p = p
        this%nelem = nelem
        this%npoin = npoin
        this%maxIters = maxIters
        this%tol = tol
		
		this%outUnit = outUnit
		this%spatialWriteStep = max(1, npoin/spatialWrites)
		this%temporalWriteStep = max(1, nsteps/temporalWrites)

        allocate (this%localOperator(npoin, npoin), source=0.0_rp)
		allocate (this%nodes(npoin), source=0.0_rp)
        allocate (this%state(npoin), source=0.0_rp)

    end function

    subroutine Diffusion1D_free(this)
        class(Diffusion1D_Base_t), intent(inout) :: this

        this%viscosity = 0.0_rp
        this%advectionVelocity = 0.0_rp
        this%domainSize = 0.0_rp

        this%p = 0_ip
        this%nelem = 0_ip
        this%npoin = 0_ip
        this%maxIters = 0_ip
        this%tol = 0_ip

        nullify (this%localOperator)
		nullify (this%nodes)
        nullify (this%state)

    end subroutine

	subroutine Diffusion1D_initialize(this)
        class(Diffusion1D_Base_t), intent(inout) :: this

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
