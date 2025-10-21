module mod_AdvectionDiffusion1D_Base

    use iso_c_binding, only: ip => c_int32_t, rp => c_float, dp => c_double
    use iso_c_binding, only: c_f_pointer, c_loc, c_funloc

    implicit none

    type AdvectionDiffusion1D_Base_t
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

        real(rp), contiguous, pointer :: localJac(:, :)
        real(rp), contiguous, pointer :: state(:)

    contains
        procedure, pass :: free => AdvectionDiffusion1D_free
        procedure, pass :: finalize => AdvectionDiffusion1D_finalize
        procedure, pass :: solve => AdvectionDiffusion1D_solve

    end type

    interface AdvectionDiffusion1D_Base_t
        procedure ::  this
    end interface

contains

    type(AdvectionDiffusion1D_Base_t) function this()

        ! physical parameters
        real(rp), parameter :: viscosity = 0.01
        real(rp), parameter :: advectionVelocity = 0.8
        real(rp), parameter :: domainSize = 16.

        ! solver parameters
        integer(ip), parameter :: p = 5   ! CG polynomial order
        integer(ip), parameter :: nelem = 200
        integer(ip), parameter :: maxIters = 100
        real(rp), parameter :: tol = 1e-6

		! time parameters
		integer(ip), parameter :: nsteps = 100
		real(rp) :: time
        real(rp), parameter :: deltaT = 0.01

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

        allocate (this%localJac(npoin, npoin), source=0.0_rp)
        allocate (this%state(npoin), source=0.0_rp)

    end function

    subroutine AdvectionDiffusion1D_free(this)
        class(AdvectionDiffusion1D_Base_t):: this

        this%viscosity = 0.0_rp
        this%advectionVelocity = 0.0_rp
        this%domainSize = 0.0_rp

        this%p = 0_ip
        this%nelem = 0_ip
        this%npoin = 0_ip
        this%maxIters = 0_ip
        this%tol = 0_ip

        nullify (this%localJac)
        nullify (this%state)

    end subroutine

    subroutine AdvectionDiffusion1D_finalize(this)
        class(AdvectionDiffusion1D_Base_t):: this
        if (associated(this%localJac)) deallocate (this%localJac)
        if (associated(this%state)) deallocate (this%state)
        stop 1

    end subroutine

    subroutine AdvectionDiffusion1D_solve(this)
        class(AdvectionDiffusion1D_Base_t):: this

        write (*, *) "ERROR : AdvectionDiffusion1D_solve : Method not implemented"
        stop 1

    end subroutine

end module
