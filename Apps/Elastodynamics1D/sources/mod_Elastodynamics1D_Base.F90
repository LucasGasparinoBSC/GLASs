module mod_ElastoDynamics1D_Base

    use iso_c_binding, only: ip => c_int32_t, rp => c_float, dp => c_double
    use iso_c_binding, only: c_f_pointer, c_loc, c_funloc
    use cg_wrapper_mod

    implicit none

    integer(ip) :: GLASS_ED1D_OUT_UNIT = 19900502

    type :: ElastoDynamics1D_t
        ! Type of problem
        character(len=:), allocatable :: type
        ! Physical properties
        real(rp)          :: E     ! Young modulus
        real(rp)          :: A     ! Area
        real(rp)          :: rho   ! Density
        real(rp)          :: L     ! Length
        ! Physical parameters (derived)
        real(rp)          :: Le    ! Element lentgh
        real(rp)          :: c     ! Elastic wave velocity
        ! Harmonic excitation
        real(rp)          :: H     ! Amplitude 
        real(rp)          :: w     ! Period
        ! Time integration parameters
        real(rp)          :: time
        real(rp)          :: dtime
        integer(ip)       :: nsteps
        ! Discretization parameters
        integer(ip)       :: nelem
        integer(ip)       :: npoin
        integer(ip)       :: nnode
        ! Arrays
        real(rp), contiguous, pointer :: stifm(:)  ! Elemental stiffness matrix
        real(rp), contiguous, pointer :: massm(:)  ! Elemental mass matrix
        real(rp), contiguous, pointer :: displ(:)  ! Nodal diplacements
        real(rp), contiguous, pointer :: veloc(:)  ! Nodal velocities
        real(rp), contiguous, pointer :: accel(:)  ! Nodal accelerations
        real(rp), contiguous, pointer :: fexte(:)  ! Nodal external forces
        ! CG parameters
        integer(ip)       :: niter
        real(dp)          :: toler
        ! Output
        integer(ip)       :: out_freq
        integer(4)        :: out_unit
    contains
        ! Class methods
        procedure, pass :: free_parent => ElastoDynamics1D_free
        procedure, pass :: initialize_parent => ElastoDynamics1D_initialize
        procedure, pass :: finalize_parent => ElastoDynamics1D_finalize
        procedure, pass :: cleanVect => ElastoDynamics1D_cleanVect
        procedure, pass :: computeCoeffs => ElastoDynamics1D_computeCoeffs
        procedure, pass :: computeExternalForce => ElastoDynamics1D_computeExternalForce
        procedure, pass :: updateExternalForce => ElastoDynamics1D_updateExternalForce
        procedure, pass :: outputNodeSets => ElastoDynamics1D_outputNodeSets
        ! Child methods
        procedure, pass :: free => ElastoDynamics1D_free
        procedure, pass :: initialize => ElastoDynamics1D_initialize
        procedure, pass :: finalize => ElastoDynamics1D_finalize
        procedure, pass :: solve => ElastoDynamics1D_solve
        procedure, pass :: applyFMOp => ElastoDynamics1D_applyFreeMatrixOperator
        procedure, pass :: assembleRHS => ElastoDynamics1D_assembleRHS
    end type ElastoDynamics1D_t

    private

    public :: ip, rp, dp
    public :: ElastoDynamics1D_t

    contains

    subroutine ElastoDynamics1D_free(this)
        
        class(ElastoDynamics1D_t), intent(inout) :: this

        this%E = 0.0_rp
        this%A = 0.0_rp
        this%rho = 0.0_rp
        this%L = 0.0_rp
        this%le = 0.0_rp
        this%c = 0.0_rp
        this%dtime = 0.0_rp
        this%nsteps = 0_ip
        this%time = 0.0_rp
        this%dtime = 0.0_rp
        this%nnode = 0_ip
        this%nelem = 0_ip
        this%H = 0.0_rp
        this%w = 0.0_rp
        nullify(this%stifm)
        nullify(this%massm)
        this%npoin = 0_ip
        nullify(this%displ)
        nullify(this%veloc)
        nullify(this%accel)
        nullify(this%fexte)

    end subroutine ElastoDynamics1D_free


    subroutine ElastoDynamics1D_initialize(this)
        
        class(ElastoDynamics1D_t), intent(inout) :: this

        real(rp) :: denom

        ! Free variables: set zero and nullify pointers
        call this%free()

        !$acc enter data copyin(this)

        ! Input parameters
        this%E = 210.0e9_rp
        this%A = 1.0e-4_rp
        this%rho = 7800.0_rp
        this%L = 100.0_rp 
        this%H = 1000.0_rp
        this%w = 5.0e-4_rp

        this%dtime = 1.0e-6_rp
        this%nsteps = 5000_ip
        this%time = 0.0_rp

        ! Nodes per element
        this%nnode = 2_ip

        ! Elemental arrays
        this%nelem = 100_ip 
        allocate(this%stifm(this%nelem), source=0.0_rp)
        allocate(this%massm(this%nelem), source=0.0_rp)
        !$acc enter data copyin(this%stifm, this%massm)

        ! Nodal arrays
        this%npoin = this%nelem + 1_ip
        allocate(this%displ(this%npoin), source=0.0_rp)
        allocate(this%veloc(this%npoin), source=0.0_rp)
        allocate(this%accel(this%npoin), source=9.81_rp)
        allocate(this%fexte(this%npoin), source=0.0_rp)
        !$acc enter data copyin(this%displ, this%veloc, this%accel, this%fexte)

        ! Derived parameters
        this%le = this%L/real(this%nelem,kind=rp)
        this%c = sqrt(this%E/this%rho)
        this%dtime = 0.5_rp * this%le / this%c

        ! SOlver
        this%niter = 100_ip
        this%toler = 1.0e-12_dp

        ! Output
        this%out_freq = 1_ip
        this%out_unit = GLASS_ED1D_OUT_UNIT

        !$acc update device(this)

    end subroutine ElastoDynamics1D_initialize


    subroutine ElastoDynamics1D_finalize(this)
        
        class(ElastoDynamics1D_t), intent(inout) :: this
        
        if( associated(this%stifm) ) deallocate(this%stifm)
        if( associated(this%massm) ) deallocate(this%massm)
        if( associated(this%displ) ) deallocate(this%displ)
        if( associated(this%veloc) ) deallocate(this%veloc)
        if( associated(this%accel) ) deallocate(this%accel)
        if( associated(this%fexte) ) deallocate(this%fexte)

        call this%free()

    end subroutine ElastoDynamics1D_finalize


    subroutine ElastoDynamics1D_solve(this)
        
        class(ElastoDynamics1D_t), intent(inout) :: this
        
    end subroutine ElastoDynamics1D_solve

    
    subroutine ElastoDynamics1D_assembleRHS(this, x_p, r_p)
        
        class(ElastoDynamics1D_t), intent(inout) :: this
        real(rp),                  intent(in)    :: x_p(this%npoin)
        real(rp),                  intent(inout) :: r_p(this%npoin)

    end subroutine ElastoDynamics1D_assembleRHS

    
    subroutine ElastoDynamics1D_applyFreeMatrixOperator(this, x_p, y_p)
        
        class(ElastoDynamics1D_t), intent(inout) :: this
        real(rp),                  intent(in)    :: x_p(this%npoin)
        real(rp),                  intent(inout) :: y_p(this%npoin)

    end subroutine ElastoDynamics1D_applyFreeMatrixOperator


    subroutine ElastoDynamics1D_cleanVect(this, x_p)
        
        class(ElastoDynamics1D_t),         intent(inout) :: this
        real(rp),                  target, intent(inout) :: x_p(*)

        integer(ip) :: p

        !$acc parallel loop present(x_p) 
        do p = 1, this%npoin
            x_p(p) = 0.0_rp
        end do

    end subroutine ElastoDynamics1D_cleanVect


    subroutine ElastoDynamics1D_computeCoeffs(this)
        
        class(ElastoDynamics1D_t), intent(inout) :: this

        integer(ip) :: e, nelem

        real(rp), contiguous, pointer :: m_e(:)
        real(rp), contiguous, pointer :: k_e(:)

        ! Elemental matrix
        nelem = this%nelem
        k_e => this%stifm
        !$acc parallel loop present(k_e)
        do e = 1, nelem
            k_e(e) = this%E * this%A / this%Le
        end do

        ! Elemental stiffness
        m_e => this%massm
        !$acc parallel loop present(m_e)
        do e = 1, nelem
            m_e(e) = this%rho * this%A * this%Le / 6.0_rp
        end do

    end subroutine ElastoDynamics1D_computeCoeffs


    subroutine ElastoDynamics1D_computeExternalForce(this)
        
        class(ElastoDynamics1D_t), intent(inout) :: this

        integer(ip) :: e, pi, pj, nelem

        real(rp), contiguous, pointer :: m_e(:)
        real(rp), contiguous, pointer :: f_p(:)

        ! Recover some pointers
        nelem = this%nelem
        m_e => this%massm
        f_p => this%fexte

        !$acc parallel loop present(m_e, f_p)
        do e = 1, nelem
            ! Global indices
            pi = e
            pj = e + 1_ip
            ! Gracity force
            !$acc atomic update
            f_p(pi) = f_p(pi) + 3.0_rp * 9.81_rp * m_e(e)
            !$acc atomic update
            f_p(pj) = f_p(pj) + 3.0_rp * 9.81_rp * m_e(e)
        end do

    end subroutine ElastoDynamics1D_computeExternalForce


    subroutine ElastoDynamics1D_updateExternalForce(this)
        
        class(ElastoDynamics1D_t), intent(inout) :: this

        this%fexte(this%npoin) = this%fexte(this%npoin) + this%H * 6.0_rp / 0.25_rp * cos(6.0_rp * this%time/0.25_rp) * this%dtime
        !$acc update device(this%fexte(this%npoin)) 

    end subroutine ElastoDynamics1D_updateExternalForce


    subroutine ElastoDynamics1D_outputNodeSets(this)
        
        class(ElastoDynamics1D_t), intent(inout) :: this

        !$acc update host(this%displ(this%npoin), this%veloc(this%npoin), this%accel(this%npoin)) 
        write(this%out_unit, '(4(E22.15,2x))') this%time, this%displ(this%npoin), this%veloc(this%npoin), this%accel(this%npoin)

    end subroutine ElastoDynamics1D_outputNodeSets



end module mod_ElastoDynamics1D_Base