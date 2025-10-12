module mod_ElastoDynamics1D

    use iso_c_binding, only: ip => c_int32_t, rp => c_float, dp => c_double
    use iso_c_binding, only: c_f_pointer, c_loc, c_funloc
    use cg_wrapper_mod

    implicit none

    integer(ip) :: GLASS_ED1D_OUT_UNIT = 19900502

    type ElastoDynamics1D_t
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
        real(rp)          :: alpha
        real(rp)          :: beta
        real(rp)          :: gamma 
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
        procedure, pass :: free => ElastoDynamics1D_free
        procedure, pass :: initialize => ElastoDynamics1D_initialize
        procedure, pass :: finalize => ElastoDynamics1D_finalize
        procedure, pass :: run => ElastoDynamics1D_run
        procedure, pass :: assembleRHS => ElastoDynamics1D_assembleRHS
        procedure, pass :: cleanVect => ElastoDynamics1D_cleanVect
        procedure, pass :: applyFMOp => ElastoDynamics1D_applyFreeMatrixOperator
        procedure, pass :: computeCoeffs => ElastoDynamics1D_computeCoeffs
        procedure, pass :: computeExternalForce => ElastoDynamics1D_computeExternalForce
        procedure, pass :: outputNodeSets => ElastoDynamics1D_outputNodeSets
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
        this%beta = 0.0_rp
        this%gamma = 0.0_rp
        this%alpha = 0.0_rp
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
        this%L = 1.0_rp 
        this%beta = 0.25_rp
        this%gamma = 0.5_rp
        this%H = 1000.0_rp
        this%w = 5.0e-4_rp

        this%dtime = 1.0e-6_rp
        this%nsteps = 10000_ip
        this%time = 0.0_rp

        ! Derived types
        denom = this%beta * this%dtime * this%dtime
        if( abs(denom) > tiny(0.0_rp) )then
            this%alpha = 1.0_rp / denom
        else
            this%alpha = 1.0_rp
        endif

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
        this%dtime = 0.50_rp * this%le / this%c

        ! SOlver
        this%niter = 100_ip
        this%toler = 1.0e-12_dp

        ! Output
        this%out_freq = 1_ip
        this%out_unit = GLASS_ED1D_OUT_UNIT

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


    subroutine ElastoDynamics1D_run(this)
        
        class(ElastoDynamics1D_t), intent(inout) :: this

        real(rp), contiguous, pointer :: u1_p(:)
        real(rp), contiguous, pointer :: v1_p(:)
        real(rp), contiguous, pointer :: a1_p(:)
        real(rp), contiguous, pointer :: fe_p(:)
        real(rp), contiguous, pointer :: r_p(:)
        real(rp), contiguous, pointer :: x_p(:)
        real(rp), contiguous, pointer :: y_p(:)

        integer(ip)  :: n, p

        type(c_ptr)    :: cgSolver
        type(c_ptr)    :: opData
        type(c_funptr) :: opFunction

        ! Create auxiliar arrays
        allocate(r_p(this%npoin), source=0.0_rp)
        allocate(x_p(this%npoin), source=0.0_rp)
        !$acc enter data copyin(r_p, x_p)

        ! Recover some pointers
        u1_p => this%displ
        v1_p => this%veloc
        a1_p => this%accel
        fe_p => this%fexte

        ! Create solver instance and setup
        cgSolver = cg_create_u32_f(this%npoin, this%niter, this%toler)

        ! Get function pointer for matvec
        opFunction = c_funloc(ElastoDynamics1D_operator_c)

        ! Get pointer for the elastodynamics tpe
        select type( op => this )
        type is( ElastoDynamics1D_t )
            opData = c_loc(op)
        end select

        ! Compute K and M coefficients
        call this%computeCoeffs()

        ! Compute External Force (gravity + body)
        call this%computeExternalForce()

        ! Temporal loop
        do n = 1, this%nsteps
            !
            ! Update time
            this%time = this%time + this%dtime
            !
            ! Update temporal variables
            fe_p(this%npoin) = fe_p(this%npoin) + this%H * 6.0_rp / 0.005_rp * cos(6.0_rp * this%time/0.005_rp) * this%dtime
            !
            ! Update displacement
            !$acc parallel loop present(u1_p, v1_p, a1_p)
            do p = 1, this%npoin
                u1_p(p) = u1_p(p) + this%dtime * (v1_p(p) + 0.5_rp * this%dtime * a1_p(p))
            end do
            u1_p(1) = 0.0_rp
            !
            ! Assemble RHS
            !   r = fext - K * u
            call this%cleanVect(r_p)
            call this%assembleRHS(u1_p, r_p)
            r_p(1) = 0.0_rp
            !
            ! Solve M u_{n+1} = rhs  on free set
            call cg_setup_u32_f(cgSolver, a1_p, r_p)
            call cg_solve_u32_f(cgSolver, opFunction, opData)
            call cg_get_solution_u32_f(cgSolver, x_p)
            a1_p(1) = 0.0_rp
            !
            ! Correct predictions
            !$acc parallel loop present(x_p, v1_p, a1_p)
            do p = 1, this%npoin
                v1_p(p) = v1_p(p) + 0.5_rp * this%dtime * (x_p(p) + a1_p(p))
                a1_p(p) = x_p(p)
            end do
            v1_p(1) = 0.0_rp
            !
            ! Output monitoring node
            if( mod(n,this%out_freq) == 0_ip )then
                call this%outputNodeSets()
            endif

        end do

        ! Free solver
        call cg_destroy_u32_f(cgSolver)

        ! Free auxiliar arrays
        deallocate(x_p)
        deallocate(r_p)
        
    end subroutine ElastoDynamics1D_run

    
    subroutine ElastoDynamics1D_assembleRHS(this, u_p, r_p)
        
        class(ElastoDynamics1D_t), intent(inout) :: this
        real(rp),                  intent(in)    :: u_p(this%npoin)
        real(rp),                  intent(inout) :: r_p(this%npoin)

        integer(ip) :: e, p, pi, pj
        real(rp)    :: ui, uj 
        real(rp)    :: ri, rj

        real(rp), contiguous, pointer :: k_e(:)
        real(rp), contiguous, pointer :: f_p(:)

        ! Add intertial force contribution
        k_e => this%stifm
        !$acc parallel loop present(k_e, u_p, r_p)
        do e = 1, this%nelem
            ! Global indices
            pi = e
            pj = e + 1_ip
            ! Gather displaceent
            ui = u_p(pi)
            uj = u_p(pj)
            ! Compute elemental contribution
            !    re += Ke * ue = ke * [[1,-1],[-1,1]] * [ui, uj]^T
            ri = k_e(e) * (ui - uj)
            rj = k_e(e) * (uj - ui)
            ! Assemble to global
            !$acc atomic update
            r_p(pi) = r_p(pi) - ri
            !$acc atomic update
            r_p(pj) = r_p(pj) - rj
        end do

        ! Add body/external force
        f_p => this%fexte
        !$acc parallel loop present(r_p, f_p)
        do p = 1, this%npoin
            r_p(p) = r_p(p) + f_p(p)
        end do

    end subroutine ElastoDynamics1D_assembleRHS

    
    subroutine ElastoDynamics1D_applyFreeMatrixOperator(this, x_p, y_p)
        
        class(ElastoDynamics1D_t), intent(inout) :: this
        real(rp),                  intent(in)    :: x_p(this%npoin)
        real(rp),                  intent(inout) :: y_p(this%npoin)

        integer(ip) :: e, pi, pj
        real(rp)    :: xi, xj 
        real(rp)    :: yi, yj

        real(rp), contiguous, pointer :: m_e(:)

        ! Recover some pointers
        m_e => this%massm

        !$acc parallel loop present(m_e, x_p, y_p)
        do e = 1, this%nelem
            ! Global indices
            pi = e
            pj = e + 1_ip
            ! Gather acceleration
            xi = x_p(pi)
            xj = x_p(pj)
            ! Compute mass contribution
            !    Me * ae = m * [[2,1],[1,2]] * [ai, aj]^T
            !    x is the nodal acceleration
            yi = m_e(e) * (2.0_rp * xi + xj)
            yj = m_e(e) * (2.0_rp * xj + xi)
            ! Assemble to global
            !$acc atomic update
            y_p(pi) = y_p(pi) + yi
            !$acc atomic update
            y_p(pj) = y_p(pj) + yj
        end do

    end subroutine ElastoDynamics1D_applyFreeMatrixOperator


    subroutine ElastoDynamics1D_cleanVect(this, x_p)
        
        class(ElastoDynamics1D_t), intent(inout) :: this
        real(rp),                  intent(inout) :: x_p(this%npoin)

        integer(ip) :: p

        !$acc parallel loop present(x_p)
        do p = 1, this%npoin
            x_p(p) = 0.0_rp
        end do

    end subroutine ElastoDynamics1D_cleanVect


    subroutine ElastoDynamics1D_computeCoeffs(this)
        
        class(ElastoDynamics1D_t), intent(inout) :: this

        integer(ip) :: e

        real(rp), contiguous, pointer :: m_e(:)
        real(rp), contiguous, pointer :: k_e(:)

        ! Elemental matrix
        k_e => this%stifm
        !$acc parallel loop present(k_e)
        do e = 1, this%nelem
            k_e(e) = this%E * this%A / this%Le
        end do

        ! Elemental stiffness
        m_e => this%massm
        !$acc parallel loop present(m_e)
        do e = 1, this%nelem
            m_e(e) = this%rho * this%A * this%Le / 6.0_rp
        end do

    end subroutine ElastoDynamics1D_computeCoeffs


    subroutine ElastoDynamics1D_computeExternalForce(this)
        
        class(ElastoDynamics1D_t), intent(inout) :: this

        integer(ip) :: e, pi, pj

        real(rp), contiguous, pointer :: m_e(:)
        real(rp), contiguous, pointer :: f_p(:)

        ! Recover some pointers
        m_e => this%massm
        f_p => this%fexte

        !$acc parallel loop present(m_e, f_p)
        do e = 1, this%nelem
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


    subroutine ElastoDynamics1D_outputNodeSets(this)
        
        class(ElastoDynamics1D_t), intent(inout) :: this

        !$acc update host(this%time, this%displ(this%npoin), this%veloc(this%npoin), this%accel(this%npoin))
        write(this%out_unit, '(4(E22.15,2x))') this%time, this%displ(this%npoin), this%veloc(this%npoin), this%accel(this%npoin)

    end subroutine ElastoDynamics1D_outputNodeSets


    subroutine ElastoDynamics1D_operator_c(x_in, x_out, opData) bind(C)
        
        real(c_float), intent(in)  :: x_in(*)
        real(c_float), intent(out) :: x_out(*)
        type(c_ptr),   value       :: opData
        
        type(ElastoDynamics1D_t), pointer :: ED1D
 
        call c_f_pointer(opData, ED1D)
        call ED1D%cleanVect(x_out)
        call ED1D%applyFMOp(x_in, x_out)
 
    end subroutine ElastoDynamics1D_operator_c
    

end module mod_ElastoDynamics1D