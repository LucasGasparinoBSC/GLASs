module mod_ElastoDynamics1D

    use iso_c_binding, only: ip => c_int32_t, rp => c_float, dp => c_double
    use iso_c_binding, only: c_f_pointer, c_loc, c_funloc
    use cg_wrapper_mod

    implicit none

    type ElastoDynamics1D_t
        ! Physical properties
        real(rp)          :: E     ! Young modulus
        real(rp)          :: A     ! Area
        real(rp)          :: rho   ! Density
        real(rp)          :: L     ! Length
        ! Physical parameters (derived)
        real(rp)          :: Le    ! Element lentgh
        real(rp)          :: c     ! Elastic wave velocity
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
        this%E = 270.0e9_rp
        this%A = 1.0e-4_rp
        this%rho = 7800.0_rp
        this%L = 0.1_rp 
        this%dtime = 1.0e-4_rp
        this%nsteps = 100_ip
        this%beta = 0.25_rp
        this%gamma = 0.5_rp
        this%alpha = 1.0_rp
        this%time = 0.0_rp

        ! Derived types
        denom = this%beta * this%dtime * this%dtime
        if( abs(denom) > tiny(0.0_rp) )then
            this%alpha = 1.0_rp / denom
        endif

        ! Nodes per element
        this%nnode = 2_ip

        ! Elemental arrays
        this%nelem = 50_ip 
        allocate(this%stifm(this%nelem), source=0.0_rp)
        allocate(this%massm(this%nelem), source=0.0_rp)
        !$acc enter data copyin(this%ke, this%me)

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

        ! SOlver
        this%niter = 10000_ip
        this%toler = 1.0e-12_dp

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

        real(rp), contiguous, pointer :: u1_p(:), u0_p(:)
        real(rp), contiguous, pointer :: v1_p(:), v0_p(:)
        real(rp), contiguous, pointer :: a1_p(:), a0_p(:)
        real(rp), contiguous, pointer :: b_p(:)
        real(rp), contiguous, pointer :: x_p(:)

        integer(ip)  :: n, p
        real(rp) :: cd1, cd2, cv1, cv2, ca1

        type(c_ptr)    :: cgSolver
        type(c_ptr)    :: opData
        type(c_funptr) :: opFunction

        ! Create auxiliar arrays
        allocate(u0_p(this%npoin), source=0.0_rp)
        allocate(v0_p(this%npoin), source=0.0_rp)
        allocate(a0_p(this%npoin), source=0.0_rp)
        allocate(b_p(this%npoin), source=0.0_rp)
        allocate(x_p(this%npoin), source=0.0_rp)
        !$acc enter data copyin(u0_p, v0_p, a1_p, b_p, x_p)

        ! Recover some pointers
        u1_p => this%displ
        v1_p => this%veloc
        a1_p => this%accel

        ! Constant coefficients
        cd1 = this%dtime
        cd2 = this%dtime * this%dtime * (0.5 - this%beta)
        cv1 = this%dtime * (1.0_rp - this%gamma)
        cv2 = this%dtime * this%gamma
        ca1 = this%alpha

        ! Create solver instance and setup
        !cgSolver = cg_create_u32_f(this%npoin, this%niter, this%toler)
        !call cg_setup_u32_f(cgSolver, u0_p, b_p)

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
            ! Compute predictors
            do p = 1, this%npoin
                u1_p(p) = u0_p(p) + cd1 * v1_p(p) + cd2 * a0_p(p)
                v1_p(p) = v0_p(p) + cv1 * a0_p(p)
            end do
            !
            ! Assemble RHS
            !   b = fext + alpha * M * u
            call this%cleanVect(b_p)
            call this%assembleRHS(u1_p, b_p)
            !
            ! Solve (K + alpha M) u_{n+1} = rhs  on free set
            cgSolver = cg_create_u32_f(this%npoin, this%niter, this%toler)
            call cg_setup_u32_f(cgSolver, u1_p, b_p)
            call cg_solve_u32_f(cgSolver, opFunction, opData)
            call cg_get_solution_u32_f(cgSolver, x_p)
            call cg_destroy_u32_f(cgSolver)
            !
            ! Correct predictions
            do p = 1, this%npoin
                a1_p(p) = ca1 * (x_p(p) - u1_p(p))
                v1_p(p) = v0_p(p) + cv2 * a0_p(p)
                u1_p(p) = x_p(p)
            end do

            write(99,*) u1_p(this%npoin), v1_p(this%npoin), a1_p(this%npoin)

            ! Update time arrays
            u0_p = u1_p
            v0_p = v1_p
            a0_p = a1_p

        end do

        ! Free solver
        !call cg_destroy_u32_f(cgSolver)

        ! Free auxiliar arrays
        deallocate(u0_p)
        deallocate(v0_p)
        deallocate(a0_p)
        deallocate(b_p)
        
    end subroutine ElastoDynamics1D_run

    
    subroutine ElastoDynamics1D_assembleRHS(this, u_p, b_p)
        
        class(ElastoDynamics1D_t), intent(inout) :: this
        real(rp),                  intent(in)    :: u_p(this%npoin)
        real(rp),                  intent(inout) :: b_p(this%npoin)

        integer(ip) :: e, p, pi, pj
        real(rp)    :: ui, uj 
        real(rp)    :: bi, bj

        real(rp), contiguous, pointer :: m_e(:)
        real(rp), contiguous, pointer :: f_p(:)

        ! Add intertial force contribution
        m_e => this%massm
        !$-acc parallel loop present(u,y,) gang vector
        do e = 1, this%nelem
            ! Global indices
            pi = e
            pj = e + 1_ip
            ! Gather displaceent
            ui = u_p(pi)
            uj = u_p(pj)
            ! Compute elemental contribution
            !    Me * ue = m * [[2,1],[1,2]] * [ui, uj]^T
            bi = this%alpha * m_e(e) * (2.0_rp * ui + uj)
            bj = this%alpha * m_e(e) * (2.0_rp * uj + ui)
            ! Assemble to global
            !$acc atomic update
            b_p(pi) = b_p(pi) + bi
            !$acc atomic update
            b_p(pj) = b_p(pj) + bj
        end do

        ! Add body/external force
        f_p => this%fexte
        !$acc parallel loop present(b_p, f_p)
        do p = 1, this%npoin
            b_p(p) = b_p(p) + f_p(p)
        end do

    end subroutine ElastoDynamics1D_assembleRHS

    
    subroutine ElastoDynamics1D_applyFreeMatrixOperator(this, u_p, y_p)
        
        class(ElastoDynamics1D_t), intent(inout) :: this
        real(rp),                  intent(in)    :: u_p(this%npoin)
        real(rp),                  intent(inout) :: y_p(this%npoin)

        integer(ip) :: e, pi, pj
        real(rp)    :: ui, uj 
        real(rp)    :: bi, bj

        real(rp), contiguous, pointer :: m_e(:)
        real(rp), contiguous, pointer :: k_e(:)

        ! Recover some pointers
        m_e => this%massm
        k_e => this%stifm

        !$-acc parallel loop present(u,y,) gang vector
        do e = 1, this%nelem
            ! Global indices
            pi = e
            pj = e + 1_ip
            ! Gather displaceent
            ui = u_p(pi)
            uj = u_p(pj)
            ! Compute stiffness contribution
            !    Ke * ue = k * [[1,-1],[-1,1]] * [ui, uj]^T
            bi = k_e(e) * (ui - uj)
            bj = k_e(e) * (uj - ui)
            ! Compute mass contribution
            !    Me * ue = m * [[2,1],[1,2]] * [ui, uj]^T
            bi = bi + this%alpha * m_e(e) * (2.0_rp * ui + uj)
            bj = bj + this%alpha * m_e(e) * (2.0_rp * uj + ui)
            ! Assemble to global
            !$acc atomic update
            y_p(pi) = y_p(pi) + bi
            !$acc atomic update
            y_p(pj) = y_p(pj) + bj
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

        ! Recover some pointers
        m_e => this%massm
        k_e => this%stifm

        !$-acc parallel loop present(u,y,) gang vector
        do e = 1, this%nelem
            k_e(e) = this%E * this%A / this%Le
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

        !$-acc parallel loop present(u,y,) gang vector
        do e = 1, this%nelem
            ! Global indices
            pi = e
            pj = e + 1_ip
            ! Gracity force
            f_p(pi) = f_p(pi) + 3.0_rp * 9.81_rp * m_e(e)
            f_p(pj) = f_p(pj) + 3.0_rp * 9.81_rp * m_e(e)
        end do

    end subroutine ElastoDynamics1D_computeExternalForce


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