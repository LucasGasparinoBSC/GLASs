module mod_ElastoDynamics1D_Implicit

    use iso_c_binding, only: ip => c_int32_t, rp => c_float, dp => c_double
    use iso_c_binding, only: c_f_pointer, c_loc, c_funloc
    use cg_wrapper_mod
    use mod_ElastoDynamics1D_Base, only: ElastoDynamics1D_t

    implicit none

    integer(ip) :: GLASS_ED1D_OUT_UNIT = 20210423

    type, extends(ElastoDynamics1D_t) :: ElastoDynamics1DImplicit_t
        real(rp)          :: alpha
        real(rp)          :: beta
        real(rp)          :: gamma 
    contains
        procedure, pass :: free => ElastoDynamics1DImplicit_free
        procedure, pass :: initialize => ElastoDynamics1DImplicit_initialize
        procedure, pass :: solve => ElastoDynamics1DImplicit_solve
        procedure, pass :: assembleRHS => ElastoDynamics1DImplicit_assembleRHS
        procedure, pass :: applyFMOp => ElastoDynamics1DImplicit_applyFreeMatrixOperator
    end type ElastoDynamics1DImplicit_t

    private

    public :: ip, rp, dp
    public :: ElastoDynamics1DImplicit_t

    contains

    subroutine ElastoDynamics1DImplicit_free(this)
        
        class(ElastoDynamics1DImplicit_t), intent(inout) :: this

        call this%free_parent()

        this%beta = 0.0_rp
        this%gamma = 0.0_rp
        this%alpha = 0.0_rp

    end subroutine ElastoDynamics1DImplicit_free


    subroutine ElastoDynamics1DImplicit_initialize(this)
        
        class(ElastoDynamics1DImplicit_t), intent(inout) :: this

        real(rp) :: denom

        ! Free variables: set zero and nullify pointers
        call this%free()

        !$acc enter data copyin(this)

        call this%initialize_parent()

        ! Increate dtime
        this%dtime = 10.0_rp * this%dtime
        this%nsteps = this%nsteps/10_ip

        ! Beta Newmark scheme parameters
        this%beta = 0.25_rp
        this%gamma = 0.5_rp

        denom = this%beta * this%dtime * this%dtime
        if( abs(denom) > tiny(0.0_rp) )then
            this%alpha = 1.0_rp / denom
        else
            this%alpha = 1.0_rp
        endif

        !$acc update device(this%beta, this%gamma, this%alpha)

    end subroutine ElastoDynamics1DImplicit_initialize


    subroutine ElastoDynamics1DImplicit_solve(this)
        
        class(ElastoDynamics1DImplicit_t), intent(inout) :: this

        real(rp), contiguous, pointer :: u1_p(:), u0_p(:)
        real(rp), contiguous, pointer :: v1_p(:), v0_p(:)
        real(rp), contiguous, pointer :: a1_p(:), a0_p(:)
        real(rp), contiguous, pointer :: fe_p(:)
        real(rp), contiguous, pointer :: r_p(:)
        real(rp), contiguous, pointer :: x_p(:)

        integer(ip)  :: n, p, npoin
        real(rp)     :: cd1, cd2, cv1, cv2, ca1

        type(c_ptr)    :: cgSolver
        type(c_ptr)    :: opData
        type(c_funptr) :: opFunction

        ! Create auxiliar arrays
        allocate(u0_p(this%npoin), source=0.0_rp)
        allocate(v0_p(this%npoin), source=0.0_rp)
        allocate(a0_p(this%npoin), source=9.81_rp)
        allocate(r_p(this%npoin), source=0.0_rp)
        allocate(x_p(this%npoin), source=0.0_rp)
        !$acc enter data copyin(u0_p, v0_p, a0_p, r_p, x_p)

        ! Alias variables & pointers
        npoin = this%npoin
        u1_p => this%displ
        v1_p => this%veloc
        a1_p => this%accel
        fe_p => this%fexte

        ! Constant coefficients
        cd1 = this%dtime
        cd2 = this%dtime * this%dtime * (0.5 - this%beta)
        cv1 = this%dtime * (1.0_rp - this%gamma)
        cv2 = this%dtime * this%gamma
        ca1 = this%alpha

        ! Create solver instance and setup
        cgSolver = cg_create_u32_f(this%npoin, this%niter, this%toler)

        ! Get function pointer for matvec
        opFunction = c_funloc(ElastoDynamics1DImplicit_operator_c)

        ! Get pointer for the elastodynamics tpe
        select type( op => this )
        type is( ElastoDynamics1DImplicit_t )
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
            call this%updateExternalForce()
            !
            ! Compute predictors
            !$acc parallel loop present(u0_p, u1_p, v0_p, v1_p, a0_p)
            do p = 1, npoin
                if( p > 1 )then
                    u1_p(p) = u0_p(p) + cd1 * v0_p(p) + cd2 * a0_p(p)
                    v1_p(p) = v0_p(p) + cv1 * a0_p(p)
                else
                    u1_p(p) = 0.0_rp
                    v1_p(p) = 0.0_rp
                endif
            end do
            !
            ! Assemble RHS
            !   b = fext + alpha * M * u
            call this%cleanVect(r_p)
            call this%assembleRHS(u1_p, r_p)
            r_p(1) = 0.0_rp
            !$acc update device(r_p(1))
            !
            ! Solve (K + alpha M) u_{n+1} = rhs  on free set
            !$acc host_data use_device(u1_p, r_p, x_p)
            call cg_setup_u32_f(cgSolver, u1_p, r_p)
            call cg_solve_u32_f(cgSolver, opFunction, opData)
            call cg_get_solution_u32_f(cgSolver, x_p)
            !$acc end host_data
            !
            ! Correct predictions
            !$acc parallel loop present(x_p, u1_p, v1_p, a1_p)
            do p = 1, npoin
                if( p > 1 )then
                    a1_p(p) = ca1 * (x_p(p) - u1_p(p))
                    v1_p(p) = v1_p(p) + cv2 * a1_p(p)
                    u1_p(p) = x_p(p)
                else
                    a1_p(p) = 0.0_rp
                    v1_p(p) = 0.0_rp
                    u1_p(p) = 0.0_rp
                endif
            end do
            !
            ! Output monitoring nodell
            if( mod(n,this%out_freq) == 0_ip )then
                call this%outputNodeSets()
            endif

            ! Update time arrays
            !$acc kernels
            u0_p = u1_p
            v0_p = v1_p
            a0_p = a1_p
            !$acc end kernels

        end do

        ! Free solver
        call cg_destroy_u32_f(cgSolver)

        ! Free auxiliar arrays
        deallocate(u0_p)
        deallocate(v0_p)
        deallocate(a0_p)
        deallocate(r_p)
        
    end subroutine ElastoDynamics1DImplicit_solve

    
    subroutine ElastoDynamics1DImplicit_assembleRHS(this, x_p, r_p)
        
        class(ElastoDynamics1DImplicit_t), intent(inout) :: this
        real(rp),                          intent(in)    :: x_p(this%npoin)
        real(rp),                          intent(inout) :: r_p(this%npoin)

        integer(ip) :: e, p, pi, pj
        integer(ip) :: nelem, npoin
        real(rp)    :: ui, uj 
        real(rp)    :: ri, rj
        real(rp)    :: alpha

        real(rp), contiguous, pointer :: m_e(:)
        real(rp), contiguous, pointer :: f_p(:)

        ! Add intertial force contribution
        nelem = this%nelem
        npoin = this%npoin
        alpha = this%alpha
        m_e => this%massm

        !$acc parallel loop present(m_e, x_p, r_p)
        do e = 1, nelem
            ! Global indices
            pi = e
            pj = e + 1_ip
            ! Gather displaceent
            ui = x_p(pi)
            uj = x_p(pj)
            ! Compute elemental contribution
            !    Me * ue = m * [[2,1],[1,2]] * [ui, uj]^T
            ri = alpha * m_e(e) * (2.0_rp * ui + uj)
            rj = alpha * m_e(e) * (2.0_rp * uj + ui)
            ! Assemble to global
            !$acc atomic update
            if(pi > 1 ) r_p(pi) = r_p(pi) + ri
            !$acc atomic update
            r_p(pj) = r_p(pj) + rj
        end do

        ! Add body/external force
        f_p => this%fexte
        !$acc parallel loop present(r_p, f_p)
        do p = 1, npoin
            r_p(p) = r_p(p) + f_p(p)
        end do

    end subroutine ElastoDynamics1DImplicit_assembleRHS

    
    subroutine ElastoDynamics1DImplicit_applyFreeMatrixOperator(this, x_p, y_p)
        
        class(ElastoDynamics1DImplicit_t), intent(inout) :: this
        real(rp),                          intent(in)    :: x_p(this%npoin)
        real(rp),                          intent(inout) :: y_p(this%npoin)

        integer(ip) :: e, pi, pj, nelem
        real(rp)    :: ui, uj 
        real(rp)    :: yi, yj
        real(rp)    :: alpha

        real(rp), contiguous, pointer :: m_e(:)
        real(rp), contiguous, pointer :: k_e(:)

        ! Recover some pointers
        nelem = this%nelem
        alpha = this%alpha
        m_e => this%massm
        k_e => this%stifm

        !$acc parallel loop deviceptr(x_p, y_p) present(m_e, k_e)
        do e = 1, nelem
            ! Global indices
            pi = e
            pj = e + 1_ip
            ! Gather displaceent
            ui = x_p(pi)
            uj = x_p(pj)
            ! Compute stiffness contribution
            !    Ke * ue = k * [[1,-1],[-1,1]] * [ui, uj]^T
            yi = k_e(e) * (ui - uj)
            yj = k_e(e) * (uj - ui)
            ! Compute mass contribution
            !    Me * ue = m * [[2,1],[1,2]] * [ui, uj]^T
            yi = yi + alpha * m_e(e) * (2.0_rp * ui + uj)
            yj = yj + alpha * m_e(e) * (2.0_rp * uj + ui)
            ! Assemble to global
            !$acc atomic update
            if(pi > 1 ) y_p(pi) = y_p(pi) + yi
            !$acc atomic update
            y_p(pj) = y_p(pj) + yj
        end do

    end subroutine ElastoDynamics1DImplicit_applyFreeMatrixOperator


    subroutine ElastoDynamics1DImplicit_operator_c(x_in, x_out, opData) bind(C)
        
        real(c_float), intent(in)  :: x_in(*)
        real(c_float), intent(out) :: x_out(*)
        type(c_ptr),   value       :: opData
        
        type(ElastoDynamics1DImplicit_t), pointer :: ED1D
        integer(ip) :: p, npoin
 
        call c_f_pointer(opData, ED1D)

        npoin = ED1D%npoin
        !$acc parallel loop deviceptr(x_out) 
        do p = 1, npoin
            x_out(p) = 0.0_rp
        end do

        call ED1D%applyFMOp(x_in, x_out)
 
    end subroutine ElastoDynamics1DImplicit_operator_c


end module mod_ElastoDynamics1D_Implicit
