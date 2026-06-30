module mod_ElastoDynamics1D_IMEX

    use iso_c_binding, only: ip => c_int32_t, rp => c_float, dp => c_double
    use iso_c_binding, only: c_loc, c_funloc
    use cg_wrapper_mod
    use mod_ElastoDynamics1D_Base, only: ElastoDynamics1D_t

    implicit none

    integer(ip) :: GLASS_ED1D_OUT_UNIT = 19900502

    type, extends(ElastoDynamics1D_t) :: ElastoDynamics1DIMEX_t
    contains
        procedure, pass :: assembleRHS => ElastoDynamics1DIMEX_assembleRHS
        procedure, pass :: applyFMOp => ElastoDynamics1DIMEX_applyFreeMatrixOperator
        procedure, pass :: solve => ElastoDynamics1DIMEX_solve
    end type ElastoDynamics1DIMEX_t

    private

    public :: ElastoDynamics1DIMEX_t

    contains


    subroutine ElastoDynamics1DIMEX_solve(this)
        
        class(ElastoDynamics1DIMEX_t), intent(inout) :: this

        real(rp), contiguous, pointer :: u1_p(:)
        real(rp), contiguous, pointer :: v1_p(:)
        real(rp), contiguous, pointer :: a1_p(:)
        real(rp), contiguous, pointer :: fe_p(:)
        real(rp), contiguous, pointer :: r_p(:)
        real(rp), contiguous, pointer :: x_p(:)
        real(rp), contiguous, pointer :: y_p(:)

        integer(ip)  :: n, p, npoin
        real(rp)     :: dtime

        type(c_ptr)    :: cgSolver
        type(c_ptr)    :: opData
        type(c_funptr) :: opFunction

        ! Create auxiliar arrays
        allocate(r_p(this%npoin), source=0.0_rp)
        allocate(x_p(this%npoin), source=0.0_rp)
        !$acc enter data copyin(r_p, x_p)

        ! Alias variables & pointers
        npoin = this%npoin
        dtime = this%dtime
        u1_p => this%displ
        v1_p => this%veloc
        a1_p => this%accel
        fe_p => this%fexte

        ! Create solver instance and setup
        cgSolver = cg_create_u32_f(this%npoin, this%niter, this%toler)

        ! Get function pointer for matvec
        opFunction = c_funloc(ElastoDynamics1DIMEX_operator_c)

        ! Get pointer for the elastodynamics tpe
        select type( op => this )
        type is( ElastoDynamics1DIMEX_t )
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
            ! Update displacement
            !$acc parallel loop present(u1_p, v1_p, a1_p)
            do p = 1, npoin
                if( p > 1 )then
                    u1_p(p) = u1_p(p) + dtime * (v1_p(p) + 0.5_rp * dtime * a1_p(p))
                else
                    u1_p(p) = 0.0_rp
                endif
            end do
            !
            ! Assemble RHS
            !   r = fext - K * u
            call this%cleanVect(r_p)
            call this%assembleRHS(u1_p, r_p)
            r_p(1) = 0.0_rp
            !$acc update device(r_p(1))
            !
            ! Solve M u_{n+1} = rhs on free set
            !$acc host_data use_device(u1_p, r_p, x_p)
            call cg_setup_u32_f(cgSolver, u1_p, r_p)
            call cg_solve_u32_f(cgSolver, opFunction, opData)
            call cg_get_solution_u32_f(cgSolver, x_p)
            !$acc end host_data
            !
            ! Correct predictions
            !$acc parallel loop present(x_p, v1_p, a1_p)
            do p = 1, npoin
                if( p > 1 )then
                    v1_p(p) = v1_p(p) + 0.5_rp * dtime * (x_p(p) + a1_p(p))
                    a1_p(p) = x_p(p)
                else
                    v1_p(p) = 0.0_rp
                    a1_p(p) = 0.0_rp
                endif
            end do
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
        
    end subroutine ElastoDynamics1DIMEX_solve

    
    subroutine ElastoDynamics1DIMEX_assembleRHS(this, x_p, r_p)
        
        class(ElastoDynamics1DIMEX_t), intent(inout) :: this
        real(rp),                      intent(in)    :: x_p(this%npoin)
        real(rp),                      intent(inout) :: r_p(this%npoin)

        integer(ip) :: e, p, pi, pj, nelem
        real(rp)    :: ui, uj 
        real(rp)    :: ri, rj

        real(rp), contiguous, pointer :: k_e(:)
        real(rp), contiguous, pointer :: f_p(:)

        ! Add intertial force contribution
        nelem = this%nelem
        k_e => this%stifm
        !$acc parallel loop present(k_e, x_p, r_p)
        do e = 1, nelem
            ! Global indices
            pi = e
            pj = e + 1_ip
            ! Gather displaceent
            ui = x_p(pi)
            uj = x_p(pj)
            ! Compute elemental contribution
            !    re += Ke * ue = ke * [[1,-1],[-1,1]] * [ui, uj]^T
            ri = k_e(e) * (ui - uj)
            rj = k_e(e) * (uj - ui)
            ! Assemble to global
            !$acc atomic update
            r_p(pi) = r_p(pi) + ri
            !$acc atomic update
            r_p(pj) = r_p(pj) + rj
        end do

        ! Add body/external force
        f_p => this%fexte
        !$acc parallel loop present(r_p, f_p)
        do p = 1, this%npoin
            r_p(p) = f_p(p) - r_p(p)
        end do

    end subroutine ElastoDynamics1DIMEX_assembleRHS

    
    subroutine ElastoDynamics1DIMEX_applyFreeMatrixOperator(this, x_p, y_p)
        
        class(ElastoDynamics1DIMEX_t), intent(inout) :: this
        real(rp),                      intent(in)    :: x_p(this%npoin)
        real(rp),                      intent(inout) :: y_p(this%npoin)

        integer(ip) :: e, pi, pj, nelem
        real(rp)    :: xi, xj 
        real(rp)    :: yi, yj

        real(rp), contiguous, pointer :: m_e(:)

        ! Recover some pointers
        nelem = this%nelem
        m_e => this%massm

        !$acc parallel loop deviceptr(x_p, y_p) present(m_e)
        do e = 1, nelem
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

    end subroutine ElastoDynamics1DIMEX_applyFreeMatrixOperator


    subroutine ElastoDynamics1DIMEX_operator_c(x_in, x_out, opData) bind(C)
        
        real(c_float), intent(in)  :: x_in(*)
        real(c_float), intent(out) :: x_out(*)
        type(c_ptr),   value       :: opData
        
        type(ElastoDynamics1DIMEX_t), pointer :: ED1D
        integer(ip) :: p, npoin
 
        call c_f_pointer(opData, ED1D)
        
        npoin = ED1D%npoin
        !$acc parallel loop deviceptr(x_out) 
        do p = 1, npoin
            x_out(p) = 0.0_rp
        end do

        call ED1D%applyFMOp(x_in, x_out)

    end subroutine ElastoDynamics1DIMEX_operator_c
    

end module mod_ElastoDynamics1D_IMEX