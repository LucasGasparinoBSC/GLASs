module mod_AdvectionDiffusion1D_Implicit

    use iso_c_binding, only: ip => c_int32_t, rp => c_float, dp => c_double
    use iso_c_binding, only: c_loc, c_funloc
    !use cg_wrapper_mod
    use mod_AdvectionDiffusion1D_Jacobian
    use mod_AdvectionDiffusion1D_Base

    implicit none

    type, extends(AdvectionDiffusion1D_Base_t) :: AdvectionDiffusion1D_Implicit_t
    contains
        procedure, pass :: matvec => AdvectionDiffusion1D_Implicit_matvec
        procedure, pass :: solve => AdvectionDiffusion1D_Implicit_solve
    end type

    private

    public :: AdvectionDiffusion1D_Implicit_t

contains

    ! Fortran concrete routine
    subroutine AdvectionDiffusion1D_Implicit_matvec(this, x_in, x_out)
        class(AdvectionDiffusion1D_Implicit_t), intent(inout) :: this
        real(rp), intent(in)    :: x_in(this%npoin)
        real(rp), intent(out)   :: x_out(this%npoin)
        integer(4) :: i, r, elemId, elemStartIdx, outIdx

        ! need to treat x_out differently if p divides i-1
        ! instead of *checking* whether p divides i-1, it's better to "impose" the remainder with p loops

        !$acc parallel loop deviceptr(x_in, x_out) present(this%localJac)
        do i = 2, ((this%npoin - 1)/this%p)
            outIdx = this%p*i + 1
            x_out(outIdx) = dot_product(this%localJac(1, :), x_in(outIdx:outIdx + this%p)) &
                            + dot_product(this%localJac(this%p + 1, :), x_in(outIdx - this%p:outIdx))
        end do
        !$acc end parallel loop
        !$acc parallel loop deviceptr(x_in, x_out) present(this%localJac) collapse(2)
        do i = 1, (this%npoin/this%p) - 1
            do r = 1, this%p - 1
                elemStartIdx = this%p*i + 1
                x_out(elemStartIdx + r) = dot_product(this%localJac(r + 1,:),x_in(elemStartIdx:elemStartIdx + this%p))
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

        type(AdvectionDiffusion1D_Implicit_t), pointer :: dMatrix

        call c_f_pointer(user_data, dMatrix)
        call dMatrix%matvec(x_in, x_out)

    end subroutine AdvectionDiffusion1D_Implicit_matvec_c

    subroutine AdvectionDiffusion1D_Implicit_solve(this)
        class(AdvectionDiffusion1D_Implicit_t):: this

        type(c_ptr) :: cgSolver, opData
        type(c_funptr) :: matvec_funptr
        real(rp), allocatable :: x0(:), b(:), s(:)

        class(mod_AdvectionDiffusion1D_Jacobian_t), allocatable :: jac_class

        integer :: i

        ! Fill user_data

        allocate (this%localJac(this%p + 1, this%p + 1))

        jac_class = mod_AdvectionDiffusion1D_Jacobian_t(p=this%p, nelem=this%nelem, &
                                                        advectionVelocity=this%advectionVelocity, &
                                                        viscosity=this%viscosity, &
                                                        domainSize=this%domainSize &
                                                        )
        call jac_class%getLocalJacobian(this%localJac)

        !$acc enter data copyin(mat, this%localJac)
        ! Get pointer for the elastodynamics tpe
        select type (op => this)
        type is (AdvectionDiffusion1D_Implicit_t)
            opData = c_loc(op)
        end select

        ! Allocate and initialize vectors
        allocate (x0(this%npoin), source=0.001_rp)
        allocate (b(this%npoin), source=1.0_rp)
        allocate (s(this%npoin), source=0.0_rp)
        !$acc enter data copyin(x0, b, s)

        ! Create solver instance and setup
        cgSolver = cg_create_u32_f(this%npoin, this%maxIters, this%tol)
        call cg_setup_u32_f(cgSolver, x0, b)

        ! Get function pointer for matvec
        matvec_funptr = c_funloc(AdvectionDiffusion1D_Implicit_matvec_c)

        ! Solve the system
        call cg_solve_u32_f(cgSolver, matvec_funptr, opData)

        ! Recover the solution
        call cg_get_solution_u32_f(cgSolver, s)
        !$acc update host(x0)

    end subroutine

end module
