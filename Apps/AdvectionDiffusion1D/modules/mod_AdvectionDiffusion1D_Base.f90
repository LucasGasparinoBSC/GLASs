module mod_AdvectionDiffusion1D_Base

    use iso_c_binding
    use cg_wrapper_mod
    use mod_AdvectionDiffusion1D_Jacobian

    implicit none

    type AdvectionDiffusion1D_Base_t
        integer(c_int32_t) :: p, nelem, npoin
        real(c_float), pointer :: jac(:, :)
    contains
        procedure, pass :: matvec => AdvectionDiffusion1D_matvec
    end type

contains

    ! Fortran concrete routine
    subroutine AdvectionDiffusion1D_matvec(this, x_in, x_out)
        implicit none
        class(mod_AdvectionDiffusion1D_Base), intent(inout) :: this
        real(c_float), intent(in)    :: x_in(this%npoin)
        real(c_float), intent(out)   :: x_out(this%npoin)
        integer(4) :: i, r, elemId, outIdx

        ! need to treat x_out differently if p divides i-1
        ! instead of *checking* whether p divides i-1, it's better to "impose" the remainder with p loops

        !$acc parallel loop deviceptr(x_in, x_out) present(this%jac)
        do i = 2, ((this%npoin - 1)/this%p)
            outIdx = this%p*i + 1
            x_out(outIdx) = this%jac(1)*x_in(outIdx:outIdx + this%p) &
                            + this%jac(this%p + 1)*x_in(outIdx - this%p:outIdx)
        end do
        !$acc end parallel loop
        !$acc parallel loop deviceptr(x_in, x_out) present(this%jac) collapse(2)
        do i = 1, (this%npoin/this%p) - 1
            do r = 1, this%p - 1
                elemStartIdx = this%p*i + 1
                x_out(elemStartIdx + r) = this%jac(r + 1)*x_in(elemStartIdx:elemStartIdx + this%p)
            end do
        end do
        !$acc end parallel loop

    end subroutine AdvectionDiffusion1D_matvec

    ! C wrapper for the Fortran matvec
    subroutine AdvectionDiffusion1D_matvec_c(x_in, x_out, user_data) bind(C)
        use iso_c_binding, only: c_ptr, c_float
        implicit none
        real(c_float), intent(in)  :: x_in(*)
        real(c_float), intent(out) :: x_out(*)
        type(c_ptr), value       :: user_data

        type(AdvectionDiffusion1D_Base_t), pointer :: dMatrix

        call c_f_pointer(user_data, dMatrix)
        call dMatrix%matvec(x_in, x_out)

    end subroutine AdvectionDiffusion1D_matvec_c

    subroutine AdvectionDiffusion1D_solve(this)
        integer(c_int32_t), parameter :: p = 5    ! CG polynomial order
        integer(c_int32_t), parameter :: nelems = 200
        integer(c_int32_t), parameter :: maxIters = 10   ! Max iterations
        real(c_double), parameter :: tol = 1.0e-6    ! Tolerance

        real(c_float), parameter :: viscosity = 0.001_c_float
        real(c_float), parameter :: advectionVelocity = 0.8_c_float
        real(c_float), parameter :: domainSize = 16.0_c_float

        type(c_ptr) :: cgSolver, user_data
        type(c_funptr) :: matvec_funptr
        real(c_float), allocatable :: x0(:), b(:), s(:)
        class(AdvectionDiffusion1D_Base_t) :: this
        class(mod_AdvectionDiffusion1D_Jacobian_t) :: jac_class
        integer :: i

        ! Fill user_data
        this%n = p*nelems + 1
        allocate (this%jac(p + 1, p + 1))

        jac_class = mod_AdvectionDiffusion1D_Jacobian_t(p=p, nelem=nelem, &
                                                        advectionVelocity=advectionVelocity, &
                                                        viscosity=viscosity, &
                                                        domainSize=domainSize &
                                                        )
        call jac_class%getLocalJacobian(this%jac)

        !$acc enter data copyin(mat, this%jac)
        user_data = c_loc(this)

        ! Allocate and initialize vectors
        allocate (x0(this%npoin), source=0.001_c_float)
        allocate (b(this%npoin), source=1.0_c_float)
        allocate (s(this%npoin), source=0.0_c_float)
        !$acc enter data copyin(x0, b, s)

        ! Create solver instance and setup
        cgSolver = cg_create_u32_f(this%npoin, maxIters, tol)
        call cg_setup_u32_f(cgSolver, x0, b)

        ! Get function pointer for matvec
        matvec_funptr = c_funloc(AdvectionDiffusion1D_matvec_c)

        ! Solve the system
        call cg_solve_u32_f(cgSolver, matvec_funptr, user_data)

        ! Recover the solution
        call cg_get_solution_u32_f(cgSolver, s)
        !$acc update host(x0)

    end subroutine

end module

program unitt_32

    use iso_c_binding
    use cg_wrapper_mod
    use fortranOps, only: MATDiag_matvec_c, MATDiag_t

    implicit none

    integer(c_int32_t), parameter :: n = 1000000     ! Size of the system
    integer(c_int32_t), parameter :: maxIters = 10   ! Max iterations
    real(c_double), parameter :: tol = 1.0e-6    ! Tolerance

    type(c_ptr) :: cgSolver, user_data
    type(c_funptr) :: matvec_funptr
    real(c_float), allocatable :: x0(:), b(:), s(:)
    type(MATDiag_t), target :: mat
    integer :: i

    ! Fill user_data
    mat%n = n
    allocate (mat%jac(n), source=4.0_c_float)
    !$acc enter data copyin(mat, mat%jac)
    user_data = c_loc(mat)

    ! Allocate and initialize vectors
    allocate (x0(n), source=0.001_c_float)
    allocate (b(n), source=1.0_c_float)
    allocate (s(n), source=0.0_c_float)
    !$acc enter data copyin(x0, b, s)

    print *, 'Metrics in the host before:', norm2(x0), norm2(b), norm2(s)

    ! Create solver instance and setup
    cgSolver = cg_create_u32_f(n, maxIters, tol)
    call cg_setup_u32_f(cgSolver, x0, b)

    ! Get function pointer for matvec
    matvec_funptr = c_funloc(MATDiag_matvec_c)

    ! Solve the system
    call cg_solve_u32_f(cgSolver, matvec_funptr, user_data)

    ! Recover the solution
    call cg_get_solution_u32_f(cgSolver, s)
    !$acc update host(x0)

    print *, 'Metrics in the host after:', norm2(x0), norm2(b), norm2(s)

    ! Testing the solution
    do i = 1, n
        if (abs(s(i) - b(i)/mat%jac(i)) > 1.0e-6_c_float) then
            write (*, *) "Test failed: solution is not correct:", i, s(i), b(i)/mat%jac(i)
            stop 1
        end if
    end do

    write (*, *) "Test passed: solution is approximately correct."
    stop 0; 
end program unitt_32
