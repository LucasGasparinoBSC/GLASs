module cg_wrapper_mod
    use iso_c_binding
    implicit none
    interface
        ! Constructor call
        function cg_create_u32_f(arrSize, maxIters, tol) bind(C, name="cg_create_u32_f")
            import :: c_ptr, c_int32_t, c_double
            implicit none
            integer(c_int32_t), value :: arrSize
            integer(c_int32_t), value :: maxIters
            real(c_double), value :: tol
            type(c_ptr) :: cg_create_u32_f
        end function cg_create_u32_f

        ! Destructor call
        subroutine cg_destroy_u32_f(solver) bind(C, name="cg_destroy_u32_f")
            import :: c_ptr
            implicit none
            type(c_ptr), value :: solver
        end subroutine cg_destroy_u32_f

        ! Setup
        subroutine cg_setup_u32_f(solver, inicond, rhs) bind(C, name="cg_setup_u32_f")
            import :: c_ptr, c_float
            implicit none
            type(c_ptr), value :: solver
            real(c_float), intent(in) :: inicond(*)
            real(c_float), intent(in) :: rhs(*)
        end subroutine cg_setup_u32_f

        ! Solver call
        subroutine cg_solve_u32_f(solver, matvec, user_data) bind(C, name="cg_solve_u32_f")
            import :: c_ptr, c_funptr
            implicit none
            type(c_ptr), value :: solver
            type(c_funptr), value :: matvec
            type(c_ptr), value :: user_data
        end subroutine cg_solve_u32_f

        ! Get solution
        function cg_get_solution_u32_f(solver) bind(C, name="cg_get_solution_u32_f")
            import :: c_ptr, c_float
            implicit none
            type(c_ptr), value :: solver
            real(c_float), pointer :: cg_get_solution_u32_f(:)
        end function cg_get_solution_u32_f

        ! Abstract interface for matvec
        abstract interface
            subroutine matvec_f32(x_in, x_out, user_data) bind(C)
                use iso_c_binding, only : c_ptr
                implicit none
                real(kind=4), intent(in) :: x_in(*)
                real(kind=4), intent(out) :: x_out(*)
                type(c_ptr), value :: user_data
            end subroutine matvec_f32
        end interface
    end interface
end module cg_wrapper_mod

program unitt_32_ConjGrad
    use iso_c_binding
    use cg_wrapper_mod
    implicit none
    integer(c_int32_t), parameter :: nArrays = 10 ! Array size
    type(c_ptr) :: solver
    real(c_float), allocatable, target :: x0(:), b(:)

    ! Create a solver instance
    solver = cg_create_u32_f(nArrays, 100_c_int32_t, 1.0e-6_c_double)

    ! Setup inicond and rhs arrays (example values)
    allocate(x0(nArrays), b(nArrays))
    x0 = 0.001
    b = 1.0
    call cg_setup_u32_f(solver, x0, b)

    ! Destroy the solver instance
    !call cg_destroy_u32_f(solver)
end program unitt_32_ConjGrad