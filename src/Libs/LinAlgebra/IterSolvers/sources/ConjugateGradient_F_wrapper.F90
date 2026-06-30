!module cg_wrapper_mod
!    use iso_c_binding
!    implicit none
!    interface
!        !! ---- uint32_t / float ----
!
!        ! Constructor call
!        function cg_create_u32_f(arrSize, maxIters, tol) bind(C, name="cg_create_u32_f")
!            import :: c_ptr, c_int32_t, c_double
!            implicit none
!            integer(c_int32_t), value :: arrSize
!            integer(c_int32_t), value :: maxIters
!            real(c_double), value :: tol
!            type(c_ptr) :: cg_create_u32_f
!        end function cg_create_u32_f
!
!        ! Parallel constructor call
!        function cg_create_u32_pf(comm, arrSize, maxIters, tol) bind(C, name="cg_create_u32_pf")
!            import :: c_ptr, c_int32_t, c_double
!            implicit none
!            integer(c_int32_t), value :: comm
!            integer(c_int32_t), value :: arrSize
!            integer(c_int32_t), value :: maxIters
!            real(c_double), value :: tol
!            type(c_ptr) :: cg_create_u32_pf
!        end function cg_create_u32_pf
!
!        ! Destructor call
!        subroutine cg_destroy_u32_f(solver) bind(C, name="cg_destroy_u32_f")
!            import :: c_ptr
!            implicit none
!            type(c_ptr), value :: solver
!        end subroutine cg_destroy_u32_f
!
!        ! Setup
!        subroutine cg_setup_u32_f(solver, inicond, rhs) bind(C, name="cg_setup_u32_f")
!            import :: c_ptr, c_float
!            implicit none
!            type(c_ptr), value :: solver
!#ifdef USE_GPU
!            real(c_float), device, intent(in) :: inicond(*)
!            real(c_float), device, intent(in) :: rhs(*)
!#else
!            real(c_float), intent(in) :: inicond(*)
!            real(c_float), intent(in) :: rhs(*)
!#endif
!        end subroutine cg_setup_u32_f
!
!        ! CG solver call
!        subroutine cg_solve_u32_f(solver, matvec, halocomm, user_data) bind(C, name="cg_solve_u32_f")
!            import :: c_ptr, c_funptr
!            implicit none
!            type(c_ptr), value :: solver
!            type(c_funptr), value :: matvec
!            type(c_funptr), value :: halocomm
!            type(c_ptr), value :: user_data
!        end subroutine cg_solve_u32_f
!
!        ! FPCG solver call
!        subroutine fpcg_solve_u32_f(solver, matvec, precond, user_data) bind(C, name="fpcg_solve_u32_f")
!            import :: c_ptr, c_funptr
!            implicit none
!            type(c_ptr), value :: solver
!            type(c_funptr), value :: matvec
!            type(c_funptr), value :: precond
!            type(c_ptr), value :: user_data
!        end subroutine fpcg_solve_u32_f
!
!        ! Get solution
!        subroutine cg_get_solution_u32_f(solver, sol) bind(C, name="cg_get_solution_u32_f")
!            import :: c_ptr, c_float
!            implicit none
!            type(c_ptr), value :: solver
!#ifdef USE_GPU
!            real(c_float), device, intent(out) :: sol(*)
!#else
!            real(c_float), intent(out) :: sol(*)
!#endif
!        end subroutine cg_get_solution_u32_f
!
!        !! ---- uint32_t / double ----
!
!        ! Constructor call
!        function cg_create_u32_d(arrSize, maxIters, tol) bind(C, name="cg_create_u32_d")
!            import :: c_ptr, c_int32_t, c_double
!            implicit none
!            integer(c_int32_t), value :: arrSize
!            integer(c_int32_t), value :: maxIters
!            real(c_double), value :: tol
!            type(c_ptr) :: cg_create_u32_d
!        end function cg_create_u32_d
!
!        ! Parallel constructor call
!        function cg_create_u32_pd(comm, arrSize, maxIters, tol) bind(C, name="cg_create_u32_pd")
!            import :: c_ptr, c_int32_t, c_double
!            implicit none
!            integer(c_int32_t), value :: comm
!            integer(c_int32_t), value :: arrSize
!            integer(c_int32_t), value :: maxIters
!            real(c_double), value :: tol
!            type(c_ptr) :: cg_create_u32_pd
!        end function cg_create_u32_pd
!
!        ! Destructor call
!        subroutine cg_destroy_u32_d(solver) bind(C, name="cg_destroy_u32_d")
!            import :: c_ptr
!            implicit none
!            type(c_ptr), value :: solver
!        end subroutine cg_destroy_u32_d
!
!        ! Setup
!        subroutine cg_setup_u32_d(solver, inicond, rhs) bind(C, name="cg_setup_u32_d")
!            import :: c_ptr, c_double
!            implicit none
!            type(c_ptr), value :: solver
!#ifdef USE_GPU
!            real(c_double), device, intent(in) :: inicond(*)
!            real(c_double), device, intent(in) :: rhs(*)
!#else
!            real(c_double), intent(in) :: inicond(*)
!            real(c_double), intent(in) :: rhs(*)
!#endif
!        end subroutine cg_setup_u32_d
!
!        ! CG solver call
!        subroutine cg_solve_u32_d(solver, matvec, halocomm, user_data) bind(C, name="cg_solve_u32_d")
!            import :: c_ptr, c_funptr
!            implicit none
!            type(c_ptr), value :: solver
!            type(c_funptr), value :: matvec
!            type(c_funptr), value :: halocomm
!            type(c_ptr), value :: user_data
!        end subroutine cg_solve_u32_d
!
!        ! FPCG solver call
!        subroutine fpcg_solve_u32_d(solver, matvec, precond, user_data) bind(C, name="fpcg_solve_u32_d")
!            import :: c_ptr, c_funptr
!            implicit none
!            type(c_ptr), value :: solver
!            type(c_funptr), value :: matvec
!            type(c_funptr), value :: precond
!            type(c_ptr), value :: user_data
!        end subroutine fpcg_solve_u32_d
!
!        ! Get solution
!        subroutine cg_get_solution_u32_d(solver, sol) bind(C, name="cg_get_solution_u32_d")
!            import :: c_ptr, c_double
!            implicit none
!            type(c_ptr), value :: solver
!#ifdef USE_GPU
!            real(c_double), device, intent(out) :: sol(*)
!#else
!            real(c_double), intent(out) :: sol(*)
!#endif
!        end subroutine cg_get_solution_u32_d
!    end interface
!    ! Abstract interface for matvec
!    abstract interface
!        subroutine matvec_f32(x_in, x_out, user_data) bind(C)
!            use iso_c_binding, only : c_ptr
!            implicit none
!            real(kind=4), intent(in) :: x_in(*)
!            real(kind=4), intent(out) :: x_out(*)
!            type(c_ptr), value :: user_data
!        end subroutine matvec_f32
!
!        subroutine matvec_f64(x_in, x_out, user_data) bind(C)
!            use iso_c_binding, only : c_ptr
!            implicit none
!            real(kind=8), intent(in) :: x_in(*)
!            real(kind=8), intent(out) :: x_out(*)
!            type(c_ptr), value :: user_data
!        end subroutine matvec_f64
!
!        subroutine precond_f32(r_in, z_out, user_data) bind(C)
!            use iso_c_binding, only : c_ptr
!            implicit none
!            real(kind=4), intent(in) :: r_in(*)
!            real(kind=4), intent(out) :: z_out(*)
!            type(c_ptr), value :: user_data
!        end subroutine precond_f32
!
!        subroutine precond_f64(r_in, z_out, user_data) bind(C)
!            use iso_c_binding, only : c_ptr
!            implicit none
!            real(kind=8), intent(in) :: r_in(*)
!            real(kind=8), intent(out) :: z_out(*)
!            type(c_ptr), value :: user_data
!        end subroutine precond_f64
!
!        subroutine halocomm_f32(x_inout, user_data) bind(C)
!            use iso_c_binding, only : c_ptr
!            implicit none
!            real(kind=4), intent(inout) :: x_inout(*)
!            type(c_ptr), value :: user_data
!        end subroutine halocomm_f32
!
!        subroutine halocomm_f64(x_inout, user_data) bind(C)
!            use iso_c_binding, only : c_ptr
!            implicit none
!            real(kind=8), intent(inout) :: x_inout(*)
!            type(c_ptr), value :: user_data
!        end subroutine halocomm_f64
!    end interface
!end module cg_wrapper_mod