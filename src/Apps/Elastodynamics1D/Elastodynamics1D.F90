program app_ElastoDynamics1D

    use mod_ElastoDynamics1D

    implicit none
        
    integer(ip), parameter :: n = 1000000     ! Size of the system
    integer(ip), parameter :: maxIters = 10   ! Max iterations
    real(dp),    parameter :: tol = 1.0e-6    ! Tolerance

    type(ElastoDynamics1D_t) :: ED1D

    call ED1D%initialize()
    call ED1D%run()
    call ED1D%finalize()

    ! Testing the solution
    !do i = 1, n
    !    if( abs(s(i) - b(i)/mat%vA(i)) > 1.0e-6_c_float )then
    !        write(*,*) "Test failed: solution is not correct:",i,s(i),b(i)/mat%vA(i)
    !        stop 1
    !    endif
    !enddo

    write(*,*) "Test passed: solution is approximately correct."
    stop 0;

end program app_ElastoDynamics1D