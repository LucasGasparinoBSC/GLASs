module mod_AdvectionDiffusion1D_LGL
    use iso_c_binding

contains
    subroutine qAndLevaluation(N, x, q, qp, LOut)
        ! Evaluates Nth Legendre polynomial at x
        ! also evaluates q = L_{N+1} - L_{N-1} and its derivative
        integer, intent(in) :: N
        real(c_float), intent(in) :: x

        real(c_float), intent(out) :: q, qp, LOut
        real(c_float), dimension(:), allocatable :: L
        real(c_float), dimension(:), allocatable :: Lp

        integer k

        allocate (L(4))
        allocate (Lp(4))

        ! first two entries of L will be used in recursion
        ! only last entry will be returned

        L(1) = 1
        L(2) = x

        Lp(1) = 0
        Lp(2) = 1

        do k = 2, N
            L(3) = (REAL(2*k - 1)/k)*x*L(2) - ((REAL(k - 1)/k)*L(1))
            Lp(3) = Lp(1) + (2*k - 1)*L(2)
            L(1) = L(2)
            L(2) = L(3)
            Lp(1) = Lp(2)
            Lp(2) = Lp(3)

        end do

        k = N + 1
        L(4) = (REAL(2*k - 1)/k)*x*L(3) - ((REAL(k - 1)/k)*L(1))
        Lp(4) = Lp(2) + (2*k - 1)*L(3)
        q = L(4) - L(1)
        qp = Lp(4) - Lp(1)
        LOut = L(3)

        deallocate (L)
        deallocate (Lp)
    end subroutine

    subroutine LegendreGaussLobattoNodeEval(N, xvals, wvals)
        ! calculates the Gauss-Lobatto nodes and weights
        integer :: N
        real(c_float) :: xvals(N), wvals(N)

        real(c_float) pi
        real(c_float) cosarg
        real(c_float) tol, delta

        real(c_float) q, qp, L
        integer j, k, n_iter

        tol = 3*epsilon(1.)

        n_iter = 100

        if (N .EQ. 1) then
            xvals(1) = -1.
            xvals(2) = 1.
            wvals(1) = 1.
            wvals(2) = wvals(1)
            return
        end if
        xvals(1) = -1.
        wvals(1) = 2./(N*(N + 1))
        xvals(N + 1) = 1.
        wvals(N + 1) = wvals(1)
        pi = 2*acos(0.)

        outer: do j = 2, floor(((REAL(N) + 1.)/2))
            cosarg = ((((j - 1) + 0.25)*pi)/N) - 3.0/(8*N*pi*((j - 1) + 0.25))
            xvals(j) = -cos(cosarg)
            inner: do k = 1, n_iter
                call qAndLevaluation(N, xvals(j), q, qp, L)
                delta = -q/qp
                xvals(j) = xvals(j) + delta
                if (abs(delta) .LE. tol*abs(xvals(j))) then
                    exit inner
                end if
            end do inner
            ! call qAndLevaluation(N, xvals(j), q, qp, L)
            xvals(N - j + 2) = -xvals(j)
            wvals(j) = 2./(N*(N + 1)*L*L)
            wvals(N - j + 2) = wvals(j)
        end do outer
        if (mod(N, 2) .EQ. 0) then
            xvals((N/2) + 1) = 0.
            call qAndLevaluation(N, xvals((N/2) + 1), q, qp, L)
            wvals((N/2) + 1) = 2./(N*(N + 1)*L*L)
        end if
    end subroutine
end module
