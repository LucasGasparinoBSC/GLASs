module mod_Diffusion1D_LGL
    use iso_c_binding, only: ip => c_int32_t, rp => c_float, dp => c_double

contains
    subroutine qAndLevaluation(N, x, q, qp, LOut)
        ! Evaluates Nth Legendre polynomial at x
        ! also evaluates q = L_{N+1} - L_{N-1} and its derivative
        integer, intent(in) :: N
        real(rp), intent(in) :: x

        real(rp), intent(out) :: q, qp, LOut
        real(rp), dimension(:), allocatable :: L
        real(rp), dimension(:), allocatable :: Lp

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

    subroutine LegendreGaussLobattoNodeEval(p, xvals, wvals)
        ! calculates the Gauss-Lobatto nodes and weights
        integer :: p ! POLYNOMIAL ORDER
        real(rp) :: xvals(p), wvals(p)

        real(rp) pi
        real(rp) cosarg
        real(rp) tol, delta

        real(rp) q, qp, L
        integer j, k, n_iter

        tol = 3*epsilon(1.)

        n_iter = 100

        if (p .EQ. 1) then
            xvals(1) = -1.
            xvals(2) = 1.
            wvals(1) = 1.
            wvals(2) = wvals(1)
            return
        end if
        xvals(1) = -1.
        wvals(1) = 2./(p*(p + 1))
        xvals(p + 1) = 1.
        wvals(p + 1) = wvals(1)
        pi = 2*acos(0.)

        outer: do j = 2, floor(((REAL(p) + 1.)/2))
            cosarg = ((((j - 1) + 0.25)*pi)/p) - 3.0/(8*p*pi*((j - 1) + 0.25))
            xvals(j) = -cos(cosarg)
            inner: do k = 1, n_iter
                call qAndLevaluation(p, xvals(j), q, qp, L)
                delta = -q/qp
                xvals(j) = xvals(j) + delta
                if (abs(delta) .LE. tol*abs(xvals(j))) then
                    exit inner
                end if
            end do inner
            ! call qAndLevaluation(N, xvals(j), q, qp, L)
            xvals(p - j + 2) = -xvals(j)
            wvals(j) = 2./(p*(p + 1)*L*L)
            wvals(p - j + 2) = wvals(j)
        end do outer
        if (mod(p, 2) .EQ. 0) then
            xvals((p/2) + 1) = 0.
            call qAndLevaluation(p, xvals((p/2) + 1), q, qp, L)
            wvals((p/2) + 1) = 2./(p*(p + 1)*L*L)
        end if
    end subroutine
end module
