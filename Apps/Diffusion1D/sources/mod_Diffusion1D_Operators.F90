module mod_Diffusion1D_Operators

    use mod_Diffusion1D_LGL
    use iso_c_binding, only: ip => c_int32_t, rp => c_float, dp => c_double

    type Diffusion1D_Operators_t
        integer(ip) p
        integer(ip) nelem
        real(rp) :: advectionVelocity
        real(rp) :: viscosity
        real(rp) :: domainSize

    contains
        procedure, public :: getLocalOperators => mod_Diffusion1D_Operators_get_local_ops
		procedure, public :: getLocalImplicitOperator => mod_Diffusion1D_Operators_get_implicit_operator


    end type Diffusion1D_Operators_t
contains

    subroutine getBarycentricWeights(p, xvals, weights)
        ! calculates the Lagrange barycentric weights at each xval
		integer(ip), intent(in) :: p
        real(rp), intent(in) :: xvals(p+1)
        real(rp), intent(out) :: weights(p+1)
        integer(ip) :: j, k
        do j = 1, p + 1
            weights(j) = 1
        end do
        do j = 2, p + 1
            do k = 1, j-1
                weights(k) = weights(k)*(xvals(k) - xvals(j))
                weights(j) = weights(j)*(xvals(j) - xvals(k))
            end do
        end do
        do j = 1, p + 1
            weights(j) = 1/weights(j)
        end do
        return
    end subroutine

    subroutine getLocalDerivMatrix(p, nelem, derivMatrix, lagrangeWeights, refXVals)
        implicit none

        integer(ip), intent(in) :: p, nelem
		real(rp), intent(in) :: lagrangeWeights(p + 1), refXVals(p + 1)
        real(rp), intent(out) :: derivMatrix(p + 1, p + 1)

        integer(ip) :: i, j

        derivMatrix = 0
        do i = 1, p + 1
            derivMatrix(i, i) = 0
            do j = 1, p + 1
                if (i /= j) then
                    derivMatrix(i, j) = lagrangeWeights(j)/( &
                                        lagrangeWeights(i)*(refXVals(i) - refXVals(j)) &
                                        )
                    derivMatrix(i, i) = derivMatrix(i, i) - derivMatrix(i, j)

                end if
            end do
        end do
        return
    end subroutine getLocalDerivMatrix

    subroutine mod_Diffusion1D_Operators_get_local_ops(this, jacobian, lglWeights)
		implicit none
        class(Diffusion1D_Operators_t), intent(inout) :: this
        real(rp), intent(out) :: jacobian(this%p + 1, this%p + 1)

        real(rp) :: derivMat(this%p + 1, this%p + 1), diffuseMatrix(this%p + 1, this%p + 1)
        real(rp) :: lglWeights(this%p + 1), xVals(this%p + 1), lagrangeWeights(this%p + 1)
        real(rp) :: deltaX
        integer(ip) :: j

        call LegendreGaussLobattoNodeEval(this%p, xVals, lglWeights)
        call getBarycentricWeights(this%p, xVals, lagrangeWeights)
        call getLocalDerivMatrix(this%p, this%nelem, derivMat, lagrangeWeights, xVals)


		deltaX = this%domainSize/this%nelem

		lglWeights = (deltaX/2) * lglWeights

        do j = 1, this%p + 1
			jacobian(j,:) = lglWeights(j)*derivMat(j,:)
        end do

        jacobian = matmul(transpose(derivMat), jacobian)

        jacobian = this%viscosity*2./deltaX*jacobian

    end subroutine
	subroutine mod_Diffusion1D_Operators_get_implicit_operator(this, deltaT, implOp, lglWeights)
        class(Diffusion1D_Operators_t), intent(inout) :: this

		real(rp) :: implOp(this%p + 1, this%p + 1)
		real(rp) :: weightMat(this%p+1,this%p+1)
		real(rp) :: jacobian(this%p+1,this%p+1)
		real(rp) :: lglWeights(this%p+1)

		integer i

		weightMat = 0 

		call this%getLocalOperators(jacobian, lglWeights)

		do i = 1, this%p + 1
			weightMat(i,i) = lglWeights(i)
		end do

		weightMat = weightMat - deltaT*jacobian
	end subroutine
end module
