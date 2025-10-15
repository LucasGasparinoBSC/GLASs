module mod_ElastoDynamics1D

    use mod_Elastodynamics1D_Base
    use mod_ElastoDynamics1D_IMEX
    use mod_ElastoDynamics1D_Implicit

    implicit none

    public


    contains

    subroutine create_Elastodnyamics1D(type, driver)
        
        character(len=*),                   intent(in)    :: type
        class(ElastoDynamics1D_t), pointer, intent(inout) :: driver

        select case(trim(type))
        case( "IMPLICIT" ); allocate(ElastoDynamics1DImplicit_t :: driver)
        case( "IMEX"     ); allocate(ElastoDynamics1DIMEX_t :: driver)
        case default;
            write(*,*) "ERORR : create_Elastodnyamics1D : Type not programmed"
            stop 1
        end select

    end subroutine create_Elastodnyamics1D
    

end module mod_ElastoDynamics1D