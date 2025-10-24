program app_ElastoDynamics1D

    use mod_ElastoDynamics1D, only: ElastoDynamics1D_t, create_Elastodnyamics1D

    implicit none
        
    class(ElastoDynamics1D_t), pointer :: ED1D

    call create_Elastodnyamics1D('IMEX', ED1D)
    call ED1D%initialize()
    call ED1D%solve()
    call ED1D%finalize()

end program app_ElastoDynamics1D