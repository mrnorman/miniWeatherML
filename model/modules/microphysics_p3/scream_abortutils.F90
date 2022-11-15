module scream_abortutils

!-------------------------------------------------
!Utility to stop the model in case of
!catastrophic errors
!-------------------------------------------------
implicit none
private

!public subroutines
public :: endscreamrun

contains

  subroutine endscreamrun (msg)
    !-------------------------------------------------
    ! This subroutine will print the optional message
    ! received via optional arg "msg" and stop
    ! the simulation
    !-------------------------------------------------
    implicit none

    !intent-ins
    character(len=*), intent(in), optional :: msg
    !Stop the model when run in non-MPI mode
    write(*,*)'ERROR: Aborting...'
    if(present(msg)) write(*,*)trim(adjustl(msg))
    call abort()
  end subroutine endscreamrun

end module scream_abortutils
