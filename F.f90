!! This program is for solving Fokker-Plank equation
!! and to calculate the translocation time of polymer chain
!!/////////////////////////////////////////////

!! This example is for tethered polymer are confined inside a 
!! spere with radius R0. The tethered point is near surfacr of spere.
!!/////////////////////////////////////////////


program F
	! calculation of dynamics of an event from
	! free energy landscape
implicit none

integer*4 max_t
parameter (max_t=100000)
	
integer max_s
parameter (max_s=200)

DOUBLE PRECISION     Ft(0:max_s) ! this is the actual total free energy profile
	
DOUBLE PRECISION     derF(0:max_s)  ! derivative of free energy by reaction coordinate
DOUBLE PRECISION     wmt(0:max_s, 0:max_t) ! W(s, t ) is the probability of a chain of length
	           ! N, with one initial segment in the trans compartment N − 1 segments in cis 
			   ! compartment) at time 0, to have s segments in trans compartment at time t

DOUBLE PRECISION     dt !timestep ; note relationship to the friction coeff

DOUBLE PRECISION     u(1:1000) ! store result of triangular linear system solution
DOUBLE PRECISION     A(1:1000),B(1:1000),C(1:1000),D(1:1000) ! diagonals and the rhs of the same

DOUBLE PRECISION     zN(0:max_s),  pTimeN(0:max_t), pTime0(0:max_t) 
	! translocation and rejection time distributions (s reaches N and s reaches 0 respectively)


DOUBLE PRECISION    time,timeback ! average time of translocation and rejection
	
DOUBLE PRECISION   Translocation_Rate, pT, pF ! probabilities of successful and unsuccesful events

DOUBLE PRECISION    Jreturn(0:max_t), time_p_N(0:max_t) ,time_p_0(0:max_t) !w for s=N and s=0


integer*4  Nt ! number of timesteps
integer*4  N  ! actual number of segments
	
integer*4 i, j ! temporary variables
integer*4 timedisplay,intdisplay
DOUBLE PRECISION   z, mid 
character*24 filein,fileout
open(1,file = 'new_input.txt') 
read(1,*)N,z
read(1,*)Nt,dt,timedisplay,intdisplay
do i=0,N
 read(1,*)zN(i),Ft(i)
end do
close(1)
!	timedisplay=Nt
!	intdisplay=1

time = 0.0d0
timeback=0.0d0
do i = 1,N-1							! calculate the first derivative of Ft by s
   derF(i) = (Ft(i+1)-Ft(i-1))/2.0
enddo   
	! also calculate for first and last segments
derF(0) = Ft(1)-Ft(0) 
derF(N) = Ft(N)-Ft(N-1)

  do i =1,N-1 ! AV: preparing for implicit solution in finite differences
   A(i) = derF(i)*dt/4.0 -dt/2.0
   B(i) = 1.-dt/4.0*(derF(i+1)-derF(i-1))+dt
   C(i) = -(derF(i)/4.0+1./2.)*dt
  enddo

do j = 0, Nt ! initilize wmt not filled later
  wmt(0,j)=0.0d0  ! wmt has not beei initialize anyway
enddo
	
do i = 0, N
    wmt(i,0)=0.0d0
enddo 
    
wmt(1,0) = 1.0d0 ! initial condition: one segment in trans-space at time 0
do j =1,Nt		 ! loop over time	     
    do i = 1,N-1
       D(i) = wmt(i,j-1)+dt/4.0*(derF(i+1)-derF(i-1))*wmt(i,j-1) + &
        dt/4.0* derF(i) * (wmt(i+1,j-1)-wmt(i-1,j-1)) + &
        dt/2.0*(wmt(i+1,j-1)+wmt(i-1,j-1)-2.0*wmt(i,j-1))
    enddo

    call tridag(A, B, C, D, u, N-1)  ! solve linear eq system

    do i =1,N-1 ! store the results and go over again
            wmt(i,j) = u(i)
    enddo

enddo		   !basically that's all!!!!

do j=0, Nt ! loop over time
    pTimeN(j) = -wmt(N,j)+wmt(N-1,j) ! step N-1 to N occuring at time j
    pTime0(j)	= wmt(1,j) - wmt(0,j)  ! step 1 to 0 occuring at time j
enddo
pTime0(0) = 0


do j=0,Nt 
    time_p_N(j)=pTimeN(j)
    time_p_0(j)=pTime0(j)
enddo

	! calculating success and failure probabilities and times
pT = 0
pF = 0
do j=0,Nt
    pT= pT + pTimeN(j)*dt
    pF= pF + pTime0(j)*dt
enddo

Translocation_Rate = pT/(pF+pT)
if (Translocation_Rate < 1e-44) Translocation_Rate = 0
!print *, pT*1e38

mid = 0.0 ! AV^ normallization of probabilities
do j=0,Nt
    mid = mid + pTimeN(j)
    
enddo
do j =0, Nt
  !PRINT *, pTimeN(j)
  !if ((pTimeN(j))<1E-40) PRINT *, pTimeN(j)
  !if (j == 5) PRINT *, j
  
  pTimeN(j) = pTimeN(j)*1.0/mid
  if (abs(pTimeN(j))<1E-44) pTimeN(j) = 0.0
  
  !print *, pTimeN(j)
  
  
enddo
	     
time = 0
	      
do j = 0,Nt
    time =  time + j*pTimeN(j)*dt
enddo 

mid = 0.0
do j=1,Nt
    mid = mid + pTime0(j)
enddo
do j =1, Nt
    
    pTime0(j) = pTime0(j)*1.0/mid
    if (abs(pTime0(j)*1E38)<1E-44) pTime0(j) = 0.0
    !if (abs(pTime0(j)) < 1e-50) pTime0(j) = 0.0
enddo
	     
timeback = 0
	      
do j = 0,Nt
 timeback =  timeback + j*pTime0(j)*dt
enddo 

     
do j=1,Nt
    Jreturn(j)=0
    do i=1,20
        Jreturn(j)=Jreturn(j)+ exp(-3.1416*3.1416*j*i*i/(N*N)) &
        *3.1416*i/N*sin(1.0*3.1416*i/N)*cos(3.1416*i)
    enddo
    Jreturn(j) = -Jreturn(j)*2.0/N   
enddo
print*, 'Fortran runs'
open(2,file = './new_output.txt') 
write(2,'(a,i7,a,e13.6,a,e13.6)') &
        ' N = ',N,' success rate=',abs(Translocation_Rate),' time=', time
write(2,'(a,i7,a,e13.6,a,e13.6)') &
        ' Nt = ',Nt,' failure rate=',1.0-Translocation_Rate,' time=', timeback
do j=0,timedisplay,intdisplay
    write(2,'(2e13.6)') abs(pTimeN(j)),abs(pTime0(j)) !,wmt(1,j),wmt(N-1,j) !,wmt(j,2),wmt(j,1000)
enddo
close(2)

!	enddo		!loop for Kkk
end 



!*********************************************************************** 
!* 
!* the Numerical Recipes routine TRIDAG  (page 40) 
!* 
!*	solves  A v = rhs for v 
!*	where A is a tridiagonal matrix 
!* 
!************************************************************************ 
 
subroutine tridag(lower, diag, upper, rhs, v, n) 
  
implicit none

integer NDIMM, ZDIMM
parameter ( NDIMM=1000, ZDIMM=1000 )

integer	n 
real*8	lower(NDIMM), diag(NDIMM), upper(NDIMM), rhs(NDIMM), v(NDIMM) 
integer	j 
real*8	gam(NDIMM), bet  
  
!	if( diag(1) .eq. 0. ) pause 
 
bet = diag(1) 
v(1) = rhs(1)/bet 
 
do j = 2, n 
    gam(j) = upper(j-1)/bet 
    bet = diag(j) - lower(j)*gam(j) 
!        	if( bet .eq. 0. ) pause 
    v(j) = ( rhs(j)-lower(j)*v(j-1) )/bet 
enddo 
 
do j = n-1, 1, -1 
    v(j) = v(j) - gam(j+1)*v(j+1) 
enddo 
 
return 
end 

