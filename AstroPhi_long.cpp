/*
    AstroPhi code
*/
#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <memory.h>

// Number of MIC-threads
#define MIC_NUM_THREADS 22

// float type
#define real double

// PI value
#define PI 3.141592653589793238462

// Curant-Fredrichs-Levi value
#define CFL 0.2

// Adiabatic index
#define GAMMA (5.0/3.0)

// Minimal density
#define MIN_RHO 2.0e-5

// Computational box
real xm = 3.2;				
real ym = 3.2;
real zm = 3.2;
real ftime = 1.0;		

// Global mesh
int FULL_NX = 16384, 
    FULL_NY = 512, 
    FULL_NZ = FULL_NY;

// Timer 
double start_timer, stop_timer;

// One size of cell
real h  = (xm/FULL_NX);	

// Time count
int iTimeCount = 100;

// Print tau
real print_tau;

// Time step
real tau;				

// solver parameters
real *R, *RVx, *RVy, *RVz;

// special parameters
real *Vx, *Vy, *Vz;

// internal parameters
real *RNext;

// Local mesh
int NX, NY, NZ;

// Number first cell
int i_start_index;

// Number of process and size of topology
int rank, size;		

// Send-Recv buffer
real *buffer;

void exchange(real *a)
{
	int i, k ,l;

        if(size == 1) return;
	// НАЧАЛО РАБОТЫ первого процесса
	if( rank == 0 )			
	{
		// Формируем крайний правый массив под передачу
		for(k=0;k<NY;k++)
			for(l=0;l<NZ;l++)
			{
				i = NX - 2;
				buffer[k*NZ+l] = a[i*NZ*NY+k*NZ+l];
			}
		// Отправляем данные направо и получаем справа
		MPI_Send(buffer,NY*NZ,MPI_FLOAT,rank+1,1001,MPI_COMM_WORLD);
		MPI_Recv(buffer,NY*NZ,MPI_FLOAT,rank+1,999,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

		// Записываем полученные данные как правое граничное условие 
		for(k=0;k<NY;k++)
			for(l=0;l<NZ;l++)
			{
				i = NX - 1;
				a[i*NZ*NY+k*NZ+l] = buffer[k*NZ+l];
			}
			
	}
	// КОНЕЦ РАБОТЫ первого процесса


	// НАЧАЛО РАБОТЫ последнего процесса
	if( rank == size-1 )			
	{

		// Получаем слева
		MPI_Recv(buffer,NY*NZ,MPI_FLOAT,rank-1,1001,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

		// Записываем полученные данные как левое граничное условие 
		for(k=0;k<NY;k++)
			for(l=0;l<NZ;l++)
			{
				i = 0;
				a[i*NZ*NY+k*NZ+l] = buffer[k*NZ+l];
			}


		// Формируем крайний левый массив под передачу
		for(k=0;k<NY;k++)
			for(l=0;l<NZ;l++)
			{
				i = 1;
				buffer[k*NZ+l] = a[i*NZ*NY+k*NZ+l];
			}

		// Отправляем данные налево
		MPI_Send(buffer,NY*NZ,MPI_FLOAT,rank-1,999,MPI_COMM_WORLD);
	}
	// КОНЕЦ РАБОТЫ последнего процесса

	// НАЧАЛО РАБОТЫ серединного процесса
	if( rank!=0 && rank!=size-1 )			
	{
		// Получаем слева
		MPI_Recv(buffer,NY*NZ,MPI_FLOAT,rank-1,1001,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

		// Записываем полученные данные как левое граничное условие 
		for(k=0;k<NY;k++)
			for(l=0;l<NZ;l++)
			{
				i = 0;
				a[i*NZ*NY+k*NZ+l] = buffer[k*NZ+l];
			}

		// Формируем крайний правый массив под передачу
		for(k=0;k<NY;k++)
			for(l=0;l<NZ;l++)
			{
				i = NX - 2;
				buffer[k*NZ+l] = a[i*NZ*NY+k*NZ+l];
			}
		
		// Отправляем данные направо
		MPI_Send(buffer,NY*NZ,MPI_FLOAT,rank+1,1001,MPI_COMM_WORLD);

		// Получаем справа
		MPI_Recv(buffer,NY*NZ,MPI_FLOAT,rank+1,999,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

		// Записываем полученные данные как правое граничное условие 
		for(k=0;k<NY;k++)
			for(l=0;l<NZ;l++)
			{
				i = NX - 1;
				a[i*NZ*NY+k*NZ+l] = buffer[k*NZ+l];
			}

		// Формируем крайний левый массив под передачу
		for(k=0;k<NY;k++)
			for(l=0;l<NZ;l++)
			{
				i = 1;
				buffer[k*NZ+l] = a[i*NZ*NY+k*NZ+l];
			}

		// Отправляем данные налево
		MPI_Send(buffer,NY*NZ,MPI_FLOAT,rank-1,999,MPI_COMM_WORLD);

	}
	// КОНЕЦ РАБОТЫ серединного процесса
	MPI_Barrier(MPI_COMM_WORLD);
}

void boundary(real* a, real value)
{
	int i, k, l;

	// Краевые условия по оси y
	for(i=0;i<NX;i++)
		for(l=0;l<NZ;l++)
		{
			k = 0;
			a[i*NZ*NY+k*NZ+l] = value;
			k = NY-1;
			a[i*NZ*NY+k*NZ+l] = value;
		}

	// Краевые условия по оси z
	for(i=0;i<NX;i++)
		for(k=0;k<NY;k++)
		{
			l = 0;
			a[i*NZ*NY+k*NZ+l] = value;
			l = NZ-1;
			a[i*NZ*NY+k*NZ+l] = value;
		}

	// НАЧАЛО РАБОТЫ первого процесса
	if( rank == 0 )			
	{
		// Краевые условия по оси x ТОЛЬКО СЛЕВА
		for(k=0;k<NY;k++)
			for(l=0;l<NZ;l++)
			{
				i = 0;
				a[i*NZ*NY+k*NZ+l] = value;
			}
	}
	// КОНЕЦ РАБОТЫ первого процесса


	// НАЧАЛО РАБОТЫ последнего процесса
	if( rank == size-1 )			
	{
		// Краевые условия по оси x ТОЛЬКО СПРАВА
		for(k=0;k<NY;k++)
			for(l=0;l<NZ;l++)
			{
				i = NX-1;
				a[i*NZ*NY+k*NZ+l] = value;
			}
	}
	// КОНЕЦ РАБОТЫ последнего процесса

}

void LogLaw(FILE* fout, real timer)
{
	int i, k, l;
	int iplus = ((rank == 0) ? 0 : -1 );
	real x, y, z;
	real dmass, dlvx, dlvy, dlvz, denrg, dvelq;
	real dmassg, dlvxg, dlvyg, dlvzg, denrgg;
	real dkin, dint, dgrav, dking, dintg, dgravg;
	int iStartX = (rank!=0);
	int iEndX   = NX - (rank!=(size-1));
	dmass = 0.0;
	dlvx  = 0.0;
	dlvy  = 0.0;
	dlvz  = 0.0;
	denrg = 0.0;
	dkin  = 0.0;
	dint  = 0.0; 
	dgrav = 0.0; 

	for(i=iStartX ; i<iEndX ; i++)
		for(k=0 ; k<NY ; k++)
			for(l=0 ; l<NZ ; l++)
			{
				// Coordinates
				x = (i+i_start_index+iplus+0.5)*h - 0.5*xm;
				y = (k+0.5)*h - 0.5*ym;
				z = (l+0.5)*h - 0.5*zm;
				
				dvelq = Vx[i*NZ*NY+k*NZ+l] * Vx[i*NZ*NY+k*NZ+l] +
					    Vy[i*NZ*NY+k*NZ+l] * Vy[i*NZ*NY+k*NZ+l] +
						Vz[i*NZ*NY+k*NZ+l] * Vz[i*NZ*NY+k*NZ+l];
				
				// Computational law
				dmass += R[i*NZ*NY+k*NZ+l];

				dlvx  += (y*RVz[i*NZ*NY+k*NZ+l] - z*RVy[i*NZ*NY+k*NZ+l]);
				dlvy  += (z*RVx[i*NZ*NY+k*NZ+l] - x*RVz[i*NZ*NY+k*NZ+l]);
				dlvz  += (x*RVy[i*NZ*NY+k*NZ+l] - y*RVx[i*NZ*NY+k*NZ+l]);
				dkin  += R[i*NZ*NY+k*NZ+l]*dvelq/2.0;
			}
	dmass *= h*h*h;
	dlvx  *= h*h*h;
	dlvy  *= h*h*h;
	dlvz  *= h*h*h;
	denrg *= h*h*h;
	dkin  *= h*h*h;
	dint  *= h*h*h;
	dgrav *= h*h*h;
	MPI_Allreduce(&dmass,&dmassg,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&dlvx,&dlvxg,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&dlvy,&dlvyg,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&dlvz,&dlvzg,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&denrg,&denrgg,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&dkin,&dking,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&dint,&dintg,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&dgrav,&dgravg,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);

	if(rank == 0)
		fprintf(fout,"%f %f %f %f %f %f %f %f %f\n",
				timer, dmassg, dlvxg, dlvyg, dlvzg, dking, dintg, dgravg, denrgg);
}

real dmedium(real* a, int ipm1, int kpm1, int lpm1, 
 	     int i, int k, int l, int NX, int NY, int NZ)
{
	return (	a[ipm1*NZ*NY+kpm1*NZ+lpm1]+
				a[ipm1*NZ*NY+kpm1*NZ+l   ]+
				a[ipm1*NZ*NY+k   *NZ+lpm1]+
				a[ipm1*NZ*NY+k   *NZ+l   ]+
				a[i   *NZ*NY+kpm1*NZ+lpm1]+
				a[i   *NZ*NY+kpm1*NZ+l   ]+
				a[i   *NZ*NY+k   *NZ+lpm1]+
				a[i   *NZ*NY+k   *NZ+l   ] )/8.0;
}

void Advec(real *a, real *anext, real *Vx, real *Vy, real *Vz, int NX, int NY, int NZ)
{
	int i,k,l;
	real dmv = tau/h;
	real FXP,FXM,FYP,FYM,FZP,FZM;     
	real vppp,vppm,vpmp,vpmm;
	real vmpp,vmpm,vmmp,vmmm;         
	real rvppp,rvppm,rvpmp,rvpmm;
	real rvmpp,rvmpm,rvmmp,rvmmm;          

	#pragma omp parallel for default(none) shared(dmv,a,anext,Vx,Vy,Vz,NX,NY,NZ) private(i,k,l,FXP,FXM,FYP,FYM,FZP,FZM,vppp,vppm,vpmp,vpmm,vmpp,vmpm,vmmp,vmmm,rvppp,rvppm,rvpmp,rvpmm,rvmpp,rvmpm,rvmmp,rvmmm) num_threads(MIC_NUM_THREADS) schedule (dynamic)
	for(i=1 ; i<NX-1 ; i++)
	 for(k=1 ; k<NY-1 ; k++)
	  for(l=1 ; l<NZ-1 ; l++)
		{
		 // Axis X flow
             vppp = dmedium(Vx,i+1,k+1,l+1,i,k,l,NX,NY,NZ); 
             vppm = dmedium(Vx,i+1,k+1,l-1,i,k,l,NX,NY,NZ); 
             vpmp = dmedium(Vx,i+1,k-1,l+1,i,k,l,NX,NY,NZ); 
             vpmm = dmedium(Vx,i+1,k-1,l-1,i,k,l,NX,NY,NZ); 
             vmpp = dmedium(Vx,i-1,k+1,l+1,i,k,l,NX,NY,NZ); 
             vmpm = dmedium(Vx,i-1,k+1,l-1,i,k,l,NX,NY,NZ); 
             vmmp = dmedium(Vx,i-1,k-1,l+1,i,k,l,NX,NY,NZ); 
             vmmm = dmedium(Vx,i-1,k-1,l-1,i,k,l,NX,NY,NZ);
             rvppp = a[i*NZ*NY+k*NZ+l];
             rvppm = a[i*NZ*NY+k*NZ+l];
             rvpmp = a[i*NZ*NY+k*NZ+l];
             rvpmm = a[i*NZ*NY+k*NZ+l];
             rvmpp = a[i*NZ*NY+k*NZ+l];
             rvmpm = a[i*NZ*NY+k*NZ+l];
             rvmmp = a[i*NZ*NY+k*NZ+l];
             rvmmm = a[i*NZ*NY+k*NZ+l];
             if( vppp < 0.0 ) rvppp = a[(i+1)*NZ*NY+k*NZ+l];
             if( vppm < 0.0 ) rvppm = a[(i+1)*NZ*NY+k*NZ+l];
             if( vpmp < 0.0 ) rvpmp = a[(i+1)*NZ*NY+k*NZ+l];
             if( vpmm < 0.0 ) rvpmm = a[(i+1)*NZ*NY+k*NZ+l];
             if( vmpp > 0.0 ) rvmpp = a[(i-1)*NZ*NY+k*NZ+l];
             if( vmpm > 0.0 ) rvmpm = a[(i-1)*NZ*NY+k*NZ+l];
             if( vmmp > 0.0 ) rvmmp = a[(i-1)*NZ*NY+k*NZ+l];
             if( vmmm > 0.0 ) rvmmm = a[(i-1)*NZ*NY+k*NZ+l];
             FXP = (vppp*rvppp+vppm*rvppm+vpmp*rvpmp+vpmm*rvpmm)/4.0;
             FXM = (vmpp*rvmpp+vmpm*rvmpm+vmmp*rvmmp+vmmm*rvmmm)/4.0;

             // Axis Y flow
             vppp = dmedium(Vy,i+1,k+1,l+1,i,k,l,NX,NY,NZ); 
             vppm = dmedium(Vy,i+1,k+1,l-1,i,k,l,NX,NY,NZ); 
             vpmp = dmedium(Vy,i+1,k-1,l+1,i,k,l,NX,NY,NZ); 
             vpmm = dmedium(Vy,i+1,k-1,l-1,i,k,l,NX,NY,NZ); 
             vmpp = dmedium(Vy,i-1,k+1,l+1,i,k,l,NX,NY,NZ); 
             vmpm = dmedium(Vy,i-1,k+1,l-1,i,k,l,NX,NY,NZ); 
             vmmp = dmedium(Vy,i-1,k-1,l+1,i,k,l,NX,NY,NZ); 
             vmmm = dmedium(Vy,i-1,k-1,l-1,i,k,l,NX,NY,NZ);
             rvppp = a[i*NZ*NY+k*NZ+l];
             rvppm = a[i*NZ*NY+k*NZ+l];
             rvpmp = a[i*NZ*NY+k*NZ+l];
             rvpmm = a[i*NZ*NY+k*NZ+l];
             rvmpp = a[i*NZ*NY+k*NZ+l];
             rvmpm = a[i*NZ*NY+k*NZ+l];
             rvmmp = a[i*NZ*NY+k*NZ+l];
             rvmmm = a[i*NZ*NY+k*NZ+l];
             if( vppp < 0.0 ) rvppp = a[i*NZ*NY+(k+1)*NZ+l];
             if( vppm < 0.0 ) rvppm = a[i*NZ*NY+(k+1)*NZ+l];
             if( vmpp < 0.0 ) rvmpp = a[i*NZ*NY+(k+1)*NZ+l];
             if( vmpm < 0.0 ) rvmpm = a[i*NZ*NY+(k+1)*NZ+l];
             if( vpmp > 0.0 ) rvpmp = a[i*NZ*NY+(k-1)*NZ+l];
             if( vpmm > 0.0 ) rvpmm = a[i*NZ*NY+(k-1)*NZ+l];
             if( vmmp > 0.0 ) rvmmp = a[i*NZ*NY+(k-1)*NZ+l];
             if( vmmm > 0.0 ) rvmmm = a[i*NZ*NY+(k-1)*NZ+l];
             FYP = (vppp*rvppp+vppm*rvppm+vmpp*rvmpp+vmpm*rvmpm)/4.0;
             FYM = (vpmp*rvpmp+vpmm*rvpmm+vmmp*rvmmp+vmmm*rvmmm)/4.0;

             // Axis Z flow
             vppp = dmedium(Vz,i+1,k+1,l+1,i,k,l,NX,NY,NZ); 
             vppm = dmedium(Vz,i+1,k+1,l-1,i,k,l,NX,NY,NZ); 
             vpmp = dmedium(Vz,i+1,k-1,l+1,i,k,l,NX,NY,NZ); 
             vpmm = dmedium(Vz,i+1,k-1,l-1,i,k,l,NX,NY,NZ); 
             vmpp = dmedium(Vz,i-1,k+1,l+1,i,k,l,NX,NY,NZ); 
             vmpm = dmedium(Vz,i-1,k+1,l-1,i,k,l,NX,NY,NZ); 
             vmmp = dmedium(Vz,i-1,k-1,l+1,i,k,l,NX,NY,NZ); 
             vmmm = dmedium(Vz,i-1,k-1,l-1,i,k,l,NX,NY,NZ);
             rvppp = a[i*NZ*NY+k*NZ+l];
             rvppm = a[i*NZ*NY+k*NZ+l];
             rvpmp = a[i*NZ*NY+k*NZ+l];
             rvpmm = a[i*NZ*NY+k*NZ+l];
             rvmpp = a[i*NZ*NY+k*NZ+l];
             rvmpm = a[i*NZ*NY+k*NZ+l];
             rvmmp = a[i*NZ*NY+k*NZ+l];
             rvmmm = a[i*NZ*NY+k*NZ+l];
             if( vppp < 0.0 ) rvppp = a[i*NZ*NY+k*NZ+(l+1)];
             if( vpmp < 0.0 ) rvpmp = a[i*NZ*NY+k*NZ+(l+1)];
             if( vmpp < 0.0 ) rvmpp = a[i*NZ*NY+k*NZ+(l+1)];
             if( vmmp < 0.0 ) rvmmp = a[i*NZ*NY+k*NZ+(l+1)];
             if( vppm > 0.0 ) rvppm = a[i*NZ*NY+k*NZ+(l-1)];
             if( vpmm > 0.0 ) rvpmm = a[i*NZ*NY+k*NZ+(l-1)];
             if( vmpm > 0.0 ) rvmpm = a[i*NZ*NY+k*NZ+(l-1)];
             if( vmmm > 0.0 ) rvmmm = a[i*NZ*NY+k*NZ+(l-1)];
             FZP = (vppp*rvppp+vpmp*rvpmp+vmpp*rvmpp+vmmp*rvmmp)/4.0;
             FZM = (vppm*rvppm+vpmm*rvpmm+vmpm*rvmpm+vmmm*rvmmm)/4.0;

	     anext[i*NZ*NY+k*NZ+l] = a[i*NZ*NY+k*NZ+l] - dmv*(FXP-FXM) - dmv*(FYP-FYM) - dmv*(FZP-FZM);
		}
}


real ComputationalTau(real timer, real tend)
{
	int i, k, l;
	real maxvcur, maxvgl, vsound, dens;
	maxvcur = 0.0;

	for( i=0 ; i<NX ; i++ )
		for( k=0 ; k<NY ; k++ )
			for( l=0 ; l<NZ ; l++ )
			{
				dens = R[i*NZ*NY+k*NZ+l];
				vsound = sqrt(GAMMA) * pow((double)dens, (double)((GAMMA-1.0)/2.0));
				if( fabs(Vx[i*NZ*NY+k*NZ+l])+vsound > maxvcur ) 
					maxvcur = fabs(Vx[i*NZ*NY+k*NZ+l])+vsound;
				if( fabs(Vy[i*NZ*NY+k*NZ+l])+vsound > maxvcur ) 
					maxvcur = fabs(Vy[i*NZ*NY+k*NZ+l])+vsound;
				if( fabs(Vz[i*NZ*NY+k*NZ+l])+vsound > maxvcur ) 
					maxvcur = fabs(Vz[i*NZ*NY+k*NZ+l])+vsound;
			}
	MPI_Allreduce(&maxvcur,&maxvgl,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);
	tau = CFL * h / maxvgl;
	if( timer + tau >= tend ) return tend - timer;
	else return tau;
}

void EulerStage(real *R, real *RVx, real *RVy, real *RVz, real *Vx, real *Vy, real *Vz, int NX, int NY, int NZ)
{
	int i, k, l;
	real divv;
	real hmic = h, taumic = tau;

	start_timer = MPI_Wtime();
	#pragma omp parallel for default(none) shared(R,RVx,RVy,RVz,Vx,Vy,Vz,NX,NY,NZ,taumic,hmic) private(i,k,l,divv) num_threads(MIC_NUM_THREADS)
	for(i=1 ; i<NX-1 ; i++)
	 for(k=1 ; k<NY-1 ; k++)
	  for(l=1 ; l<NZ-1 ; l++)
		{
			RVx[i*NZ*NY+k*NZ+l] = -taumic*(R[(i+1)*NZ*NY+k*NZ+l]-R[(i-1)*NZ*NY+k*NZ+l])/2.0/hmic + RVx[i*NZ*NY+k*NZ+l];
			RVy[i*NZ*NY+k*NZ+l] = -taumic*(R[i*NZ*NY+(k+1)*NZ+l]-R[i*NZ*NY+(k-1)*NZ+l])/2.0/hmic + RVy[i*NZ*NY+k*NZ+l];
			RVz[i*NZ*NY+k*NZ+l] = -taumic*(R[i*NZ*NY+k*NZ+(l+1)]-R[i*NZ*NY+k*NZ+(l-1)])/2.0/hmic + RVz[i*NZ*NY+k*NZ+l];
		}
	exchange(RVx);
	exchange(RVy);
	exchange(RVz);
	stop_timer = MPI_Wtime();
	printf("Lagrange stage: %lf sec \n",stop_timer - start_timer);

	boundary(RVx,0.0);
	boundary(RVy,0.0);
	boundary(RVz,0.0);
}

void Imp2Vel()
{
	int i, k, l;
	real dsummro, dsummro_std, iro, fRMed_std, fRMed;

	for(i=1 ; i<NX-1 ; i++)
		for(k=1 ; k<NY-1 ; k++)
			for(l=1 ; l<NZ-1 ; l++)
				{
					dsummro = 8.0 * R[(i)*NZ*NY+(k)*NZ+(l)] +
					4.0 * ( R[(i+1)*NZ*NY+(k)*NZ+(l)] + R[(i-1)*NZ*NY+(k)*NZ+(l)] +
							R[(i)*NZ*NY+(k+1)*NZ+(l)] + R[(i)*NZ*NY+(k-1)*NZ+(l)] + 
							R[(i)*NZ*NY+(k)*NZ+(l+1)] + R[(i)*NZ*NY+(k)*NZ+(l-1)] ) +
					2.0 * ( R[(i+1)*NZ*NY+(k+1)*NZ+(l)] + R[(i-1)*NZ*NY+(k+1)*NZ+(l)] +
							R[(i+1)*NZ*NY+(k-1)*NZ+(l)] + R[(i-1)*NZ*NY+(k-1)*NZ+(l)] +
							R[(i)*NZ*NY+(k+1)*NZ+(l+1)] + R[(i)*NZ*NY+(k-1)*NZ+(l+1)] +
							R[(i)*NZ*NY+(k+1)*NZ+(l-1)] + R[(i)*NZ*NY+(k-1)*NZ+(l-1)] +
							R[(i+1)*NZ*NY+(k)*NZ+(l+1)] + R[(i-1)*NZ*NY+(k)*NZ+(l+1)] +
							R[(i+1)*NZ*NY+(k)*NZ+(l-1)] + R[(i-1)*NZ*NY+(k)*NZ+(l-1)] ) +
							R[(i+1)*NZ*NY+(k+1)*NZ+(l+1)] + R[(i-1)*NZ*NY+(k+1)*NZ+(l+1)] +
							R[(i+1)*NZ*NY+(k+1)*NZ+(l-1)] + R[(i-1)*NZ*NY+(k+1)*NZ+(l-1)] +
							R[(i+1)*NZ*NY+(k-1)*NZ+(l+1)] + R[(i-1)*NZ*NY+(k-1)*NZ+(l+1)] +
							R[(i+1)*NZ*NY+(k-1)*NZ+(l-1)] + R[(i-1)*NZ*NY+(k-1)*NZ+(l-1)] +
							64.0 * MIN_RHO;

					dsummro_std = R[(i)*NZ*NY+(k)*NZ+(l)] + MIN_RHO;

					if( R[(i)*NZ*NY+(k)*NZ+(l)] > MIN_RHO )
					{
						fRMed     = 4096*R[(i)*NZ*NY+(k)*NZ+(l)]/dsummro/dsummro;
						fRMed_std = 1.0/dsummro_std; 
					}
					else
					{
						fRMed     = 0.0;
						fRMed_std = 0.0; 
					}

					iro = fRMed_std < fRMed ? fRMed_std : fRMed; 
                
					Vx[i*NZ*NY+k*NZ+l] = RVx[i*NZ*NY+k*NZ+l]*iro;
					Vy[i*NZ*NY+k*NZ+l] = RVy[i*NZ*NY+k*NZ+l]*iro;
					Vz[i*NZ*NY+k*NZ+l] = RVz[i*NZ*NY+k*NZ+l]*iro;
				}

	boundary(Vx,0);
	boundary(Vy,0);
	boundary(Vz,0);
	exchange(Vx);
	exchange(Vy);
	exchange(Vz);
}

void LagrangeStage()
{
	// Advection
	start_timer = MPI_Wtime();
	Advec(R,RNext,Vx,Vy,Vz,NX,NY,NZ);
	exchange(RNext);
	stop_timer = MPI_Wtime();
	printf("Lagrange stage: %lf sec \n",4.0*(stop_timer - start_timer));
	boundary(RNext,0.0);
	memcpy(R,RNext,NX*NY*NZ*sizeof(real));

	Advec(RVx,RNext,Vx,Vy,Vz,NX,NY,NZ);
	boundary(RNext,0.0);
	exchange(RNext);
	memcpy(RVx,RNext,NX*NY*NZ*sizeof(real));

	Advec(RVy,RNext,Vx,Vy,Vz,NX,NY,NZ);
	boundary(RNext,0.0);
	exchange(RNext);
	memcpy(RVy,RNext,NX*NY*NZ*sizeof(real));

	Advec(RVz,RNext,Vx,Vy,Vz,NX,NY,NZ);
	boundary(RNext,0.0);
	exchange(RNext);
	memcpy(RVz,RNext,NX*NY*NZ*sizeof(real));
}

void allocation()
{
	// Allocation memory for hydrodynamical array
	R   = new real[NX*NY*NZ];	
	RVx = new real[NX*NY*NZ];	
	RVy = new real[NX*NY*NZ];	
	RVz = new real[NX*NY*NZ];	
	RNext = new real[NX*NY*NZ];	
	Vx = new real[NX*NY*NZ];
	Vy = new real[NX*NY*NZ];
	Vz = new real[NX*NY*NZ];
}

void deallocation()
{
	// Deallocation memory
	delete R;	
	delete RVx;	
	delete RVy;	
	delete RVz;	
	delete RNext;	
	delete Vx;
	delete Vy;
	delete Vz;
	delete buffer;
}

void parallel_init()
{
	int *number_of_dimension_nx = new int[size];
	int FULL_LOCAL_NX, locN;
    
        for( int i=0 ; i<size ; i++ )
	    number_of_dimension_nx[i] = FULL_NX/size;
	for( int i=0 ; i<FULL_NX%size ; i++ )
	    number_of_dimension_nx[i]++;
        locN = number_of_dimension_nx[rank];
        i_start_index = 0;
	for( int i=0 ; i<rank ; i++ )
		i_start_index += number_of_dimension_nx[i];
		        
	// Computational real local size with overflow
	if( rank == 0 || rank == size-1 )	
		// for first and end processes
		FULL_LOCAL_NX = locN + 1;		
	else
		// median processes
		FULL_LOCAL_NX = locN + 2;		

	NX = FULL_NX; //FULL_LOCAL_NX;
	NY = FULL_NY;
	NZ = FULL_NZ;

	buffer = new real[NY*NZ];
}


void LoadProblem()
{
	int i, k, l;
	int iplus;
	real x, y, z, rad;

	iplus = ((rank == 0) ? 0 : -1 );

	for( i=0 ; i<NX ; i++ )
		for( k=0 ; k<NY ; k++ )
			for( l=0 ; l<NZ ; l++ )
			{
				// Background
				R[i*NZ*NY+k*NZ+l]  = 0.0;
				Vx[i*NZ*NY+k*NZ+l] = 0.0;
				Vy[i*NZ*NY+k*NZ+l] = 0.0;
				Vz[i*NZ*NY+k*NZ+l] = 0.0;

				// Coordinates
				x = (i+i_start_index+iplus+0.5)*h - 0.5*xm;
				y = (k+0.5)*h - 0.5*ym;
				z = (l+0.5)*h - 0.5*zm;
				rad = sqrt(x*x + y*y + z*z ); 

				if( rad <= 1.0 )
				{
					R[i*NZ*NY+k*NZ+l]  = 1.0 - rad;
				}

				// Impulses
				RVx[i*NZ*NY+k*NZ+l] = R[i*NZ*NY+k*NZ+l] * Vx[i*NZ*NY+k*NZ+l];
				RVy[i*NZ*NY+k*NZ+l] = R[i*NZ*NY+k*NZ+l] * Vy[i*NZ*NY+k*NZ+l];
				RVz[i*NZ*NY+k*NZ+l] = R[i*NZ*NY+k*NZ+l] * Vz[i*NZ*NY+k*NZ+l];
			}
	// Exchange arrays
	exchange(R);
	exchange(Vx);
	exchange(Vy);
	exchange(Vz);
	exchange(RVx);
	exchange(RVy);
	exchange(RVz);

}

void Save(real* a, real timer)
{
	int i, k, l;
	real x, y;
	real tp;
	FILE *fout;
	char FileName[256];
	int iplus = ((rank == 0) ? 0 : -1 );
	int iStartX, iEndX;
	
	// Сохранения экваториального среза
	iStartX = (rank!=0);
	iEndX   = NX - (rank!=(size-1));
	sprintf(FileName,"T%f_%02d.datp",timer,rank);
	fout = fopen(FileName,"w");
	for(i=iStartX ; i<iEndX ; i++)
		for(k=0 ; k<NY ; k++)
		{
			x = (i+i_start_index+iplus+0.5)*h - 0.5*xm;
			y = (k+0.5)*h - 0.5*ym;
			tp  = 0.0; 
			
			for(l=0;l<NZ;l++)
				tp += a[i*NZ*NY+k*NZ+l];
			
			fprintf(fout,"%f %f %f\n", x,y,tp/NZ);
		}
	fclose(fout);

}

// Driver function
int main(int argc, char *  argv [])
{	
	// file with conservationa laws
	FILE* fout = NULL;

	// Current time
	real timer = 0.0, tend;

	// Start MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(rank == 0)
		fout = fopen("const.log","w");
	parallel_init();

	// Allocation memory for arrays
	allocation();

	// Load problem
	LoadProblem();

	// Save init
	Save(R,timer);
	
	print_tau = ftime/iTimeCount;
	for(int i=0 ; i<iTimeCount ; i++)
	{
		tend = print_tau * (i+1);
		while(timer < tend)
		{
			tau = ComputationalTau(timer,tend);
			timer += tau;
			EulerStage(R,RVx,RVy,RVz,Vx,Vy,Vz,NX,NY,NZ);
			Imp2Vel();
			LagrangeStage();
			Imp2Vel();
			LogLaw(fout,timer);
		}
		Save(R,timer);
		if(rank == 0) printf("Time = %f\n",timer);
	}
	
	// Deallocation memory
	if(rank == 0)
		fclose(fout);
	deallocation();
	MPI_Finalize();
	return 0;	
}