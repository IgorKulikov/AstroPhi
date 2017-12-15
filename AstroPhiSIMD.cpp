/* 
	AstroPhi SIMD 
	HLL-method for SIMD extension 
*/
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<immintrin.h>
#include<omp.h>
#define ALIGN 64

// Number of MIC-threads
#define MIC_NUM_THREADS 240

const double xm = 6.4;	// domain [xm * ym * zm] * Time
const double ym = 3.2;	
const double zm = 3.2;	
const double Time = 1.0;

#define NX 1056			// computational mesh
#define NY 100
#define NZ 100
#define NHYDRO 8		// number of hydrodynamics parameter

#define GAMMA (5.0/3.0)	// adiabatic index
#define CFL 0.2			// CFL number

#define RHO_INDEX 0		// total density
#define RH2_INDEX 1		// density of molecular hydrogen
#define RHP_INDEX 2		// density of hydrogen ion H+
#define RHM_INDEX 3		// density of hydrogen ion H-
#define RVX_INDEX 4		// moment of impulse
#define RVY_INDEX 5
#define RVZ_INDEX 6
#define RSE_INDEX 7		// entropy

double *U, *Unew, *FX, *FY, *FZ, *Physics, *Sound;

inline int index(int i, int k, int l, int HydroIndex)
{
	return i * NHYDRO*NZ*NY + k * NHYDRO*NZ + l * NHYDRO + HydroIndex;
}

double random(double tmin, double tmax)
{
	double hloc = tmax - tmin;
	return tmin + ((double)rand())/((double)RAND_MAX) * hloc;
}


int main()
{
	int i, k, l;
	double x, y, z, h, tau, timer, maxvel, curvel, rad, tstart, tend; 
	double init_rho, init_ent, init_vel, fpress, fvx, fvy, fvz, locsnd;
	FILE *fout = fopen("solution.dat","w");
	
	// _m512 type
	int ivector;
	__m512d vecsl, vecsr, vecfp, vecum, vecup;
	__m512d xp_vecsl, xp_vecsr, xp_vecincs, xp_vecfp, xp_vecfm, xp_vecup, xp_vecum;
	__m512d xm_vecsl, xm_vecsr, xm_vecincs, xm_vecfp, xm_vecfm, xm_vecup, xm_vecum;
	__m512d yp_vecsl, yp_vecsr, yp_vecincs, yp_vecfp, yp_vecfm, yp_vecup, yp_vecum;
	__m512d ym_vecsl, ym_vecsr, ym_vecincs, ym_vecfp, ym_vecfm, ym_vecup, ym_vecum;
	__m512d zp_vecsl, zp_vecsr, zp_vecincs, zp_vecfp, zp_vecfm, zp_vecup, zp_vecum;
	__m512d zm_vecsl, zm_vecsr, zm_vecincs, zm_vecfp, zm_vecfm, zm_vecup, zm_vecum;

	__m512d FXP, FXM, FYP, FYM, FZP, FZM;


	/* allocate memory */
	// (double*)_mm_malloc(NX*NY*NZ*NHYDRO*sizeof(double), 64);
	U    = (double*)_mm_malloc(NX*NY*NZ*NHYDRO*sizeof(double), ALIGN);
	Unew = (double*)_mm_malloc(NX*NY*NZ*NHYDRO*sizeof(double), ALIGN);
	FX   = (double*)_mm_malloc(NX*NY*NZ*NHYDRO*sizeof(double), ALIGN);
	FY   = (double*)_mm_malloc(NX*NY*NZ*NHYDRO*sizeof(double), ALIGN);
	FZ   = (double*)_mm_malloc(NX*NY*NZ*NHYDRO*sizeof(double), ALIGN);
	Physics = (double*)_mm_malloc(NX*NY*NZ*NHYDRO*sizeof(double), ALIGN);
 	Sound   = (double*)_mm_malloc(NX*NY*NZ*NHYDRO*sizeof(double), ALIGN); 
	
	/* formulation problem */
	h = xm/NX;
	for(i=0;i<NX;i++)
	 for(k=0;k<NY;k++)
	  for(l=0;l<NZ;l++)
	  {
		x = i*h+0.5*h-0.5*xm;
		y = k*h+0.5*h-0.5*ym;
		z = l*h+0.5*h-0.5*zm;
		init_rho = 1.0;
		init_ent = 1.0;
		init_vel = 1.0;
		rad = sqrt(x*x+y*y+z*z);
		if(rad < 1.0)
		{
			init_rho = 1.1;
			init_vel = 0.0;
		}

		U[index(i,k,l,RHO_INDEX)] = init_rho;
		U[index(i,k,l,RH2_INDEX)] = init_rho;
		U[index(i,k,l,RHP_INDEX)] = init_rho;
		U[index(i,k,l,RHM_INDEX)] = init_rho;
		U[index(i,k,l,RVX_INDEX)] = init_rho * (init_vel + random(-0.1,0.1));
		U[index(i,k,l,RVY_INDEX)] = init_rho * random(-0.01,0.01);
		U[index(i,k,l,RVZ_INDEX)] = init_rho * random(-0.01,0.01);
		U[index(i,k,l,RSE_INDEX)] = init_rho * init_ent;
		
		for(ivector=0;ivector<8;ivector++)
		{
			Physics[index(i,k,l,ivector)] = 0.0;
			Unew[index(i,k,l,ivector)] = 0.0;
		}
		
	  }

	/* computational */
	timer = 0.0;
	
	while(timer < Time)
	{
		// CFL condition
		maxvel = 0.0;
		for(i=1;i<NX-1;i++)
		 for(k=1;k<NY-1;k++)
		  for(l=1;l<NZ-1;l++)
		  {
			curvel = fabs(U[index(i,k,l,RVX_INDEX)]/U[index(i,k,l,RHO_INDEX)]) + 
					 fabs(U[index(i,k,l,RVY_INDEX)]/U[index(i,k,l,RHO_INDEX)]) + 
					 fabs(U[index(i,k,l,RVZ_INDEX)]/U[index(i,k,l,RHO_INDEX)]) + 
					 sqrt(GAMMA*U[index(i,k,l,RSE_INDEX)]*
							pow(U[index(i,k,l,RHO_INDEX)],GAMMA-2.0));
			if(maxvel < curvel) maxvel = curvel;
		  }
		tau = CFL * h / maxvel;
		if(timer + tau > Time) tau = Time - timer;

		// flux computing
		for(i=0;i<NX;i++)
		 for(k=0;k<NY;k++)
		  for(l=0;l<NZ;l++)
		  {
		    	fpress = U[index(i,k,l,RSE_INDEX)]*pow(U[index(i,k,l,RHO_INDEX)],GAMMA-1.0); 
			fvx = U[index(i,k,l,RVX_INDEX)]/U[index(i,k,l,RHO_INDEX)];
			fvy = U[index(i,k,l,RVY_INDEX)]/U[index(i,k,l,RHO_INDEX)];
			fvz = U[index(i,k,l,RVZ_INDEX)]/U[index(i,k,l,RHO_INDEX)];
			locsnd = sqrt(GAMMA*U[index(i,k,l,RSE_INDEX)]*pow(U[index(i,k,l,RHO_INDEX)],GAMMA-2.0));

			Sound[index(i,k,l,0)] = (fvx + locsnd) < 0.0 ? 0.0 : (fvx + locsnd);
			Sound[index(i,k,l,1)] = (fvx - locsnd) > 0.0 ? 0.0 : (fvx - locsnd);
			Sound[index(i,k,l,2)] = (fvy + locsnd) < 0.0 ? 0.0 : (fvy + locsnd);
			Sound[index(i,k,l,3)] = (fvy - locsnd) > 0.0 ? 0.0 : (fvy - locsnd);
			Sound[index(i,k,l,4)] = (fvz + locsnd) < 0.0 ? 0.0 : (fvz + locsnd);
			Sound[index(i,k,l,5)] = (fvz - locsnd) > 0.0 ? 0.0 : (fvz - locsnd);
			Sound[index(i,k,l,6)] = locsnd;
			Sound[index(i,k,l,7)] =-locsnd;

			FX[index(i,k,l,RHO_INDEX)] = U[index(i,k,l,RVX_INDEX)];
			FX[index(i,k,l,RH2_INDEX)] = U[index(i,k,l,RVX_INDEX)];
			FX[index(i,k,l,RHP_INDEX)] = U[index(i,k,l,RVX_INDEX)];
			FX[index(i,k,l,RHM_INDEX)] = U[index(i,k,l,RVX_INDEX)];
			FX[index(i,k,l,RVX_INDEX)] = U[index(i,k,l,RVX_INDEX)]*fvx + fpress;
			FX[index(i,k,l,RVY_INDEX)] = U[index(i,k,l,RVY_INDEX)]*fvx;
			FX[index(i,k,l,RVZ_INDEX)] = U[index(i,k,l,RVZ_INDEX)]*fvx;
			FX[index(i,k,l,RSE_INDEX)] = U[index(i,k,l,RSE_INDEX)]*fvx;

			FY[index(i,k,l,RHO_INDEX)] = U[index(i,k,l,RVY_INDEX)];
			FY[index(i,k,l,RH2_INDEX)] = U[index(i,k,l,RVY_INDEX)];
			FY[index(i,k,l,RHP_INDEX)] = U[index(i,k,l,RVY_INDEX)];
			FY[index(i,k,l,RHM_INDEX)] = U[index(i,k,l,RVY_INDEX)];
			FY[index(i,k,l,RVX_INDEX)] = U[index(i,k,l,RVX_INDEX)]*fvy;
			FY[index(i,k,l,RVY_INDEX)] = U[index(i,k,l,RVY_INDEX)]*fvy + fpress;
			FY[index(i,k,l,RVZ_INDEX)] = U[index(i,k,l,RVZ_INDEX)]*fvy;
			FY[index(i,k,l,RSE_INDEX)] = U[index(i,k,l,RSE_INDEX)]*fvy;

			FZ[index(i,k,l,RHO_INDEX)] = U[index(i,k,l,RVZ_INDEX)];
			FZ[index(i,k,l,RH2_INDEX)] = U[index(i,k,l,RVZ_INDEX)];
			FZ[index(i,k,l,RHP_INDEX)] = U[index(i,k,l,RVZ_INDEX)];
			FZ[index(i,k,l,RHM_INDEX)] = U[index(i,k,l,RVZ_INDEX)];
			FZ[index(i,k,l,RVX_INDEX)] = U[index(i,k,l,RVX_INDEX)]*fvz;
			FZ[index(i,k,l,RVY_INDEX)] = U[index(i,k,l,RVY_INDEX)]*fvz;
			FZ[index(i,k,l,RVZ_INDEX)] = U[index(i,k,l,RVZ_INDEX)]*fvz + fpress;
			FZ[index(i,k,l,RSE_INDEX)] = U[index(i,k,l,RSE_INDEX)]*fvz;
		  }

		// hll method (benchmark code)
		tstart = omp_get_wtime();
		#pragma omp parallel for default(none) shared(U,Unew,FX,FY,FZ,Physics,Sound,tau,h) private(i,k,l, vecsl, vecsr, vecfp, vecum, vecup, xp_vecsl, xp_vecsr, xp_vecincs, xp_vecfp, xp_vecfm, xp_vecup, xp_vecum,xm_vecsl, xm_vecsr, xm_vecincs, xm_vecfp, xm_vecfm, xm_vecup, xm_vecum, yp_vecsl, yp_vecsr, yp_vecincs, yp_vecfp, yp_vecfm, yp_vecup, yp_vecum, ym_vecsl, ym_vecsr, ym_vecincs, ym_vecfp, ym_vecfm, ym_vecup, ym_vecum, zp_vecsl, zp_vecsr, zp_vecincs, zp_vecfp, zp_vecfm, zp_vecup, zp_vecum, zm_vecsl, zm_vecsr, zm_vecincs, zm_vecfp, zm_vecfm, zm_vecup, zm_vecum, FXP,FXM,FYP,FYM,FZP,FZM) num_threads(MIC_NUM_THREADS) schedule(dynamic) 
		#pragma ivdep
		for(i=1;i<NX-1;i++)
		 #pragma ivdep
		 for(k=1;k<NY-1;k++)
		  #pragma ivdep
		  for(l=1;l<NZ-1;l++)
		  {
		    	// left interface
			xp_vecsl   = _mm512_set1_pd(Sound[index(i,k,l,1)]);
			xp_vecsr   = _mm512_set1_pd(Sound[index(i+1,k,l,0)]);
			xp_vecincs = _mm512_set1_pd(1.0/(Sound[index(i+1,k,l,0)]-Sound[index(i,k,l,1)]));

			xp_vecfp = _mm512_load_pd(FX+index(i+1,k,l,0));
			xp_vecfm = _mm512_load_pd(FX+index(i,k,l,0));
			xp_vecup = _mm512_load_pd(U+index(i+1,k,l,0));
			xp_vecum = _mm512_load_pd(U+index(i,k,l,0));

			FXP = _mm512_mul_pd(xp_vecincs, 
				_mm512_add_pd(_mm512_sub_pd(_mm512_mul_pd(xp_vecsr,xp_vecfm),_mm512_mul_pd(xp_vecsl,xp_vecfp)),
					      _mm512_mul_pd(_mm512_sub_pd(xp_vecup,xp_vecum),_mm512_mul_pd(xp_vecsl,xp_vecsr))));
		
			// right interface
			xm_vecsl = _mm512_set1_pd(Sound[index(i-1,k,l,1)]);
			xm_vecsr = _mm512_set1_pd(Sound[index(i,k,l,0)]);
			xm_vecincs = _mm512_set1_pd(1.0/(Sound[index(i,k,l,0)]-Sound[index(i-1,k,l,1)]));

			xm_vecfp = _mm512_load_pd(FX+index(i,k,l,0));
			xm_vecfm = _mm512_load_pd(FX+index(i-1,k,l,0));
			xm_vecup = _mm512_load_pd(U+index(i,k,l,0));
			xm_vecum = _mm512_load_pd(U+index(i-1,k,l,0));

			FXM = _mm512_mul_pd(xm_vecincs, 
				_mm512_add_pd(_mm512_sub_pd(_mm512_mul_pd(xm_vecsr,xm_vecfm),_mm512_mul_pd(xm_vecsl,xm_vecfp)),
					      _mm512_mul_pd(_mm512_sub_pd(xm_vecup,xm_vecum),_mm512_mul_pd(xm_vecsl,xm_vecsr))));

			// up interface
			yp_vecsl = _mm512_set1_pd(Sound[index(i,k,l,3)]);
			yp_vecsr = _mm512_set1_pd(Sound[index(i,k+1,l,2)]);
			yp_vecincs = _mm512_set1_pd(1.0/(Sound[index(i,k+1,l,2)]-Sound[index(i,k,l,3)]));

			yp_vecfp = _mm512_load_pd(FY+index(i,k+1,l,0));
			yp_vecfm = _mm512_load_pd(FY+index(i,k,l,0));
			yp_vecup = _mm512_load_pd(U+index(i,k+1,l,0));
			yp_vecum = _mm512_load_pd(U+index(i,k,l,0));

			FYP = _mm512_mul_pd(yp_vecincs, 
				_mm512_add_pd(_mm512_sub_pd(_mm512_mul_pd(yp_vecsr,yp_vecfm),_mm512_mul_pd(yp_vecsl,yp_vecfp)),
					      _mm512_mul_pd(_mm512_sub_pd(yp_vecup,yp_vecum),_mm512_mul_pd(yp_vecsl,yp_vecsr))));
			
			// down interface
			ym_vecsl = _mm512_set1_pd(Sound[index(i,k-1,l,3)]);
			ym_vecsr = _mm512_set1_pd(Sound[index(i,k,l,2)]);
			ym_vecincs = _mm512_set1_pd(1.0/(Sound[index(i,k,l,2)]-Sound[index(i,k-1,l,3)]));

			ym_vecfp = _mm512_load_pd(FY+index(i,k,l,0));
			ym_vecfm = _mm512_load_pd(FY+index(i,k-1,l,0));
			ym_vecup = _mm512_load_pd(U+index(i,k,l,0));
			ym_vecum = _mm512_load_pd(U+index(i,k-1,l,0));

			FYM = _mm512_mul_pd(ym_vecincs, 
				_mm512_add_pd(_mm512_sub_pd(_mm512_mul_pd(ym_vecsr,ym_vecfm),_mm512_mul_pd(ym_vecsl,ym_vecfp)),
					      _mm512_mul_pd(_mm512_sub_pd(ym_vecup,ym_vecum),_mm512_mul_pd(ym_vecsl,ym_vecsr))));

			// top interface
			zp_vecsl = _mm512_set1_pd(Sound[index(i,k,l,5)]);
			zp_vecsr = _mm512_set1_pd(Sound[index(i,k,l+1,4)]);
			zp_vecincs = _mm512_set1_pd(1.0/(Sound[index(i,k,l+1,4)]-Sound[index(i,k,l,5)]));

			zp_vecfp = _mm512_load_pd(FZ+index(i,k,l+1,0));
			zp_vecfm = _mm512_load_pd(FZ+index(i,k,l,0));
			zp_vecup = _mm512_load_pd(U+index(i,k,l+1,0));
			zp_vecum = _mm512_load_pd(U+index(i,k,l,0));

			FZP = _mm512_mul_pd(zp_vecincs, 
				_mm512_add_pd(_mm512_sub_pd(_mm512_mul_pd(zp_vecsr,zp_vecfm),_mm512_mul_pd(zp_vecsl,zp_vecfp)),
					      _mm512_mul_pd(_mm512_sub_pd(zp_vecup,zp_vecum),_mm512_mul_pd(zp_vecsl,zp_vecsr))));

			// bottom interface
			zm_vecsl = _mm512_set1_pd(Sound[index(i,k,l-1,5)]);
			zm_vecsr = _mm512_set1_pd(Sound[index(i,k,l,4)]);
			zm_vecincs = _mm512_set1_pd(1.0/(Sound[index(i,k,l,4)]-Sound[index(i,k,l-1,5)]));

			zm_vecfp = _mm512_load_pd(FZ+index(i,k,l,0));
			zm_vecfm = _mm512_load_pd(FZ+index(i,k,l-1,0));
			zm_vecup = _mm512_load_pd(U+index(i,k,l,0));
			zm_vecum = _mm512_load_pd(U+index(i,k,l-1,0));

			FZM = _mm512_mul_pd(zm_vecincs, 
				_mm512_add_pd(_mm512_sub_pd(_mm512_mul_pd(zm_vecsr,zm_vecfm),_mm512_mul_pd(zm_vecsl,zm_vecfp)),
					      _mm512_mul_pd(_mm512_sub_pd(zm_vecup,zm_vecum),_mm512_mul_pd(zm_vecsl,zm_vecsr))));

			// HLL method
			vecsl = _mm512_set1_pd(tau);
			vecsr = _mm512_set1_pd(-tau/h);
			vecum = _mm512_load_pd(U+index(i,k,l,0));
			vecfp = _mm512_load_pd(Physics+index(i,k,l,0));

			vecup = _mm512_add_pd(vecum,
				  _mm512_add_pd(
				    _mm512_mul_pd(vecsl,vecfp),
				    _mm512_add_pd(
				      _mm512_mul_pd(
				         vecsr,
				         _mm512_sub_pd(FXP,FXM)
				                   ),
				      _mm512_add_pd(
					 _mm512_mul_pd(
					    vecsr,
					    _mm512_sub_pd(FYP,FYM)
					              ),
 					    _mm512_mul_pd(
 					       vecsr,
 					       _mm512_sub_pd(FZP,FZM)
 					                 )
 					              
 					           )
 					         )
 					        ) );
			_mm512_store_pd(Unew+index(i,k,l,0),vecup);
		  }
		tend = omp_get_wtime();
		printf("Time = %lf sec. FLOPS = %lf\n",tend-tstart,(592.0*NX*NY*NZ)/(tend-tstart));
		fflush(stdout);


		// recovery of U vector
	    	for(i=0;i<NX;i++)
		 for(k=0;k<NY;k++)
		  for(l=0;l<NZ;l++)
		   for(ivector=0;ivector<8;ivector++)
				U[index(i,k,l,ivector)] = Unew[index(i,k,l,ivector)];

		// boundary
	    	for(i=1;i<NX-1;i++)
		 for(k=0;k<NY;k++)
		  for(ivector=0;ivector<8;ivector++)
		  {
			U[index(i,k,0,ivector)] = U[index(i,k,NZ-2,ivector)];
			U[index(i,k,NZ-1,ivector)] = U[index(i,k,1,ivector)];
		  }
		  
	    	for(i=1;i<NX-1;i++)
		 for(l=0;l<NZ;l++)
		  for(ivector=0;ivector<8;ivector++)
		  {
			U[index(i,0,l,ivector)] = U[index(i,NY-2,l,ivector)];
			U[index(i,NY-1,l,ivector)] = U[index(i,1,l,ivector)];
		  }
		
	    	for(k=0;k<NY;k++)
		 for(l=0;l<NZ;l++)
		  for(ivector=0;ivector<8;ivector++)
		  {
			U[index(0,k,l,ivector)] = U[index(NX-2,k,l,ivector)];
			U[index(NX-1,k,l,ivector)] = U[index(1,k,l,ivector)];
		  }

		// go to next step
		timer += tau;
		printf("Time = %lf\n",timer);
	}
	
	/* save results */
	for(i=0;i<NX;i++)
	 for(k=0;k<NY;k++)
	  fprintf(fout,"%lf %lf %lf\n",i*h+0.5*h-0.5*xm,k*h+0.5*h-0.5*ym,
							U[index(i,k,NZ/2,RHO_INDEX)]);
	
	fclose(fout);

	/* free memory */
	// _mm_free(U);
	delete U;
	delete Unew;
	delete FX;
	delete FY;
	delete FZ;
	delete Physics;

	return 0;
}
