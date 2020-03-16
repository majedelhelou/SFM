#ifndef OW_MAT3_H
#define OW_MAT3_H

/*
    Header defnining basic 3 x 3 matrix operations
    
    Author:     Oliver Whyte <oliver.whyte@ens.fr>
    Date:       August 2010
    Copyright:  2010, Oliver Whyte
*/

#include <math.h>

/* Invert a 3 x 3 matrix */
inline void inv3(const double *A, double *invA)
{
    double detA = A[0]*(A[8]*A[4]-A[5]*A[7])-A[1]*(A[8]*A[3]-A[5]*A[6])+A[2]*(A[7]*A[3]-A[4]*A[6]);
    invA[0] =  (A[8]*A[4]-A[5]*A[7])/detA;
    invA[1] = -(A[8]*A[1]-A[2]*A[7])/detA;
    invA[2] =  (A[5]*A[1]-A[2]*A[4])/detA;
    invA[3] = -(A[8]*A[3]-A[5]*A[6])/detA;
    invA[4] =  (A[8]*A[0]-A[2]*A[6])/detA;
    invA[5] = -(A[5]*A[0]-A[2]*A[3])/detA;
    invA[6] =  (A[7]*A[3]-A[4]*A[6])/detA;
    invA[7] = -(A[7]*A[0]-A[1]*A[6])/detA;
    invA[8] =  (A[4]*A[0]-A[1]*A[3])/detA;
}

/* 3 x 3 rotation matrix */
inline void rot3(double tx, double ty, double tz, double *R)
{
  double T = sqrt(tx*tx + ty*ty + tz*tz);
  if (T>0) {
    tx/=T;
    ty/=T;
    tz/=T;
    double s = sin(T);
    double c = cos(T);
    double omc = 1-c;
    /*     R   = [ R[0]  R[3]  R[6]
     *             R[1]  R[4]  R[7]
     *             R[2]  R[5]  R[8]  ] */
    R[0] =     c + tx*tx*omc;
    R[1] =  tz*s + ty*tx*omc;
    R[2] = -ty*s + tz*tx*omc;
    R[3] = -tz*s + tx*ty*omc;
    R[4] =     c + ty*ty*omc;
    R[5] =  tx*s + tz*ty*omc;
    R[6] =  ty*s + tx*tz*omc;
    R[7] = -tx*s + ty*tz*omc;
    R[8] =     c + tz*tz*omc;
  } else {
    R[0] = 1;
    R[1] = 0;
    R[2] = 0;
    R[3] = 0;
    R[4] = 1;
    R[5] = 0;
    R[6] = 0;
    R[7] = 0;
    R[8] = 1;
  }
}

/* copy elements of 3 x 3 matrix from one array to another */
inline void cp3(const double *A, double *B)
{
  B[0] = A[0];
  B[1] = A[1];
  B[2] = A[2];
  B[3] = A[3];
  B[4] = A[4];
  B[5] = A[5];
  B[6] = A[6];
  B[7] = A[7];
  B[8] = A[8];
}

/* multiply two 3 x 3 matrices in-place */
inline void mmip3(const double *A, double *B)
{
  double b[9] = {B[0],B[1],B[2],B[3],B[4],B[5],B[6],B[7],B[8]};
  B[0] = A[0]*b[0] + A[3]*b[1] + A[6]*b[2];
  B[1] = A[1]*b[0] + A[4]*b[1] + A[7]*b[2];
  B[2] = A[2]*b[0] + A[5]*b[1] + A[8]*b[2];
  B[3] = A[0]*b[3] + A[3]*b[4] + A[6]*b[5];
  B[4] = A[1]*b[3] + A[4]*b[4] + A[7]*b[5];
  B[5] = A[2]*b[3] + A[5]*b[4] + A[8]*b[5];
  B[6] = A[0]*b[6] + A[3]*b[7] + A[6]*b[8];
  B[7] = A[1]*b[6] + A[4]*b[7] + A[7]*b[8];
  B[8] = A[2]*b[6] + A[5]*b[7] + A[8]*b[8];
}

/* 3 x 3 identity matrix */
inline void id3(double *B)
{
    B[0] = 1;         B[3] = 0;         B[6] = 0;
    B[1] = 0;         B[4] = 1;         B[7] = 0;
    B[2] = 0;         B[5] = 0;         B[8] = 1;
}


#endif
