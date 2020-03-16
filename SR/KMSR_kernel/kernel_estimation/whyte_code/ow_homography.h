#ifndef OW_HOMOGRAPHY_H
#define OW_HOMOGRAPHY_H

#include "ow_mat3.h"

/* 
    Header defining basic 2D homography functions
    
    Author:     Oliver Whyte <oliver.whyte@ens.fr>
    Date:       August 2010
    Copyright:  2010, Oliver Whyte
    
    This file defines:

    void compute_homography_matrix(const double *Ksharp, const double *theta, const double *invKblurry, double *H)

    void compute_prerotated_homography_matrix(const double *Ksharp, const double *theta, const double *theta_pre, const double *invKblurry, double *H)

    void project_blurry_to_sharp(const int yi, const int xi, const double *H, double *xjptr, double *yjptr)

    void interp_coeffs(const double xj, const double yj, int *xjfloorptr, int *yjfloorptr, double *coeffs)
*/

/* Function to compute homography from camera rotation */
inline void compute_homography_matrix(const double *Ksharp, const double *theta, const double *invKblurry, double *H) {
    double R[9];
    /* Compute homography */
    cp3(invKblurry,H);
    rot3(theta[0],theta[1],theta[2],R);
    mmip3(R,H);
    mmip3(Ksharp,H);
}

/* Function to compute homography from two camera rotations composed together */
inline void compute_prerotated_homography_matrix(const double *Ksharp, const double *theta, const double *theta_pre, const double *invKblurry, double *H) {
    double R[9];
    /* Compute homography */
    cp3(invKblurry,H);
    rot3(theta[0],theta[1],theta[2],R);
    mmip3(R,H);
    rot3(theta_pre[0],theta_pre[1],theta_pre[2],R);
    mmip3(R,H);
    mmip3(Ksharp,H);
}


/* Function to perform projection under homography */
inline void project_blurry_to_sharp(const int yi, const int xi, const double *H, double *xjptr, double *yjptr) {
    /* Project blurry image pixel (xi,yi) into sharp image.
        Position in sharp image is (xj,yj) */
    double denom = (H[2]*xi + H[5]*yi + H[8]) + 1e-16;
    *yjptr       = (H[1]*xi + H[4]*yi + H[7]) / denom;
    *xjptr       = (H[0]*xi + H[3]*yi + H[6]) / denom;
    /* If very close to a whole pixel, round to that pixel */
/*  if(fabs(*xjptr-round(*xjptr)) < 1e-5)
        *xjptr = round(*xjptr);
    if(fabs(*yjptr-round(*yjptr)) < 1e-5)
        *yjptr = round(*yjptr); */
    /* If at the edge, move just inside the image */
/*  if(*yjptr == h_sharp)
        *yjptr = h_sharp-1e-5;
    if(*xjptr == w_sharp)
        *xjptr = w_sharp-1e-5; */
}

/* Bilinear interpolation is used, but could be substituted for another method, eg bicubic */
inline void interp_coeffs(const double xj, const double yj, int *xjfloorptr, int *yjfloorptr, double *coeffs) {
    /* Integer parts of position */
    double yjfloor = floor(yj);
    double xjfloor = floor(xj);
    double x_m_xfloor = xj - xjfloor;
    double y_m_yfloor = yj - yjfloor;
    /* Bilinear interpolation coefficients */
    coeffs[0] = (1-(x_m_xfloor))*(1-(y_m_yfloor)); /* I(xjfloor  ,yjfloor  ) */
    coeffs[1] = (1-(x_m_xfloor))*   (y_m_yfloor) ; /* I(xjfloor  ,yjfloor+1) */
    coeffs[2] =    (x_m_xfloor) *(1-(y_m_yfloor)); /* I(xjfloor+1,yjfloor  ) */
    coeffs[3] =    (x_m_xfloor) *   (y_m_yfloor) ; /* I(xjfloor+1,yjfloor+1) */
    *xjfloorptr = (int)xjfloor;
    *yjfloorptr = (int)yjfloor;
}
int n_interp = 4;
int xoff[4] = {0,0,1,1}; /* Offsets from (xjfloor,yjfloor) to which the interpolation coefficients correspond */
int yoff[4] = {0,1,0,1};


#endif

