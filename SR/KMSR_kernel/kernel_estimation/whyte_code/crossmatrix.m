
%	Author:		Oliver Whyte <oliver.whyte@ens.fr>
%	Date:		November 2011
%	Copyright:	2011, Oliver Whyte
%	Reference:  O. Whyte, J. Sivic and A. Zisserman. "Deblurring Shaken and Partially Saturated Images". In Proc. CPCV Workshop at ICCV, 2011.
%	URL:		http://www.di.ens.fr/willow/research/saturation/

function vx = crossmatrix(v)
vx = [    0, -v(3),  v(2);...
       v(3),     0, -v(1);...
      -v(2),  v(1),     0];