function X = hnormalise(X)
% 	hnormalise   Normalise homogeneous coordinates to have 1 as the last component
% 		Xn = hnormalise(X)
%
%		Inputs:
%				X		ndims x npts array, where each column is a vector of length ndims,
%							the homogeneous coordinates of a point in ndims-1 space
%
%		Outputs:
%				Xn		ndims x npts array, where each column represents the same point as in X
%							but normalised to have the last component equal to 1

%	Author:		Oliver Whyte <oliver.whyte@ens.fr>
%	Date:		November 2011
%	Copyright:	2011, Oliver Whyte
%	Reference:  O. Whyte, J. Sivic and A. Zisserman. "Deblurring Shaken and Partially Saturated Images". In Proc. CPCV Workshop at ICCV, 2011.
%	URL:		http://www.di.ens.fr/willow/research/saturation/

for i=1:size(X,1)-1
	X(i,:) = X(i,:)./X(end,:);
end
X(end,:) = 1;