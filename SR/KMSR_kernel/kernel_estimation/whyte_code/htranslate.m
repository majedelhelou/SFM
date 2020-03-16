function T = htranslate(tt)
% 	HTRANSLATE   Translation matrix for homogeneous coordinates
% 		[T] = HTRANSLATE(TT)
% 
% 	For an input vector tt with n elements, the ouput is the n+1 x n+1 matrix which applies that translation in homogeneous coordinates.

%	Author:		Oliver Whyte <oliver.whyte@ens.fr>
%	Date:		November 2011
%	Copyright:	2011, Oliver Whyte
%	Reference:  O. Whyte, J. Sivic and A. Zisserman. "Deblurring Shaken and Partially Saturated Images". In Proc. CPCV Workshop at ICCV, 2011.
%	URL:		http://www.di.ens.fr/willow/research/saturation/

T = eye(length(tt)+1);
T(1:end-1,end) = tt(:);

end %  function