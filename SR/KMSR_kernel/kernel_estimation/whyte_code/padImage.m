% imPadded = padImage(im, padsize, padval)
%       padsize = [top, bottom, left, right]
%       padval = valid arguments for padval to padarray. e.g. 'replicate', or 0
% 
%       for negative padsize, undoes the padding

%	Author:		Oliver Whyte <oliver.whyte@ens.fr>
%	Date:		November 2011
%	Copyright:	2011, Oliver Whyte
%	Reference:  O. Whyte, J. Sivic and A. Zisserman. "Deblurring Shaken and Partially Saturated Images". In Proc. CPCV Workshop at ICCV, 2011.
%	URL:		http://www.di.ens.fr/willow/research/saturation/

function imPadded = padImage(im, padsize, padval)
if nargin < 3, padval = 0; end

if any(padsize < 0)
    padsize = -padsize;
    imPadded = im(padsize(1)+1:end-padsize(2), padsize(3)+1:end-padsize(4), :);
else
    imPadded =  padarray( ...
                padarray(im, [padsize(1) padsize(3)],padval,'pre'), ...
                             [padsize(2) padsize(4)],padval,'post');
end