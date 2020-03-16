function psf = adjust_psf_center(psf)

[X Y] = meshgrid(1:size(psf,2), 1:size(psf,1));
xc1 = sum2(psf .* X);
yc1 = sum2(psf .* Y);
xc2 = (size(psf,2)+1) / 2;
yc2 = (size(psf,1)+1) / 2;
xshift = round(xc2 - xc1);
yshift = round(yc2 - yc1);
psf = warpimage(psf, [1 0 -xshift; 0 1 -yshift]);

function val = sum2(arr)
val = sum(arr(:));
%%
% M should be an inverse transform!
function warped = warpimage(img, M)
if size(img,3) == 3
    warped(:,:,1) = warpProjective2(img(:,:,1), M);
    warped(:,:,2) = warpProjective2(img(:,:,2), M);
    warped(:,:,3) = warpProjective2(img(:,:,3), M);
    warped(isnan(warped))=0;
else
    warped = warpProjective2(img, M);
    warped(isnan(warped))=0;
end

%%
function result = warpProjective2(im,A)
%
% function result = warpProjective2(im,A)
%
% im: input image
% A: 2x3 affine transform matrix or a 3x3 matrix with [0 0 1]
% for the last row.
% if a transformed point is outside of the volume, NaN is used
%
% result: output image, same size as im
%

if (size(A,1)>2)
  A=A(1:2,:);
end

% Compute coordinates corresponding to input 
% and transformed coordinates for result
[x,y]=meshgrid(1:size(im,2),1:size(im,1));
coords=[x(:)'; y(:)'];
homogeneousCoords=[coords; ones(1,prod(size(im)))];
warpedCoords=A*homogeneousCoords;
xprime=warpedCoords(1,:);%./warpedCoords(3,:);
yprime=warpedCoords(2,:);%./warpedCoords(3,:);

result = interp2(x,y,im,xprime,yprime, 'linear');
result = reshape(result,size(im));

return;

