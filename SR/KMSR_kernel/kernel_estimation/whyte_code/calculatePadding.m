% pad = calculatePadding(image_size,non_uniform = 0,kernel)
% pad = calculatePadding(image_size,non_uniform = 1,theta_list,Kinternal)
%       where pad = [top, bottom, left, right]

%	Author:		Oliver Whyte <oliver.whyte@ens.fr>
%	Date:		November 2011
%	Copyright:	2011, Oliver Whyte
%	Reference:  O. Whyte, J. Sivic and A. Zisserman. "Deblurring Shaken and Partially Saturated Images". In Proc. CPCV Workshop at ICCV, 2011.
%	URL:		http://www.di.ens.fr/willow/research/saturation/

function [pad_replicate,Kinternal] = calculatePadding(image_size,non_uniform,theta_list,Kinternal)

h_sharp = image_size(1);
w_sharp = image_size(2);

if non_uniform
    % Calculate padding
    im_corners = [1,       1, w_sharp, w_sharp;...
                  1, h_sharp, h_sharp,       1;...
                  1,       1,       1,       1];
    pad_replicate_t = 0; % top of image
    pad_replicate_b = 0; % bottom of image
    pad_replicate_l = 0; % left of image
    pad_replicate_r = 0; % right of image
    % for each non-zero in the kernel...
    for i=1:size(theta_list,2)
        % back proect corners of blurry image to see how far out we need to pad
        % H = Ksharp*expm(crossmatrix(-theta_list(:,i)))*inv(Kblurry);
        H = Kinternal*expm(crossmatrix(-theta_list(:,i)))*inv(Kinternal);
        projected_corners_sharp = hnormalise(H*im_corners);
        offsets = abs(projected_corners_sharp - im_corners);
        if offsets(1,1) > pad_replicate_l, pad_replicate_l = ceil(offsets(1,1)); end
        if offsets(1,2) > pad_replicate_l, pad_replicate_l = ceil(offsets(1,2)); end
        if offsets(1,3) > pad_replicate_r, pad_replicate_r = ceil(offsets(1,3)); end
        if offsets(1,4) > pad_replicate_r, pad_replicate_r = ceil(offsets(1,4)); end
        if offsets(2,1) > pad_replicate_t, pad_replicate_t = ceil(offsets(2,1)); end
        if offsets(2,2) > pad_replicate_b, pad_replicate_b = ceil(offsets(2,2)); end
        if offsets(2,3) > pad_replicate_t, pad_replicate_t = ceil(offsets(2,3)); end
        if offsets(2,4) > pad_replicate_b, pad_replicate_b = ceil(offsets(2,4)); end
    end
    % Adjust calibration matrices to take account padding
    Kinternal = htranslate([pad_replicate_l ; pad_replicate_t]) * Kinternal;
else
    kernel = theta_list;
    pad_replicate_t =  ceil((size(kernel,1)-1)/2);
    pad_replicate_b = floor((size(kernel,1)-1)/2);
    pad_replicate_l =  ceil((size(kernel,2)-1)/2);
    pad_replicate_r = floor((size(kernel,2)-1)/2);
    Kinternal = [];
end

w_sharp  = w_sharp  + pad_replicate_l + pad_replicate_r;
h_sharp  = h_sharp  + pad_replicate_t + pad_replicate_b;

% top, bottom, left, right
pad_replicate = [pad_replicate_t, pad_replicate_b, pad_replicate_l, pad_replicate_r];

