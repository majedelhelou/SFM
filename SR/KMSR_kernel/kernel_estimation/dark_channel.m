function [J, J_index] = dark_channel(I, patch_size)
% function J = dark_channel(I, patch_size);

% Computes the "Dark Channel" of corresponding RGB image.
% -Finds from the input image the minimum value among all 
%  pixels within the patch centered around the location of the 
%  target pixel in the newly created dark channel image 'J'
%  J is a 2-D image (grayscale).

% Example: J = dark_channel(I, 15); % computes using 15x15 patch

% Check to see that the input is a color image
% if ndims(I) == 3
%     [M N C] = size(I);
%     J = zeros(M, N); % Create empty matrix for J
%     J_index = zeros(M, N); % Create empty index matrix
% else
%     error('Sorry, dark_channel supports only RGB images');
% end
%% for grayscale image
%
[M, N, C] = size(I);
J = zeros(M, N); % Create empty matrix for J
J_index = zeros(M, N); % Create empty index matrix

% Test if patch size has odd number
if ~mod(numel(patch_size),2) % if even number
    error('Invalid Patch Size: Only odd number sized patch supported.');
end

% pad original image
%I = padarray(I, [floor(patch_size./2) floor(patch_size./2)], 'symmetric');
I = padarray(I, [floor(patch_size./2) floor(patch_size./2)], 'replicate');

% Compute the dark channel 
for m = 1:M
        for n = 1:N
            patch = I(m:(m+patch_size-1), n:(n+patch_size-1),:);
            tmp = min(patch, [], 3);
            [tmp_val, tmp_idx] = min(tmp(:));
            J(m,n) = tmp_val;
            J_index(m,n) = tmp_idx;
        end
end

end



