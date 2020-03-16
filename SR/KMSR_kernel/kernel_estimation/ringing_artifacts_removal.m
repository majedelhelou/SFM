function [result] = ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring)
%%
% Removing artifacts in non-blind deconvolution step
%% Input:
% @y: input blurred image 
% @kernel: blur kernel
% @lambda_tv: weight for the Laplacian prior based deconvolution [1e-3,
% 1e-2];
% @lambda_l0: weight for the L0 prior based deconvolution
 %                      typically set as 2e-3, the best range is [1e-4, 2e-3]
% @weight_ring: Larger values help suppress the ringing artifacts. 
%                          weight_ring=0 imposes no suppression
%
% Ouput:
% @result: latent image 
%
% The Code is created based on the method described in the following paper 
%        Jinshan Pan, Zhe Hu, Zhixun Su, and Ming-Hsuan Yang,
%        Deblurring Text Images via L0-Regularized Intensity and Gradient
%        Prior, CVPR, 2014. 

%   Author: Jinshan Pan (sdluran@gmail.com)
%   Date  : 05/18/2014

H = size(y,1);    W = size(y,2);
y_pad = wrap_boundary_liu(y, opt_fft_size([H W]+size(kernel)-1));
Latent_tv = [];
for c = 1:size(y,3)
    Latent_tv(:,:,c) = deblurring_adm_aniso(y_pad(:,:,c), kernel, lambda_tv, 1);
end
Latent_tv = Latent_tv(1:H, 1:W, :);
if weight_ring==0
    result = Latent_tv;
    return;
end
Latent_l0 = L0Restoration(y_pad, kernel, lambda_l0, 2);
Latent_l0 = Latent_l0(1:H, 1:W, :);
%%
diff = Latent_tv - Latent_l0;
bf_diff = bilateral_filter(diff, 3, 0.1);
result = Latent_tv - weight_ring*bf_diff;