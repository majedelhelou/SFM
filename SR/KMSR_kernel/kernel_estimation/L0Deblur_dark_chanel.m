function S = L0Deblur_dark_chanel(Im, kernel, lambda, wei_grad, kappa)
%%
% Image restoration with L0 regularized intensity and gradient prior
% The objective function:
% S = argmin ||I*k - B||^2 + \lambda |D(I)|_0 + wei_grad |\nabla I|_0
%% Input:
% @Im: Blurred image
% @kernel: blur kernel
% @lambda: weight for the L0 intensity prior
% @wei_grad: weight for the L0 gradient prior
% @kappa: Update ratio in the ADM
%% Output:
% @S: Latent image
%
% The Code is created based on the method described in the following paper 
%   [1] Jinshan Pan, Deqing Sun, Hanspteter Pfister, and Ming-Hsuan Yang,
%        Blind Image Deblurring Using Dark Channel Prior, CVPR, 2016. 

if ~exist('kappa','var')
    kappa = 2.0;
end
%% pad image
% H = size(Im,1);    W = size(Im,2);
% Im = wrap_boundary_liu(Im, opt_fft_size([H W]+size(kernel)-1));
%%
S = Im;
betamax = 1e5;
fx = [1, -1];
fy = [1; -1];
[N,M,D] = size(Im);
sizeI2D = [N,M];
otfFx = psf2otf(fx,sizeI2D);
otfFy = psf2otf(fy,sizeI2D);
%%
KER = psf2otf(kernel,sizeI2D);
Den_KER = abs(KER).^2;
%%
Denormin2 = abs(otfFx).^2 + abs(otfFy ).^2;
if D>1
    Denormin2 = repmat(Denormin2,[1,1,D]);
    KER = repmat(KER,[1,1,D]);
    Den_KER = repmat(Den_KER,[1,1,D]);
end
Normin1 = conj(KER).*fft2(S);
%% pixel sub-problem
%%
dark_r = 35; %% Fixed size!
%mybeta_pixel = 2*lambda;
%[J, J_idx] = dark_channel(S, dark_r);
mybeta_pixel = lambda/(graythresh((S).^2));
maxbeta_pixel = 2^3;
while mybeta_pixel< maxbeta_pixel
    %% 
    [J, J_idx] = dark_channel(S, dark_r);
    u = J;
    if D==1
        t = u.^2<lambda/mybeta_pixel;
    else
        t = sum(u.^2,3)<lambda/mybeta_pixel;
        t = repmat(t,[1,1,D]);
    end
    u(t) = 0;
    %
    clear t;
    u = assign_dark_channel_to_pixel(S, u, J_idx, dark_r);
    %% Gradient sub-problem
    beta = 2*wei_grad;
    %beta = 0.01;
    while beta < betamax
        Denormin   = Den_KER + beta*Denormin2 + mybeta_pixel;
        %
        h = [diff(S,1,2), S(:,1,:) - S(:,end,:)];
        v = [diff(S,1,1); S(1,:,:) - S(end,:,:)];
        if D==1
            t = (h.^2+v.^2)<wei_grad/beta;
        else
            t = sum((h.^2+v.^2),3)<wei_grad/beta;
            t = repmat(t,[1,1,D]);
        end
        h(t)=0; v(t)=0;
        clear t;
        %
        Normin2 = [h(:,end,:) - h(:, 1,:), -diff(h,1,2)];
        Normin2 = Normin2 + [v(end,:,:) - v(1, :,:); -diff(v,1,1)];
        %
        FS = (Normin1 + beta*fft2(Normin2) + mybeta_pixel*fft2(u))./Denormin;
        S = real(ifft2(FS));
        %%
        beta = beta*kappa;
        if wei_grad==0
            break;
        end
    end
    mybeta_pixel = mybeta_pixel*kappa;
end
%
end
