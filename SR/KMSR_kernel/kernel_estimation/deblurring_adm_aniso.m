function [I] = deblurring_adm_aniso(B, k, lambda, alpha)
% Solving TV-\ell^2 deblurring problem via ADM/Split Bregman method
% 
% This reference of this code is :Fast Image Deconvolution using Hyper-Laplacian Priors
% Original code is created by Dilip Krishnan
% Finally modified by Jinshan Pan 2011/12/25
% Note: 
% In this model, aniso TV regularization method is adopted. 
% Thus, we do not use the Lookup table method proposed by Dilip Krishnan and Rob Fergus
% Reference: Kernel Estimation from Salient Structure for Robust Motion
% Deblurring
%Last update: (2012/6/20)
beta = 1/lambda;
beta_rate = 2*sqrt(2);
%beta_max = 5*2^10;
beta_min = 0.001;

[m n] = size(B); 
% initialize with input or passed in initialization
I = B; 

% make sure k is a odd-sized
if ((mod(size(k, 1), 2) ~= 1) | (mod(size(k, 2), 2) ~= 1))
  fprintf('Error - blur kernel k must be odd-sized.\n');
  return;
end;

[Nomin1, Denom1, Denom2] = computeDenominator(B, k);
Ix = [diff(I, 1, 2), I(:,1) - I(:,n)]; 
Iy = [diff(I, 1, 1); I(1,:) - I(m,:)]; 

%% Main loop
while beta > beta_min
    gamma = 1/(2*beta);
    Denom = Denom1 + gamma*Denom2;
    % subproblem for regularization term
    if alpha==1
        Wx = max(abs(Ix) - beta*lambda, 0).*sign(Ix);
        Wy = max(abs(Iy) - beta*lambda, 0).*sign(Iy);
        %%
    else
        Wx = solve_image(Ix, 1/(beta*lambda), alpha);
        Wy = solve_image(Iy, 1/(beta*lambda), alpha);
    end
      Wxx = [Wx(:,n) - Wx(:, 1), -diff(Wx,1,2)]; 
      Wxx = Wxx + [Wy(m,:) - Wy(1, :); -diff(Wy,1,1)]; 
        
      Fyout = (Nomin1 + gamma*fft2(Wxx))./Denom; 
      I = real(ifft2(Fyout));
      % update the gradient terms with new solution
      Ix = [diff(I, 1, 2), I(:,1) - I(:,n)]; 
      Iy = [diff(I, 1, 1); I(1,:) - I(m,:)]; 
    beta = beta/2;
end 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Nomin1, Denom1, Denom2] = computeDenominator(y, k)
%
% computes denominator and part of the numerator for Equation (3) of the
% paper
%
% Inputs: 
%  y: blurry and noisy input
%  k: convolution kernel  
% 
% Outputs:
%      Nomin1  -- F(K)'*F(y)
%      Denom1  -- |F(K)|.^2
%      Denom2  -- |F(D^1)|.^2 + |F(D^2)|.^2
%

sizey = size(y);
otfk  = psf2otf(k, sizey); 
Nomin1 = conj(otfk).*fft2(y);
Denom1 = abs(otfk).^2; 
% if higher-order filters are used, they must be added here too
Denom2 = abs(psf2otf([1,-1],sizey)).^2 + abs(psf2otf([1;-1],sizey)).^2;
