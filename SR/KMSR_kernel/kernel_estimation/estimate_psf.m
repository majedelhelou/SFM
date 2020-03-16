function psf = estimate_psf(blurred_x, blurred_y, latent_x, latent_y, weight, psf_size)

    %----------------------------------------------------------------------
    % these values can be pre-computed at the beginning of each level
%     blurred_f = fft2(blurred);
%     dx_f = psf2otf([1 -1 0], size(blurred));
%     dy_f = psf2otf([1;-1;0], size(blurred));
%     blurred_xf = dx_f .* blurred_f; %% FFT (Bx)
%     blurred_yf = dy_f .* blurred_f; %% FFT (By)
    
    latent_xf = fft2(latent_x);
    latent_yf = fft2(latent_y);
    blurred_xf = fft2(blurred_x);
    blurred_yf = fft2(blurred_y);
    % compute b = sum_i w_i latent_i * blurred_i
    b_f = conj(latent_xf)  .* blurred_xf ...
        + conj(latent_yf)  .* blurred_yf;
    b = real(otf2psf(b_f, psf_size));

    p.m = conj(latent_xf)  .* latent_xf ...
        + conj(latent_yf)  .* latent_yf;
    %p.img_size = size(blurred);
    p.img_size = size(blurred_xf);
    p.psf_size = psf_size;
    p.lambda = weight;

    psf = ones(psf_size) / prod(psf_size);
    psf = conjgrad(psf, b, 20, 1e-5, @compute_Ax, p);
    
    psf(psf < max(psf(:))*0.05) = 0;
    psf = psf / sum(psf(:));    
end

function y = compute_Ax(x, p)
    x_f = psf2otf(x, p.img_size);
    y = otf2psf(p.m .* x_f, p.psf_size);
    y = y + p.lambda * x;
end
