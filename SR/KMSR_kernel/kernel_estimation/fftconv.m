function cI=fftconv(I, filt, b_otf)

if size(I,3) == 3
    [H W ch] = size(I);
    otf = psf2otf(filt,[H W]);
    cI(:,:,1) = fftconv(I(:,:,1), otf, true);
    cI(:,:,2) = fftconv(I(:,:,2), otf, true);
    cI(:,:,3) = fftconv(I(:,:,3), otf, true);
    return;
end

if exist('b_otf', 'var') && b_otf == true
    cI = real(ifft2(fft2(I).*filt));
else
    cI = real(ifft2(fft2(I).*psf2otf(filt,size(I))));
end
