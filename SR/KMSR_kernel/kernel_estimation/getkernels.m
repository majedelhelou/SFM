clc;
clear;
close all;
addpath(genpath('image'));
addpath(genpath('whyte_code'));
addpath(genpath('cho_code'));
opts.prescale = 1; %%downsampling
opts.xk_iter = 5; %% the iterations
opts.gamma_correct = 1.0;
opts.k_thresh = 20;

% modify below line for the input dir for blurry image
inputdir = 'x2/';
% modify below line for the output dir for the kernels
outputdir = 'x2results/';

dirs = dir(inputdir);
L = length(dirs);
for l = 3:L
    picfiles = dir(strcat(inputdir, strcat(dirs(l).name, '/*.jpg')));
    LL = length(picfiles);
    for kk = 1:LL
      try
        opts.kernel_size = 25;  saturation = 0;
        lambda_dark = 4e-3; lambda_grad = 4e-3;opts.gamma_correct = 1.0;
        lambda_tv = 0.003; lambda_l0 = 5e-4; weight_ring = 1;
        img = imread([inputdir dirs(l).name '/' picfiles(kk).name]); 
        y = img;
        isselect = 0; %false or true
        if isselect ==1
            figure, imshow(y);
            %tips = msgbox('Please choose the area for deblurring:');
            fprintf('Please choose the area for deblurring:\n');
            h = imrect;
            position = wait(h);
            close;
            B_patch = imcrop(y,position);
            y = (B_patch);
        else
            y = y;
        end
        if size(y,3)==3
            yg = im2double(rgb2gray(y));
        else
            yg = im2double(y);
        end
        [kernel, interim_latent] = blind_deconv(yg, lambda_dark, lambda_grad, opts);
        
        % for plotting
        k = kernel - min(kernel(:));
        k = k./max(k(:));
        
        % save kernel
      save(sprintf('%s%s%s%s%s',outputdir,dirs(l).name,'/',string(extractBetween(picfiles(kk).name,1,length(picfiles(kk).name)-4)),'_kernel.mat'), 'kernel');
        % uncomment the following line to generate png files for kernels
        % imwrite(k,sprintf('%s%s%s%s%s',outputdir,dirs(l).name,'/',string(extractBetween(picfiles(kk).name,1,length(picfiles(kk).name)-4)),'_kernel.png'));
      catch
        1
      end
        
    end
end

