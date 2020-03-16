clc
clear
close all

n_repeat = 50;
data_name_ID = 0;

PSNR_avg1 = zeros(n_repeat, 3);
PSNR_avg2 = zeros(n_repeat, 3);
PSNR_avg4 = zeros(n_repeat, 3);
PSNR_avg8 = zeros(n_repeat, 3);
PSNR_avg16 = zeros(n_repeat, 3);

SSIM_avg1 = zeros(n_repeat, 3);
SSIM_avg2 = zeros(n_repeat, 3);
SSIM_avg4 = zeros(n_repeat, 3);
SSIM_avg8 = zeros(n_repeat, 3);
SSIM_avg16 = zeros(n_repeat, 3);
    
for DN = {'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_B', 'TwoPhoton_MICE'}
    data_name = DN{1};
    data_name_ID = data_name_ID + 1;


    addpath('./code')
    addpath('../utils')
    dir_avg1 = ['../../dataset/', data_name, '/raw/19/']; % 50 images
    dir_avg2 = ['../../dataset/', data_name, '/avg2/19/']; % 50 images
    dir_avg4 = ['../../dataset/', data_name, '/avg4/19/']; % 50 images
    dir_avg8 = ['../../dataset/', data_name, '/avg8/19/']; % 50 images
    dir_avg16 = ['../../dataset/', data_name, '/avg16/19/']; % 50 images
    dir_gt = ['../../dataset/', data_name, '/gt/19/']; % 1 image


    [avg1_array] = import_img_array(dir_avg1);
    [avg2_array] = import_img_array(dir_avg2);
    [avg4_array] = import_img_array(dir_avg4);
    [avg8_array] = import_img_array(dir_avg8);
    [avg16_array] = import_img_array(dir_avg16);
    [img_gt] = import_img_array(dir_gt);


    %%%%%%%%%%%%%%%%%%%%%%%%%%% noise realizations %%%%%%%%%%%%%%%%%%%%%%%%%%%
    TIME_avg = zeros(n_repeat, 3);

    for i_repeat = 1:n_repeat

        fprintf('\n ----- The %d / %d realization -----\n', i_repeat, n_repeat)

        % extract the images and average them
        img_avg1 = avg1_array(:,:,i_repeat);
        img_avg2 = avg2_array(:,:,i_repeat);
        img_avg4 = avg4_array(:,:,i_repeat);
        img_avg8 = avg8_array(:,:,i_repeat);
        img_avg16 = avg16_array(:,:,i_repeat);   


        % perform the denoising algorithm
        [denoise_avg1, time_avg1] = denoise_VST_EPLL(img_avg1);
        [denoise_avg2, time_avg2] = denoise_VST_EPLL(img_avg2);
        [denoise_avg4, time_avg4] = denoise_VST_EPLL(img_avg4);
        [denoise_avg8, time_avg8] = denoise_VST_EPLL(img_avg8);
        [denoise_avg16, time_avg16] = denoise_VST_EPLL(img_avg16);

        TIME_avg(i_repeat, data_name_ID) = mean([time_avg1, time_avg2, ...
            time_avg4, time_avg8, time_avg16]);

        % compute the PSNR
        PSNR_avg1(i_repeat, data_name_ID) = 10*log10(1/mean((img_gt(:)-denoise_avg1(:)).^2));
        PSNR_avg2(i_repeat, data_name_ID) = 10*log10(1/mean((img_gt(:)-denoise_avg2(:)).^2));
        PSNR_avg4(i_repeat, data_name_ID) = 10*log10(1/mean((img_gt(:)-denoise_avg4(:)).^2));
        PSNR_avg8(i_repeat, data_name_ID) = 10*log10(1/mean((img_gt(:)-denoise_avg8(:)).^2));
        PSNR_avg16(i_repeat, data_name_ID) = 10*log10(1/mean((img_gt(:)-denoise_avg16(:)).^2));
    
    end

end

fprintf('Average = 1 (PSNR: %.2f dB, SSIM: %.4f)\n', mean(PSNR_avg1(:)), mean(SSIM_avg1(:)))
fprintf('Average = 2 (PSNR: %.2f dB, SSIM: %.4f)\n', mean(PSNR_avg2(:)), mean(SSIM_avg2(:)))
fprintf('Average = 4 (PSNR: %.2f dB, SSIM: %.4f)\n', mean(PSNR_avg4(:)), mean(SSIM_avg4(:)))
fprintf('Average = 8 (PSNR: %.2f dB, SSIM: %.4f)\n', mean(PSNR_avg8(:)), mean(SSIM_avg8(:)))
fprintf('Average = 16 (PSNR: %.2f dB, SSIM: %.4f)\n', mean(PSNR_avg16(:)), mean(SSIM_avg16(:)))
fprintf('Average time for each denoising operation = %.2f\n', mean(TIME_avg(:)))
