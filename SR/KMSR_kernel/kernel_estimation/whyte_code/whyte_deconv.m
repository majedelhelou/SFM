function out = whyte_deconv(ImBlurry, kernel)
%% Whyte's code
%% Reference:
%% Reference:  O. Whyte, J. Sivic and A. Zisserman. "Deblurring Shaken and Partially 
%% Saturated Images". In Proc. CPCV Workshop at ICCV, 2011.
%% Downloaded from http://www.di.ens.fr/willow/research/saturation/
ImBlurry = ImBlurry.^(2.2);
dd.kernel = kernel;
dd.non_uniform = 0;
 RLargs = {};
CombinedRLargs  = cat(2,RLargs,'forward_saturation','prevent_ringing');

% Deblur
out  = deconvRL(ImBlurry, dd.kernel, dd.non_uniform, CombinedRLargs{:});
out = out.^(1/2.2);
