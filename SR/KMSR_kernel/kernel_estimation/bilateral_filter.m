function r_img = bilateral_filter(img, sigma_s, sigma, boundary_method, s_size)

if ~exist('boundary_method', 'var')
    boundary_method = 'replicate';
end

if isinteger(img) == 1, img = single(img) / 255; end

[h w d] = size(img);

if d == 3
  C = makecform('srgb2lab');
  lab = single( applycform(double(img), C) );
  sigma = sigma * 100;
else
  lab = img;
  sigma = sigma * sqrt(d);
end

if exist('s_size', 'var')
    fr = s_size;
else
    fr = ceil(sigma_s*3);
end

p_img = padarray(img, [fr fr], boundary_method);
p_lab = padarray(lab, [fr fr], boundary_method);

u = fr+1; b = u+h-1;
l = fr+1; r = l+w-1;

r_img = zeros(h, w, d, 'single');
w_sum = zeros(h, w, 'single');

spatial_weight = fspecial('gaussian', 2*fr+1, sigma_s);
ss = sigma * sigma;

for y = -fr:fr
  for x = -fr:fr
    
    w_s = spatial_weight(y+fr+1, x+fr+1);
    
    n_img = p_img(u+y:b+y, l+x:r+x, :);
    n_lab = p_lab(u+y:b+y, l+x:r+x, :);
    
    f_diff = lab - n_lab;
    f_dist = sum(f_diff.^2, 3);
    
    w_f = exp(-0.5 * (f_dist / ss));
    
    w_t = w_s .* w_f;
    
    r_img = r_img + n_img .* repmat(w_t, [1 1 d]);
    w_sum = w_sum + w_t;
    
  end
end

r_img = r_img ./ repmat(w_sum, [1 1 d]);

end