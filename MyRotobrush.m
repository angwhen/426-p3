folder_name = 'Frames1';
num_images = size(dir(['../' folder_name '/*.jpg']),1);
images_cell = cell(1,num_images);
for i=1:num_images
    filename = sprintf('../%s/%d.jpg',folder_name,i);
    images_cell{1,i}  = imread(filename);
end

%imshow(images_cell{1,1});
%init_mask = roipoly();
init_mask = load('frames1_mask1'); init_mask = init_mask.init_mask;

halfw = 15; density = 6;
[local_windows_image_cell local_windows_mask_cell] = create_local_windows(images_cell{1,1},init_mask,halfw,density);

[i s] = size(local_windows_image_cell);
foreground_model_cell = cell(1,s);
background_model_cell = cell(1,s);
% foreground_prob_cell = cell(1,s);
% background_prob_cell = cell(1,s);
combined_prob_cell = cell(1,s);
for i = 1:s
    [r c] = find(bwdist(local_windows_mask_cell{1,i})>4);
    foreground_pix = impixel(local_windows_image_cell{1,i},r,c);
    foreground_model_cell{1,i} = fitgmdist(foreground_pix,1);
    
    inverted = local_windows_mask_cell{1,i}==0;
    [r c] = find(bwdist(inverted)>4);
    background_pix = impixel(local_windows_image_cell{1,i},r,c);
    background_model_cell{1,i} = fitgmdist(background_pix,1);
    
    % probs
    [r c ~] = size(local_windows_image_cell{1,i});
    values = reshape(double(local_windows_image_cell{1,i}),[r*c 3]);
    
    fore_prob = pdf(foreground_model_cell{1,i},values);
%     norm_fore_prob = fore_prob - min(fore_prob(:));
%     norm_fore_prob = norm_fore_prob ./ max(norm_fore_prob(:)); 
%     foreground_prob_cell{1,i} = reshape( norm_fore_prob,[r c]);
        
    back_prob = pdf(background_model_cell{1,i},values);
%     norm_back_prob = back_prob - min(back_prob(:));
%     norm_back_prob = norm_back_prob ./ max(norm_back_prob(:));
%     background_prob_cell{1,i} = reshape( norm_back_prob,[r c]);
    
    comb_prob = fore_prob./(fore_prob+back_prob);
    norm_comb_prob = comb_prob - min(comb_prob); 
    norm_comb_prob = norm_comb_prob ./ max(norm_comb_prob(:));
    combined_prob_cell{1,i} = reshape(norm_comb_prob,[r c]); 
    %imshow(combined_prob_cell{1,i});
end


%color model confidence
color_model_confidence_cell = cell(1,s);
for i = 1:s
    d1 = bwdist(local_windows_mask_cell{1,i});
    d2 = bwdist(local_windows_mask_cell{1,i}==0);
    d = max(d1,d2);
    wc = exp(-d.^2./halfw.^2);
    numer_window = abs(local_windows_mask_cell{1,i} - combined_prob_cell{1,i}).*wc; 
    numer = sum(numer_window(:));
    denom = sum(wc(:));
    color_model_confidence_cell{1,i} = 1 - numer/denom;
end

%shape model
shape_model_confidence_mask_cell = cell(1,s);
fcutoff = 0.8; sigma_min = 2; sigma_max = halfw*2+1; r=2; a = (sigma_max-sigma_min)/(1-fcutoff)^r;
for i = 1:s
    fc = color_model_confidence_cell{1,i};
    if fc > fcutoff
        sigma_s = sigma_min+a*(fc-fcutoff)^r;
    else
        sigma_s = sigma_min;
    end
    
    d1 = bwdist(local_windows_mask_cell{1,i}); d2 = bwdist(local_windows_mask_cell{1,i}==0);
    d = max(d1,d2);
    shape_model_confidence_mask_cell{1,i} = 1 - exp(-d.^2./sigma_s.^2); %eq 3
    imshow(shape_model_confidence_mask_cell{1,i});
end

%estimate whole object motion
for i = 2:num_images
    points1 = detectSURFFeatures(images_cell{1,i-1});
    points2 = detectSURFFeatures(images_cell{1,i});
    [features1, validpts1]  = extractFeatures(images_cell{1,i-1},points1);
    [features2, validpts2] = extractFeatures(images_cell{1,i},points2);
    match_features(
end
    


function [local_windows_image_cell, local_windows_mask_cell] = create_local_windows(I, init_mask,halfwidth,density)
    b = bwboundaries(init_mask); b = b{1,1}; %coords of edge of mask
    [h w] = size(b);
    local_windows_image_cell = cell(1,floor(h/density));
    local_windows_mask_cell = cell(1,floor(h/density));
    for i = 1:floor(h/density)
        r  = b(i,1); c= b(i,2);
        my_patch = I(r-halfwidth:r+halfwidth, c-halfwidth:c+halfwidth,:);
        local_windows_image_cell{1,i} = my_patch; 
        my_patch = init_mask(r-halfwidth:r+halfwidth, c-halfwidth:c+halfwidth,:);
        local_windows_mask_cell{1,i} = my_patch;
    end
end

