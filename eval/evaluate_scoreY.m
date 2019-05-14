lst = {'../results/set5','../results/set14','../results/b100','../results/urban100'};
sota_psnr = {32.73,28.98,27.85,27.10};
sota_ssim = {0.9013,0.7910,0.7455,0.8142};
for j = 1:4
	pathGT = strcat(lst{j},'/gt/');
	pathPred = strcat(lst{j},'/predicted/');
    shave_p = 4;
	arr=[];
	filePattern_GT = fullfile(pathGT, '*.png');
	files_GT = dir(filePattern_GT);
	for i = 1:length(files_GT)
	    gt_img_path = strcat(pathGT,num2str(i),'.png');
	    pred_img_path = strcat(pathPred,num2str(i),'.png');
	    pred_img = imread(gt_img_path);
	    gt_img = imread(pred_img_path);
	    shaved_gt = shave(gt_img, [shave_p,shave_p]);
	    shaved_pred = shave(pred_img, [shave_p,shave_p]);
	    
	    mse = immse(double(shaved_gt),double(shaved_pred));
	    peaksnr = 10*log10(65025/mse);
	    ssim1 = ssim(shaved_gt,shaved_pred);
	    arr = [arr;peaksnr ssim1];
    end
    k = mean(arr);
	psnr_diff = sota_psnr{j} - k(1);
	ssim_diff = sota_ssim{j} - k(2);
	disp(strcat(lst{j}, '  PSNR ', '       SSIM ','      Diff_PSNR','      Diff_SSIM'));
	%disp('    PSNR       SSIM');
    res = {k(1),k(2),psnr_diff,ssim_diff};
	disp(res);
    %disp(mean(arr));
end
