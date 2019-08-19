CUDA_VISIBLE_DEVICES=5 python test.py --dataroot /data1/color_enhance --phase test --model color_enhance --which_model_netG context_aware --name color_enhance \
--resize_or_crop scale_width_and_crop --loadSize 448 --fineSize 448 --grayscale
