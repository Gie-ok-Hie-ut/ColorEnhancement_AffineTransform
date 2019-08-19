CUDA_VISIBLE_DEVICES=4 python train.py --dataroot /data1/color_enhance --print_freq 50 \
--gpu_ids 0 --display_winsize 200 \
--ngf 64 --no_html --name color_enhance --display_id 1 --display_freq 10 \
--display_port 6005 --model color_enhance --input_nc 3 --output_nc 3 \
--which_model_netG context_aware --which_model_netD context_aware --which_direction AtoB \
--norm batch --batchSize 1 --n_layers_D 4 --resize_or_crop scale_width_and_crop \
--loadSize 300 --fineSize 224 --grayscale
