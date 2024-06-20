import argparse
import os
import lpips

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='./imgs/ex_dir0')
parser.add_argument('-d1','--dir1', type=str, default='./imgs/ex_dir1')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()


## Initializing the model
#loss_fn = lpips.LPIPS(net='vgg',version=opt.version)
loss_fn = lpips.LPIPS(net='alex',version=opt.version)
if(opt.use_gpu):
	loss_fn.cuda()

# files:生成画像
# files1:GT画像
f = open(opt.out,'w')
files = os.listdir(opt.dir0)
files = sorted(files)

###↓trombone02_xxxxx_128.pngってでる
""" for i in files:
	#dir1
	print(i)
	#print(i.split("fake")[0]+"128.png") """

files1 = os.listdir(opt.dir1)
files1 = sorted(files1)
""" for j in files1:
	#dir1
	print(j)
	#print(j.split("fake")[0]+"128.png") """

total = 0.0
for num in range(100):
	# print(file.split("_")[0])
	# print(a)
	#print(os.path.join(opt.dir0,file))
	if(os.path.exists(os.path.join(opt.dir0,files[num]))):
		# Load images
		img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,files[num]))) # RGB image from [-1,1]
		#img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file)))
		
		#img1 = lpips.im2tensor(lpips.load_image(opt.dir1 + "/" + file.split("fake")[0]+"128.png"))
		img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,files1[num])))

		if(opt.use_gpu):
			img0 = img0.cuda()
			img1 = img1.cuda()

		# Compute distance
		dist01 = loss_fn.forward(img0,img1)
		total = total + dist01
		#print('%s: %.3f'%(files1[num],dist01))
		f.writelines('%s: %.6f\n'%(files1[num],dist01))
f.writelines('\n%.6f\n'%(total / 100.0))
f.close()

""" total = 0.0
for file in files:
	print(file)
	#print(os.path.join(opt.dir0,file))
	if(os.path.exists(os.path.join(opt.dir0,file))):
		# Load images
		img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
		#img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file)))
		img1 = lpips.im2tensor(lpips.load_image(opt.dir1 + "/" + file.split("fake")[0]+"128.png"))

		if(opt.use_gpu):
			img0 = img0.cuda()
			img1 = img1.cuda()

		# Compute distance
		dist01 = loss_fn.forward(img0,img1)
		total = total + dist01
		print('%s: %.3f'%(file,dist01))
		f.writelines('%s: %.6f\n'%(file,dist01))
f.writelines('\n%.6f\n'%(total))
f.close() """