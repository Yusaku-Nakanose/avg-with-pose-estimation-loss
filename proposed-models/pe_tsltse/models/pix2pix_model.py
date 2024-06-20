import torch
import itertools
from . import networks
from .base_model import BaseModel

# 追加
import copy
from src import util
from src.body import Body
from src.hand import Hand
import numpy as np
from PIL import Image
body_model_path = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/pytorch-openpose-master/model/body_pose_model.pth'
#hand_model_path = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/pytorch-openpose-master/model/hand_pose_model.pth'

class Pix2PixModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['G_GAN', 'G_L1', 'G_correction', 'G_coherence', 'D_real', 'D_fake']
        self.visual_names = ['real_audio', 'real_image', 'fake_image1', 'fake_image2']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        if self.isTrain:
            self.model_names = ['G1', 'G2', 'D']
        else:  # during test time, only load G
            self.model_names = ['G1', 'G2']
        if not opt.no_label:
            self.loss_names.append('C_label')
            opt.input_nc = opt.input_nc + opt.label_nc
            if self.isTrain:
                self.model_names.append('C2')
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        if not opt.no_label:
            self.netC2 = networks.define_C('C2', gpu_ids=self.gpu_ids)
        self.netG1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG1, opt.norm,
                                      not opt.no_dropout, not opt.no_attention, opt.init_type, opt.init_gain,
                                      gpu_ids=self.gpu_ids)
        self.netG2 = networks.define_G(opt.input_nc + opt.output_nc, opt.output_nc, opt.ngf, opt.netG2, opt.norm,
                                      not opt.no_dropout, not opt.no_attention, opt.init_type, opt.init_gain,
                                      gpu_ids=self.gpu_ids)
        if self.isTrain:  # only defined during training time
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, not opt.no_attention, opt.init_type, opt.init_gain,
                                          gpu_ids=self.gpu_ids)
            
            # 追加
            self.netOP = Body(body_model_path)
            #self.netHand = Hand(hand_model_path)

            self.weight_G_correction = torch.nn.Parameter(torch.tensor(1.1, requires_grad=True))
            self.weight_G_coherence = torch.nn.Parameter(torch.tensor(1.1, requires_grad=True))

        if self.isTrain:  # only defined during training time
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionL1 = torch.nn.L1Loss()
            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG1.parameters(), self.netG2.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if not opt.no_label:
                self.criterionCEN = torch.nn.CrossEntropyLoss()
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.real_audio = input['audio'].to(self.device)
        self.real_image = input['image'].to(self.device)
        self.audio_label = input['label'].to(self.device)
        self.audio_paths = input['path']

    def cat_label(self, input, label):
        # Replicate spatially and concatenate domain information.
        label = label.view(label.size(0), label.size(1), 1, 1)
        label = label.repeat(1, 1, input.size(2), input.size(3))
        input = torch.cat([input, label], dim=1)
        return input
        

    def cat_input(self, input, stage):
        if not self.opt.no_label and stage == 2:
            # input = torch.cat((input, self.fake_image1), 1)
            input = self.cat_label(input, self.res_label)
        if not self.opt.no_label and stage == 1:
            input = self.cat_label(input, self.audio_label)
        return input


    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.fake_image1 = self.netG1(self.cat_input(self.real_audio, 1)) #stage1
        self.image_label = self.netC2(self.fake_image1)
        self.res_label = self.audio_label - self.image_label
        fake_image = torch.cat((self.fake_image1, self.real_audio), 1)
        self.fake_image2 = self.netG2(self.cat_input(fake_image, 2)) #stage2


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_image1 = torch.cat((self.real_audio, self.fake_image1), 1)
        pred_fake = self.netD(self.cat_input(fake_image1, 2).detach())
        self.loss_D_fake1 = self.criterionGAN(pred_fake, False)
        fake_image2 = torch.cat((self.real_audio, self.fake_image2), 1)
        pred_fake = self.netD(self.cat_input(fake_image2, 2).detach())
        self.loss_D_fake2 = self.criterionGAN(pred_fake, False)
        # Real
        real_AB1 = torch.cat((self.real_audio, self.real_image), 1)
        pred_real = self.netD(self.cat_input(real_AB1, 2))
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D_fake = self.loss_D_fake1 + self.loss_D_fake2
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward(retain_graph=True)

    def backward_C(self):
        self.image_label2 = self.netC2(self.fake_image2)
        index = torch.max(self.audio_label, 1)[1]
        self.loss_C_label = (self.criterionCEN(self.image_label, index) + self.criterionCEN(self.image_label2, index)) * 0.5
        self.loss_C_label.backward(retain_graph=True)

    def backward_G(self):
        batch_size = self.real_audio.shape[0]
        # Discriminator
        fake_image1 = torch.cat((self.real_audio, self.fake_image1), 1)
        pred_fake1 = self.netD(self.cat_input(fake_image1, 1))
        self.loss_G_GAN1 = self.criterionGAN(pred_fake1, True)

        fake_image2 = torch.cat((self.real_audio, self.fake_image2), 1)
        pred_fake2 = self.netD(self.cat_input(fake_image2, 1))
        self.loss_G_GAN = self.criterionGAN(pred_fake2, True)

        self.loss_G_L1_1 = self.criterionL1(self.fake_image1, self.real_image) * self.opt.lambda_L1
        self.loss_G_L1 = self.criterionL1(self.fake_image2, self.real_image) * self.opt.lambda_L1

        # 追加
        if self.opt.epoch_count == 1:
            self.prev_fake_keypoints1 = np.full((batch_size,18,2), -1)
            self.prev_fake_keypoints2 = np.full((batch_size,18,2), -1)
        else:
            self.prev_fake_keypoints1 = self.fake_keypoints1 #前回のバッチデータに対する検出点を保持（∵最後のデータが必要） ex) prev_fake_keypoints[96,97,98,99]
            self.prev_fake_keypoints2 = self.fake_keypoints2
        
        self.real_keypoints = self.extract_keypoints(self.real_image, batch_size)
        self.fake_keypoints1 = self.extract_keypoints(self.fake_image1, batch_size) # 今回のバッチデータに対する検出点を抽出 ex) fake_keypoints[100,101,102,103]
        self.fake_keypoints2 = self.extract_keypoints(self.fake_image2, batch_size)

        self.prev_fake_keypoints1[0:(batch_size - 1)] = self.fake_keypoints1[0:(batch_size - 1)] #前のデータの最後の一つ以外を変更 ex) prev_fake_keypoints[100,101,102,99]
        self.prev_fake_keypoints2[0:(batch_size - 1)] = self.fake_keypoints2[0:(batch_size - 1)]
        #下の3行は順番の入れ替え
        tmp_array1 = self.prev_fake_keypoints1 # ex) tmp_array1[100,101,102,99]
        self.prev_fake_keypoints1[1:] = tmp_array1[0:(batch_size-1)] #prevの最初以外にtmpの最後以外を代入 ex) prev_fake_keypoints[100,100,101,102]
        self.prev_fake_keypoints1[0] = tmp_array1[batch_size - 1] #prevの最初にtmpの最後を代入 ex) prev_fake_keypoints[99,100,101,102]
        tmp_array2 = self.prev_fake_keypoints2
        self.prev_fake_keypoints2[1:] = tmp_array2[0:(batch_size-1)]
        self.prev_fake_keypoints2[0] = tmp_array2[batch_size - 1]
        
        # GTと生成画像間の姿勢の差に関する損失(GTの姿勢に矯正する損失)
        self.loss_G_correction1 = self.criterionOP(self.real_keypoints, self.fake_keypoints1)
        self.loss_G_correction = self.criterionOP(self.real_keypoints, self.fake_keypoints2)

        # 生成画像とその前の生成画像間で姿勢の差に関する損失(連続するフレームで姿勢を似せてコヒーレンスにする損失)
        self.loss_G_coherence1 = self.criterionOP(self.fake_keypoints1, self.prev_fake_keypoints1)
        self.loss_G_coherence = self.criterionOP(self.fake_keypoints2, self.prev_fake_keypoints2)

        self.loss_G_L1 += self.loss_G_L1_1
        self.loss_G_GAN += self.loss_G_GAN1
        self.loss_G_correction = (self.loss_G_correction + self.loss_G_correction1) * self.weight_G_correction
        self.loss_G_coherence = (self.loss_G_coherence + self.loss_G_coherence1) * self.weight_G_coherence
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_correction + self.loss_G_coherence
        self.loss_G.backward(retain_graph=True)
        print(f"weight_G_correction:{self.weight_G_correction}")
        print(f"weight_G_coherence:{self.weight_G_coherence}")

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()  # first call forward to calculate intermediate results
        self.set_requires_grad(self.netC2, False)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights

        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_C()
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights


    # 追加
    def tensor2image(self, input_image, imtype=np.uint8):
        """"Converts a Tensor array into a numpy image array.

        Parameters:
            input_image (tensor) --  the input image tensor array
            imtype (type)        --  the desired type of the converted numpy array
        """
        if not isinstance(input_image, np.ndarray):
            if isinstance(input_image, torch.Tensor):  # get the data from a variable # input_imageがtorch.Tensorである場合
                image_tensor = input_image.data
            else: # input_imageがnp.ndarrayでもtorch.Tensorでもない場合
                return input_image
            image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
            if image_numpy.shape[0] == 1:  # grayscale to RGB
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        else:  # if it is a numpy array, do nothing
            image_numpy = input_image
        return image_numpy.astype(imtype)
    
    def seve_image(self, image_numpy):
        im = image_numpy.astype(np.uint8).copy()
        self.real_image_canvas = copy.deepcopy(im)
        self.real_image_canvas = util.draw_bodypose(im, self.real_iamge_candidate, self.real_image_subset)
        self.real_image_canvas_pil = Image.fromarray(self.real_image_canvas)
        self.real_image_canvas_pil.save("result/test.png")

    def extract_keypoints(self, images, batch_size): # バッチ内のすべての画像に対して検出
        keypoints_set = []
        loop = batch_size
        if not images.shape[0] == batch_size:
            loop = len(images)
        for i in range(loop):
            image_numpy = self.tensor2image(images[i])
            keypoints = self.get_keypoints(image_numpy)
            keypoints_set.append(keypoints)
        keypoints_set = np.array(keypoints_set)
        return keypoints_set

    def get_keypoints(self, image_numpy): # 単一画像に対して検出
        im = image_numpy.astype(np.uint8).copy()
        candidate, subset = self.netOP(im)

        keypoints = np.full((18,2), -1)

        j = 0
        if subset.shape == (0, 20):
            subset = np.full((1,20), -1)
        for i in range(18):
            if not subset[0][i] == -1:
                keypoints[i][0] = candidate[j][0]
                keypoints[i][1] = candidate[j][1]
                j += 1

        return keypoints
    
    def criterionOP(self, keypoint1, keypoint2):
        total_loss = 0.0  # 損失の合計を初期化
        loop = self.opt.batch_size
        n = 0
        if not keypoint1.shape[0] == self.opt.batch_size:
            loop = keypoint1.shape[0]
        for i in range(loop):
            data1 = keypoint1[i].flatten()
            data2 = keypoint2[i].flatten()
            diff = []

            for j in range(36): # 18箇所に対してx,y座標があるからサイズは36
                if not data1[j] == -1:
                    if not data2[j] == -1: # real_data
                        diff.append(data1[j] - data2[j])
                        n += 1

            if diff:  # diffが空でない場合のみMSEを計算
                loss = torch.nn.MSELoss()(torch.tensor(diff, dtype=torch.float32), torch.tensor([0.0]))  # MSEを計算
                total_loss += loss.item()  # 損失の合計に追加
                total_loss = total_loss / n

        return total_loss  # 総合的な損失を返す