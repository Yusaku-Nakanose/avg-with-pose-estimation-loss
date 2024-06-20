#satoshi
import torch
import itertools
from . import networks
from .base_model import BaseModel
import random

#追加
import copy
from src import util
from src.body import Body
from src.hand import Hand
import numpy as np
from PIL import Image
import os
body_model_path = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/pytorch-openpose-master/model/body_pose_model.pth'
#hand_model_path = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/pytorch-openpose-master/model/hand_pose_model.pth'
output_directory = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/'
output_file_path = os.path.join(output_directory, "weights.txt")
weight = 0.01
class Pix2PixconfulladdModel(BaseModel):

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
        self.loss_names = ['G_GAN', 'G_L1', 'G_correction', 'G_coherence', 'D_i_real', 'D_i_fake',  'D_v_real', 'D_v_fake']
        self.visual_names = ['real_audio', 'real_image1', 'real_image2', 'real_image3', 'real_image4', 'fake_image1_1', 'fake_image1_2', 'fake_image1_3', 'fake_image1_4','fake_image2_1', 'fake_image2_2','fake_image2_3','fake_image2_4']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        if self.isTrain:
            self.model_names = ['G1', 'G2', 'D_i', 'D_v']
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
        self.netG1 = networks.define_G(opt.input_nc+1, opt.output_nc, opt.ngf, opt.netG1, opt.norm,
                                      not opt.no_dropout, not opt.no_attention, opt.init_type, opt.init_gain,
                                      gpu_ids=self.gpu_ids)
        ##0711変更
        self.netG2 = networks.define_G(opt.input_nc+1, opt.output_nc, opt.ngf, opt.netG2, opt.norm,
                                      not opt.no_dropout, not opt.no_attention, opt.init_type, opt.init_gain,
                                      gpu_ids=self.gpu_ids)
        """ self.netG2 = networks.define_G_3d(opt.input_nc, opt.output_nc, opt.ngf, opt.netG2, opt.norm,
                                      not opt.no_dropout, not opt.no_attention, opt.init_type, opt.init_gain,
                                      gpu_ids=self.gpu_ids) """
        if self.isTrain:  # only defined during training time
            self.netD_i = networks.define_D_i(opt.input_nc + opt.output_nc+1, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, not opt.no_attention, opt.init_type, opt.init_gain,
                                          gpu_ids=self.gpu_ids)
            self.netD_v = networks.define_D_v(20, opt.ndf, opt.netD,
                                          opt.n_layers_D, 'batch3d', not opt.no_attention, opt.init_type, opt.init_gain,
                                          gpu_ids=self.gpu_ids)
            """ print("#############")
            print(self.netD_v) """

            # 追加
            self.netBody = Body(body_model_path)
            #self.netHand = Hand(hand_model_path)

        if self.isTrain:  # only defined during training time
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionL1 = torch.nn.L1Loss()
            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            
            #self.weight_G_coherence = torch.nn.Parameter(torch.tensor(0.01), requires_grad=True)
            #self.weight_G_correction = torch.nn.Parameter(torch.tensor(0.01), requires_grad=True)
            
            #self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG1.parameters(), self.netG2.parameters(), [self.weight_G_coherence], [self.weight_G_correction]), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG1.parameters(), self.netG2.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer_W = torch.optim.Adam(itertools.chain([self.weight_G_coherence], [self.weight_G_correction]), lr=opt.lr*0.1, betas=(opt.beta1, 0.999))
            self.optimizer_D_i = torch.optim.Adam(self.netD_i.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_v = torch.optim.Adam(self.netD_v.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_i)
            self.optimizers.append(self.optimizer_D_v)


            if not opt.no_label:
                self.criterionCEN = torch.nn.CrossEntropyLoss()
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.real_audio = input['audio'].to(self.device)
        self.real_image1 = input['image1'].to(self.device)
        self.real_image2 = input['image2'].to(self.device)
        self.real_image3 = input['image3'].to(self.device)
        self.real_image4 = input['image4'].to(self.device)
        self.audio_label = input['label'].to(self.device)
        self.audio_paths = input['path']

    def cat_label(self, input, label):
        # Replicate spatially and concatenate domain information.
        label = label.view(label.size(0), label.size(1), 1, 1)
        label = label.repeat(1, 1, input.size(2), input.size(3))
        input = torch.cat([input, label], dim=1)
        return input

    def cat_input(self, input, stage, spec_num):
        if not self.opt.no_label and stage == 2:
            # input = torch.cat((input, self.fake_image1), 1)
            # より細かく条件分け
            if spec_num == 1:
                input = self.cat_label(input, self.res_label1_1)
            elif spec_num == 2:
                input = self.cat_label(input, self.res_label1_2)
            elif spec_num == 3:
                input = self.cat_label(input, self.res_label1_3)
            elif spec_num == 4:
                input = self.cat_label(input, self.res_label1_4)
            else:
                input = self.cat_label(input, self.res_label1_1)
        if not self.opt.no_label and stage == 1:
            input = self.cat_label(input, self.audio_label)
        return input


    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        
        batch_size = self.real_audio.shape[0]
        mask1 = torch.zeros(batch_size, 1, 128, 128)
        mask2 = torch.zeros(batch_size, 1, 128, 128)
        mask3 = torch.zeros(batch_size, 1, 128, 128)
        mask4 = torch.zeros(batch_size, 1, 128, 128)
        mask1[:, :, :,12:39] = 1 #13～39
        mask2[:, :, :,38:64] = 1
        mask3[:, :, :,63:89] = 1
        mask4[:, :, :,88:114] = 1
        mask1 = mask1.to(self.device)
        mask2 = mask2.to(self.device)
        mask3 = mask3.to(self.device)
        mask4 = mask4.to(self.device)
        

        self.fake_image1_1 = self.netG1(self.cat_input(torch.cat([self.real_audio, mask1], dim=1), 1, 1))
        self.fake_image1_2 = self.netG1(self.cat_input(torch.cat([self.real_audio, mask2], dim=1), 1, 2))
        self.fake_image1_3 = self.netG1(self.cat_input(torch.cat([self.real_audio, mask3], dim=1), 1, 3))
        self.fake_image1_4 = self.netG1(self.cat_input(torch.cat([self.real_audio, mask4], dim=1), 1, 4))

        

        #全種類ではなく一枚だけで分類するとふわっとしたラベルになるのかも？
        self.image_label1_1 = self.netC2(self.fake_image1_1)
        self.res_label1_1 = self.audio_label - self.image_label1_1
        self.image_label1_2 = self.netC2(self.fake_image1_2)
        self.res_label1_2 = self.audio_label - self.image_label1_2
        self.image_label1_3 = self.netC2(self.fake_image1_3)
        self.res_label1_3 = self.audio_label - self.image_label1_3
        self.image_label1_4 = self.netC2(self.fake_image1_4)
        self.res_label1_4 = self.audio_label - self.image_label1_4
        

        self.fake_image2_1 = self.netG2(self.cat_input(torch.cat([self.fake_image1_1, torch.zeros(batch_size, 1, 128, 128).to(self.device)],dim=1), 2, 1))
        self.fake_image2_2 = self.netG2(self.cat_input(torch.cat([self.fake_image1_2, torch.zeros(batch_size, 1, 128, 128).to(self.device)],dim=1), 2, 2))
        self.fake_image2_3 = self.netG2(self.cat_input(torch.cat([self.fake_image1_3, torch.zeros(batch_size, 1, 128, 128).to(self.device)],dim=1), 2, 3))
        self.fake_image2_4 = self.netG2(self.cat_input(torch.cat([self.fake_image1_4, torch.zeros(batch_size, 1, 128, 128).to(self.device)],dim=1), 2, 4))

        #self.fake_image2 = self.netG2(self.cat_input(fake_image, 2)) #stage2


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # 生成した画像をfakeとみわけたい
        #criterionGANは多分BCE
        #specとラベルいれていいのか？

        batch_size = self.real_audio.shape[0]
        mask1 = torch.zeros(batch_size, 1, 128, 128)
        mask2 = torch.zeros(batch_size, 1,128, 128)
        mask3 = torch.zeros(batch_size, 1,128, 128)
        mask4 = torch.zeros(batch_size, 1,128, 128)
        mask1[:, :, :,12:39] = 1 #13～39
        mask2[:, :, :,38:64] = 1
        mask3[:, :, :,63:89] = 1
        mask4[:, :, :,88:114] = 1
        mask1 = mask1.to(self.device)
        mask2 = mask2.to(self.device)
        mask3 = mask3.to(self.device)
        mask4 = mask4.to(self.device)

        f1_1 = torch.cat((torch.cat([self.real_audio, mask1], dim=1), self.fake_image1_1), 1)
        p1_1 = self.netD_i(self.cat_input(f1_1, 2, 1).detach())
        f1_2 = torch.cat((torch.cat([self.real_audio, mask2], dim=1), self.fake_image1_2), 1)
        p1_2 = self.netD_i(self.cat_input(f1_2, 2, 2).detach())
        f1_3 = torch.cat((torch.cat([self.real_audio, mask3], dim=1), self.fake_image1_3), 1)
        p1_3 = self.netD_i(self.cat_input(f1_3, 2, 3).detach())
        f1_4 = torch.cat((torch.cat([self.real_audio, mask4], dim=1), self.fake_image1_4), 1)
        p1_4 = self.netD_i(self.cat_input(f1_4, 2, 4).detach())
        self.loss_D_fake1_1 = (self.criterionGAN(p1_1, False) + self.criterionGAN(p1_2, False) + self.criterionGAN(p1_3, False) + self.criterionGAN(p1_4, False)) / 4

        f2_1 = torch.cat((torch.cat([self.real_audio, mask1], dim=1), self.fake_image2_1), 1)
        p2_1 = self.netD_i(self.cat_input(f2_1, 2, 1).detach())
        f2_2 = torch.cat((torch.cat([self.real_audio, mask2], dim=1), self.fake_image2_2), 1)
        p2_2 = self.netD_i(self.cat_input(f2_2, 2, 2).detach())
        f2_3 = torch.cat((torch.cat([self.real_audio, mask3], dim=1), self.fake_image2_3), 1)
        p2_3 = self.netD_i(self.cat_input(f2_3, 2, 3).detach())
        f2_4 = torch.cat((torch.cat([self.real_audio, mask4], dim=1), self.fake_image2_4), 1)
        p2_4 = self.netD_i(self.cat_input(f2_4, 2, 4).detach())
        self.loss_D_fake2_1 = (self.criterionGAN(p2_1, False) + self.criterionGAN(p2_2, False) + self.criterionGAN(p2_3, False) + self.criterionGAN(p2_4, False)) / 4

        """ fake_image2 = torch.cat((self.real_audio, self.fake_image2_1), 1)
        pred_fake = self.netD_i(self.cat_input(fake_image2, 2).detach())
        self.loss_D_fake2_1 = self.criterionGAN(pred_fake, False) """

        # Real
        real_AB1 = torch.cat((torch.cat([self.real_audio, mask1], dim=1), self.real_image1), 1)
        real_AB2 = torch.cat((torch.cat([self.real_audio, mask2], dim=1), self.real_image2), 1)
        real_AB3 = torch.cat((torch.cat([self.real_audio, mask3], dim=1), self.real_image3), 1)
        real_AB4 = torch.cat((torch.cat([self.real_audio, mask4], dim=1), self.real_image4), 1)
        pred_real1 = self.netD_i(self.cat_input(real_AB1, 2, 1))
        pred_real2 = self.netD_i(self.cat_input(real_AB2, 2, 2))
        pred_real3 = self.netD_i(self.cat_input(real_AB3, 2, 3))
        pred_real4 = self.netD_i(self.cat_input(real_AB4, 2, 4))

        self.loss_D_i_real = (self.criterionGAN(pred_real1, True) + self.criterionGAN(pred_real2, True) + self.criterionGAN(pred_real3, True) + self.criterionGAN(pred_real4, True)) / 4
        self.loss_D_i_fake = self.loss_D_fake1_1 + self.loss_D_fake2_1
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_i_fake + self.loss_D_i_real) * 0.5
        self.loss_D.backward(retain_graph=True)

    def backward_D_v(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        #print(self.fake_image1_2.size()) #[4, 3, 128, 128]
        # fake_image1 = torch.stack((self.fake_image1_4,self.fake_image1_3,self.fake_image1_2, self.fake_image1_1, self.real_audio), 2)
        # fake_image2 = torch.stack((self.fake_image1_3,self.fake_image1_1,self.fake_image1_2, self.fake_image1_4, self.real_audio), 2)
        # fake_image3 = torch.stack((self.fake_image1_2,self.fake_image1_4,self.fake_image1_1, self.fake_image1_3, self.real_audio), 2)
        # fake_image4 = torch.stack((self.fake_image1_1,self.fake_image1_4,self.fake_image1_2, self.fake_image1_3, self.real_audio), 2)
        # """ print(fake_image1.size())
        # print(a) """
        # #fake_image1 = torch.cat((self.fake_image1_4,self.fake_image1_3,self.fake_image1_2, self.fake_image1_1), 1)
        # #print(fake_image1.size()) #[4, 6, 128, 128]
        # #print(self.cat_input(fake_image1, 2).detach().size())##[4, 6, 128, 128]
        # pred_fake1 = self.netD_v(fake_image1.detach())
        # pred_fake2 = self.netD_v(fake_image2.detach())
        # pred_fake3 = self.netD_v(fake_image3.detach())
        # pred_fake4 = self.netD_v(fake_image4.detach())
        # #print(pred_fake.size()) #[4, 1, 14, 14]
        # self.loss_D_v_fake = (self.criterionGAN(pred_fake1, False) + self.criterionGAN(pred_fake2, False) + self.criterionGAN(pred_fake3, False) + self.criterionGAN(pred_fake4, False)) / 4
        # #print(a)

    
        # # Real
        # real_image1 = torch.stack((self.fake_image1_1,self.fake_image1_2,self.fake_image1_3, self.fake_image1_4, self.real_audio), 2)
        # pred_real = self.netD_v(real_image1.detach())

        # self.loss_D_v_real = self.criterionGAN(pred_real, True)

        #1つをちゃんとシャッフルする(12/23)
        # 4, 19, 4, 128, 128
        #4, 19, 128, 128

        batch_size = self.real_audio.shape[0]
        mask1 = torch.zeros(batch_size, 1, 128, 128)
        mask2 = torch.zeros(batch_size, 1,128, 128)
        mask3 = torch.zeros(batch_size, 1,128, 128)
        mask4 = torch.zeros(batch_size, 1,128, 128)
        mask1[:, :, :,12:39] = 1 #13～39
        mask2[:, :, :,38:64] = 1
        mask3[:, :, :,63:89] = 1
        mask4[:, :, :,88:114] = 1
        mask1 = mask1.to(self.device)
        mask2 = mask2.to(self.device)
        mask3 = mask3.to(self.device)
        mask4 = mask4.to(self.device)

        #self.fake_image1_1 = self.netG1(self.cat_input(torch.cat([self.real_audio, mask1], dim=1), 1, 1))

        list = [[1,2,4,3],[1,3,2,4],[1,3,4,2],[1,4,2,3],[1,4,3,2],[2,1,3,4],[2,1,4,3],[2,3,1,4],[2,3,4,1],[2,4,1,3],[2,4,3,1],[3,1,2,4],[3,1,4,2],[3,2,1,4],[3,2,4,1],[3,4,1,2],[3,4,2,1],[4,1,2,3],[4,1,3,2],[4,2,1,3],[4,2,3,1],[4,3,1,2],[4,3,2,1]]
        li_ran = random.choice(list)
        f_list = [self.fake_image1_1, self.fake_image1_2, self.fake_image1_3, self.fake_image1_4]
        m_list = [mask1, mask2, mask3, mask4]
        """ print(f_list[li_ran[0]-1].size())
        print(torch.cat([self.real_audio, m_list[li_ran[0]-1]], dim=1).size())
        print(torch.cat((f_list[li_ran[0]-1], torch.cat([self.real_audio, m_list[li_ran[0]-1]], dim=1)),dim=1).size()) """
        fake_image1 = torch.stack((
                                    self.cat_input(torch.cat((f_list[li_ran[0]-1], torch.cat([self.real_audio, m_list[li_ran[0]-1]], dim=1)),dim=1), 2, li_ran[0]),
                                    self.cat_input(torch.cat((f_list[li_ran[1]-1], torch.cat([self.real_audio, m_list[li_ran[1]-1]], dim=1)),dim=1), 2, li_ran[1]),
                                    self.cat_input(torch.cat((f_list[li_ran[2]-1], torch.cat([self.real_audio, m_list[li_ran[2]-1]], dim=1)),dim=1), 2, li_ran[2]), 
                                    self.cat_input(torch.cat((f_list[li_ran[3]-1], torch.cat([self.real_audio, m_list[li_ran[3]-1]], dim=1)),dim=1), 2, li_ran[3])), 2)
        #print(fake_image1.size())
        
        pred_fake1 = self.netD_v(fake_image1.detach())
        
        self.loss_D_v_fake = (self.criterionGAN(pred_fake1, False))

    
        # Real
        #real_image1 = torch.stack((self.cat_input(torch.cat((self.fake_image1_1, self.real_image1),1), 2, 1),self.cat_input(torch.cat((self.fake_image1_2, self.real_image2),1), 2, 2),self.cat_input(torch.cat((self.fake_image1_3, self.real_image3),1), 2, 3), self.cat_input(torch.cat((self.fake_image1_4, self.real_image1),1), 2, 4)), 2)
        real_image1 = torch.stack((self.cat_input(torch.cat((self.fake_image1_1, torch.cat([self.real_audio, mask1], dim=1)),dim=1), 2, 1),
                                    self.cat_input(torch.cat((self.fake_image1_2, torch.cat([self.real_audio, mask2], dim=1)),dim=1), 2, 2),
                                    self.cat_input(torch.cat((self.fake_image1_3, torch.cat([self.real_audio, mask3], dim=1)),dim=1), 2, 3), 
                                    self.cat_input(torch.cat((self.fake_image1_4, torch.cat([self.real_audio, mask4], dim=1)),dim=1), 2, 4)), 2)
        pred_real = self.netD_v(real_image1.detach())

        

        self.loss_D_v_real = self.criterionGAN(pred_real, True)

        # combine loss and calculate gradients
        self.loss_D_v = (self.loss_D_v_fake + self.loss_D_v_real) * 0.5
        self.loss_D_v.backward(retain_graph=True)

    def backward_C(self):
        self.image_label2 = self.netC2(self.fake_image2_1)
        index = torch.max(self.audio_label, 1)[1]
        self.loss_C_label = (self.criterionCEN(self.image_label1_1, index) + self.criterionCEN(self.image_label1_2, index) + self.criterionCEN(self.image_label1_3, index) + self.criterionCEN(self.image_label1_4, index) + self.criterionCEN(self.image_label2, index)) * 0.2
        self.loss_C_label.backward(retain_graph=True)

    def backward_G(self):
        batch_size = self.real_audio.shape[0]
        mask1 = torch.zeros(batch_size, 1, 128, 128)
        mask2 = torch.zeros(batch_size, 1,128, 128)
        mask3 = torch.zeros(batch_size, 1,128, 128)
        mask4 = torch.zeros(batch_size, 1,128, 128)
        mask1[:, :, :,12:39] = 1 #13～39
        mask2[:, :, :,38:64] = 1
        mask3[:, :, :,63:89] = 1
        mask4[:, :, :,88:114] = 1
        mask1 = mask1.to(self.device)
        mask2 = mask2.to(self.device)
        mask3 = mask3.to(self.device)
        mask4 = mask4.to(self.device)
        # 本物のように生成したいのでDをだます
        fake_im1 = torch.cat((torch.cat([self.real_audio, mask1], dim=1), self.fake_image1_1), 1)
        pred_f1_1 = self.netD_i(self.cat_input(fake_im1, 1, 1))

        fake_im2 = torch.cat((torch.cat([self.real_audio, mask2], dim=1), self.fake_image1_2), 1)
        pred_f1_2 = self.netD_i(self.cat_input(fake_im2, 1, 1))

        fake_im3 = torch.cat((torch.cat([self.real_audio, mask3], dim=1), self.fake_image1_1), 1)
        pred_f1_3 = self.netD_i(self.cat_input(fake_im3, 1, 1))

        fake_im4 = torch.cat((torch.cat([self.real_audio, mask4], dim=1), self.fake_image1_1), 1)
        pred_f1_4 = self.netD_i(self.cat_input(fake_im4, 1, 1))

        loss_G_GAN1_1 = (self.criterionGAN(pred_f1_1, True) + self.criterionGAN(pred_f1_2, True) + self.criterionGAN(pred_f1_3, True) + self.criterionGAN(pred_f1_4, True))/4

        fake_im2_1 = torch.cat((torch.cat([self.real_audio, mask1], dim=1), self.fake_image2_1), 1)
        pred_f2_1 = self.netD_i(self.cat_input(fake_im2_1, 1, 1))

        fake_im2_2 = torch.cat((torch.cat([self.real_audio, mask2], dim=1), self.fake_image2_2), 1)
        pred_f2_2 = self.netD_i(self.cat_input(fake_im2_2, 1, 1))

        fake_im2_3 = torch.cat((torch.cat([self.real_audio, mask3], dim=1), self.fake_image2_3), 1)
        pred_f2_3 = self.netD_i(self.cat_input(fake_im2_3, 1, 1))

        fake_im2_4 = torch.cat((torch.cat([self.real_audio, mask4], dim=1), self.fake_image2_4), 1)
        pred_f2_4 = self.netD_i(self.cat_input(fake_im2_4, 1, 1))

        loss_G_GAN2_1 = (self.criterionGAN(pred_f2_1, True) + self.criterionGAN(pred_f2_2, True) + self.criterionGAN(pred_f2_3, True) + self.criterionGAN(pred_f2_4, True))/4
        
        # 1枚ずつL1をとっている
        self.loss_G_L1_1 = (self.criterionL1(self.fake_image1_1, self.real_image1) * self.opt.lambda_L1 + self.criterionL1(self.fake_image1_2, self.real_image2) * self.opt.lambda_L1 + self.criterionL1(self.fake_image1_3, self.real_image3) * self.opt.lambda_L1 + self.criterionL1(self.fake_image1_4, self.real_image4) * self.opt.lambda_L1)/4
        loss_G_L2_1 = (self.criterionL1(self.fake_image2_1, self.real_image1) * self.opt.lambda_L1 + self.criterionL1(self.fake_image2_2, self.real_image2) * self.opt.lambda_L1 + self.criterionL1(self.fake_image2_3, self.real_image3) * self.opt.lambda_L1 + self.criterionL1(self.fake_image2_4, self.real_image4) * self.opt.lambda_L1)/4


        # 追加
        #"""
        # 案1：一つ前に生成したデータと比較(CAR-GANではこちら)
        # 一つ前に，0,1,2,3データから生成した0”,1”,2”,3”の画像と， 1,2,3,4データから生成した1’,2’,3’,4’の画像に対して， 0”⇔1’，1”⇔2’，2”⇔3’，3”⇔4’
        if self.opt.epoch_count == 1:
            self.prev_fake_keypoints1_1 = np.full((batch_size,18,2), -1)
            self.prev_fake_keypoints1_2 = np.full((batch_size,18,2), -1)
            self.prev_fake_keypoints1_3 = np.full((batch_size,18,2), -1)
            self.prev_fake_keypoints1_4 = np.full((batch_size,18,2), -1)
            self.prev_fake_keypoints2_1 = np.full((batch_size,18,2), -1)
            self.prev_fake_keypoints2_2 = np.full((batch_size,18,2), -1)
            self.prev_fake_keypoints2_3 = np.full((batch_size,18,2), -1)
            self.prev_fake_keypoints2_4 = np.full((batch_size,18,2), -1)
        else:
            self.prev_fake_keypoints1_1 = self.fake_keypoints1_1
            self.prev_fake_keypoints1_2 = self.fake_keypoints1_2
            self.prev_fake_keypoints1_3 = self.fake_keypoints1_3
            self.prev_fake_keypoints1_4 = self.fake_keypoints1_4
            self.prev_fake_keypoints2_1 = self.fake_keypoints2_1
            self.prev_fake_keypoints2_2 = self.fake_keypoints2_2
            self.prev_fake_keypoints2_3 = self.fake_keypoints2_3
            self.prev_fake_keypoints2_4 = self.fake_keypoints2_4
        
        self.real_keypoints1 = self.extract_keypoints(self.real_image1, batch_size)
        self.real_keypoints2 = self.extract_keypoints(self.real_image2, batch_size)
        self.real_keypoints3 = self.extract_keypoints(self.real_image3, batch_size)
        self.real_keypoints4 = self.extract_keypoints(self.real_image4, batch_size)
        self.fake_keypoints1_1 = self.extract_keypoints(self.fake_image1_1, batch_size)
        self.fake_keypoints1_2 = self.extract_keypoints(self.fake_image1_2, batch_size)
        self.fake_keypoints1_3 = self.extract_keypoints(self.fake_image1_3, batch_size)
        self.fake_keypoints1_4 = self.extract_keypoints(self.fake_image1_4, batch_size)
        self.fake_keypoints2_1 = self.extract_keypoints(self.fake_image2_1, batch_size)
        self.fake_keypoints2_2 = self.extract_keypoints(self.fake_image2_2, batch_size)
        self.fake_keypoints2_3 = self.extract_keypoints(self.fake_image2_3, batch_size)
        self.fake_keypoints2_4 = self.extract_keypoints(self.fake_image2_4, batch_size)

        self.prev_fake_keypoints1_1[0:(batch_size - 1)] = self.fake_keypoints1_1[0:(batch_size - 1)] 
        self.prev_fake_keypoints1_2[0:(batch_size - 1)] = self.fake_keypoints1_2[0:(batch_size - 1)] 
        self.prev_fake_keypoints1_3[0:(batch_size - 1)] = self.fake_keypoints1_3[0:(batch_size - 1)] 
        self.prev_fake_keypoints1_4[0:(batch_size - 1)] = self.fake_keypoints1_4[0:(batch_size - 1)] 
        self.prev_fake_keypoints2_1[0:(batch_size - 1)] = self.fake_keypoints2_1[0:(batch_size - 1)]
        self.prev_fake_keypoints2_2[0:(batch_size - 1)] = self.fake_keypoints2_2[0:(batch_size - 1)]
        self.prev_fake_keypoints2_3[0:(batch_size - 1)] = self.fake_keypoints2_3[0:(batch_size - 1)]
        self.prev_fake_keypoints2_4[0:(batch_size - 1)] = self.fake_keypoints2_4[0:(batch_size - 1)]

        tmp_array1_1 = self.prev_fake_keypoints1_1
        tmp_array1_2 = self.prev_fake_keypoints1_2
        tmp_array1_3 = self.prev_fake_keypoints1_3
        tmp_array1_4 = self.prev_fake_keypoints1_4
        tmp_array2_1 = self.prev_fake_keypoints2_1
        tmp_array2_2 = self.prev_fake_keypoints2_2
        tmp_array2_3 = self.prev_fake_keypoints2_3
        tmp_array2_4 = self.prev_fake_keypoints2_4
        self.prev_fake_keypoints1_1[1:] = tmp_array1_1[0:(batch_size - 1)]
        self.prev_fake_keypoints1_2[1:] = tmp_array1_2[0:(batch_size - 1)]
        self.prev_fake_keypoints1_3[1:] = tmp_array1_3[0:(batch_size - 1)]
        self.prev_fake_keypoints1_4[1:] = tmp_array1_4[0:(batch_size - 1)]
        self.prev_fake_keypoints2_1[1:] = tmp_array2_1[0:(batch_size - 1)]
        self.prev_fake_keypoints2_2[1:] = tmp_array2_2[0:(batch_size - 1)]
        self.prev_fake_keypoints2_3[1:] = tmp_array2_3[0:(batch_size - 1)]
        self.prev_fake_keypoints2_4[1:] = tmp_array2_4[0:(batch_size - 1)]
        self.prev_fake_keypoints1_1[0] = tmp_array1_1[batch_size - 1]
        self.prev_fake_keypoints1_2[0] = tmp_array1_2[batch_size - 1]
        self.prev_fake_keypoints1_3[0] = tmp_array1_3[batch_size - 1]
        self.prev_fake_keypoints1_4[0] = tmp_array1_4[batch_size - 1]
        self.prev_fake_keypoints2_1[0] = tmp_array2_1[batch_size - 1]
        self.prev_fake_keypoints2_2[0] = tmp_array2_2[batch_size - 1]
        self.prev_fake_keypoints2_3[0] = tmp_array2_3[batch_size - 1]
        self.prev_fake_keypoints2_4[0] = tmp_array2_4[batch_size - 1]

        # 4枚出力するが，最終的には1枚目しか使わないため，1枚目の重みを他の倍に
        self.loss_G_coherence1 = (2*self.criterionOP(self.fake_keypoints1_1, self.prev_fake_keypoints1_1) + self.criterionOP(self.fake_keypoints1_2, self.prev_fake_keypoints1_2) + self.criterionOP(self.fake_keypoints1_3, self.prev_fake_keypoints1_3) + self.criterionOP(self.fake_keypoints1_4, self.prev_fake_keypoints1_4)) / 5
        self.loss_G_coherence = (2*self.criterionOP(self.fake_keypoints2_1, self.prev_fake_keypoints2_1) + self.criterionOP(self.fake_keypoints2_2, self.prev_fake_keypoints2_2) + self.criterionOP(self.fake_keypoints2_3, self.prev_fake_keypoints2_3) + self.criterionOP(self.fake_keypoints2_4, self.prev_fake_keypoints2_4)) / 5

        #self.loss_G_coherence = abs((self.loss_G_coherence + self.loss_G_coherence1) * self.weight_G_coherence) # """
        self.loss_G_coherence = (self.loss_G_coherence + self.loss_G_coherence1) * weight


        """
        # 案2：同一データ内で比較
        # 1,2,3,4データから生成した1’,2’,3’,4’の画像に対して，1’⇔2’，2’⇔3’，3’⇔4’
        self.real_keypoints1 = self.extract_keypoints(self.real_image1, batch_size)
        self.real_keypoints2 = self.extract_keypoints(self.real_image2, batch_size)
        self.real_keypoints3 = self.extract_keypoints(self.real_image3, batch_size)
        self.real_keypoints4 = self.extract_keypoints(self.real_image4, batch_size)
        self.fake_keypoints1_1 = self.extract_keypoints(self.fake_image1_1, batch_size)
        self.fake_keypoints1_2 = self.extract_keypoints(self.fake_image1_2, batch_size)
        self.fake_keypoints1_3 = self.extract_keypoints(self.fake_image1_3, batch_size)
        self.fake_keypoints1_4 = self.extract_keypoints(self.fake_image1_4, batch_size)
        self.fake_keypoints2_1 = self.extract_keypoints(self.fake_image2_1, batch_size)
        self.fake_keypoints2_2 = self.extract_keypoints(self.fake_image2_2, batch_size)
        self.fake_keypoints2_3 = self.extract_keypoints(self.fake_image2_3, batch_size)
        self.fake_keypoints2_4 = self.extract_keypoints(self.fake_image2_4, batch_size)

        self.loss_G_coherence1 = (self.criterionOP(self.fake_keypoints1_1, self.fake_keypoints1_2) + self.criterionOP(self.fake_keypoints1_2, self.fake_keypoints1_3) + self.criterionOP(self.fake_keypoints1_3, self.fake_keypoints1_4)) / 3
        self.loss_G_coherence = (self.criterionOP(self.fake_keypoints2_1, self.fake_keypoints2_2) + self.criterionOP(self.fake_keypoints2_2, self.fake_keypoints2_3) + self.criterionOP(self.fake_keypoints2_3, self.fake_keypoints2_4)) / 3

        self.loss_G_coherence = (self.loss_G_coherence + self.loss_G_coherence1) * 0.01 """


        """
        # 案3：同じ入力に対する出力で比較
        # 一つ前に，0,1,2,3データから生成した0”,1”,2”,3”の画像と， 1,2,3,4データから生成した1’,2’,3’,4’の画像に対して， 1”⇔1’，2”⇔2’，3”⇔3’
        if self.opt.epoch_count == 1:
            self.prev_fake_keypoints1_1 = np.full((batch_size,18,2), -1)
            self.prev_fake_keypoints1_2 = np.full((batch_size,18,2), -1)
            self.prev_fake_keypoints1_3 = np.full((batch_size,18,2), -1)
            self.prev_fake_keypoints1_4 = np.full((batch_size,18,2), -1)
            self.prev_fake_keypoints2_1 = np.full((batch_size,18,2), -1)
            self.prev_fake_keypoints2_2 = np.full((batch_size,18,2), -1)
            self.prev_fake_keypoints2_3 = np.full((batch_size,18,2), -1)
            self.prev_fake_keypoints2_4 = np.full((batch_size,18,2), -1)
        else:
            self.prev_fake_keypoints1_1 = self.fake_keypoints1_1
            self.prev_fake_keypoints1_2 = self.fake_keypoints1_2
            self.prev_fake_keypoints1_3 = self.fake_keypoints1_3
            self.prev_fake_keypoints1_4 = self.fake_keypoints1_4
            self.prev_fake_keypoints2_1 = self.fake_keypoints2_1
            self.prev_fake_keypoints2_2 = self.fake_keypoints2_2
            self.prev_fake_keypoints2_3 = self.fake_keypoints2_3
            self.prev_fake_keypoints2_4 = self.fake_keypoints2_4
        
        self.real_keypoints1 = self.extract_keypoints(self.real_image1, batch_size)
        self.real_keypoints2 = self.extract_keypoints(self.real_image2, batch_size)
        self.real_keypoints3 = self.extract_keypoints(self.real_image3, batch_size)
        self.real_keypoints4 = self.extract_keypoints(self.real_image4, batch_size)
        self.fake_keypoints1_1 = self.extract_keypoints(self.fake_image1_1, batch_size)
        self.fake_keypoints1_2 = self.extract_keypoints(self.fake_image1_2, batch_size)
        self.fake_keypoints1_3 = self.extract_keypoints(self.fake_image1_3, batch_size)
        self.fake_keypoints1_4 = self.extract_keypoints(self.fake_image1_4, batch_size)
        self.fake_keypoints2_1 = self.extract_keypoints(self.fake_image2_1, batch_size)
        self.fake_keypoints2_2 = self.extract_keypoints(self.fake_image2_2, batch_size)
        self.fake_keypoints2_3 = self.extract_keypoints(self.fake_image2_3, batch_size)
        self.fake_keypoints2_4 = self.extract_keypoints(self.fake_image2_4, batch_size)

        self.prev_fake_keypoints1_1[0:(batch_size - 1)] = self.fake_keypoints1_1[0:(batch_size - 1)] 
        self.prev_fake_keypoints1_2[0:(batch_size - 1)] = self.fake_keypoints1_2[0:(batch_size - 1)] 
        self.prev_fake_keypoints1_3[0:(batch_size - 1)] = self.fake_keypoints1_3[0:(batch_size - 1)] 
        self.prev_fake_keypoints1_4[0:(batch_size - 1)] = self.fake_keypoints1_4[0:(batch_size - 1)] 
        self.prev_fake_keypoints2_1[0:(batch_size - 1)] = self.fake_keypoints2_1[0:(batch_size - 1)]
        self.prev_fake_keypoints2_2[0:(batch_size - 1)] = self.fake_keypoints2_2[0:(batch_size - 1)]
        self.prev_fake_keypoints2_3[0:(batch_size - 1)] = self.fake_keypoints2_3[0:(batch_size - 1)]
        self.prev_fake_keypoints2_4[0:(batch_size - 1)] = self.fake_keypoints2_4[0:(batch_size - 1)]

        tmp_array1_1 = self.prev_fake_keypoints1_1
        tmp_array1_2 = self.prev_fake_keypoints1_2
        tmp_array1_3 = self.prev_fake_keypoints1_3
        tmp_array1_4 = self.prev_fake_keypoints1_4
        tmp_array2_1 = self.prev_fake_keypoints2_1
        tmp_array2_2 = self.prev_fake_keypoints2_2
        tmp_array2_3 = self.prev_fake_keypoints2_3
        tmp_array2_4 = self.prev_fake_keypoints2_4
        self.prev_fake_keypoints1_1[1:] = tmp_array1_1[0:(batch_size - 1)]
        self.prev_fake_keypoints1_2[1:] = tmp_array1_2[0:(batch_size - 1)]
        self.prev_fake_keypoints1_3[1:] = tmp_array1_3[0:(batch_size - 1)]
        self.prev_fake_keypoints1_4[1:] = tmp_array1_4[0:(batch_size - 1)]
        self.prev_fake_keypoints2_1[1:] = tmp_array2_1[0:(batch_size - 1)]
        self.prev_fake_keypoints2_2[1:] = tmp_array2_2[0:(batch_size - 1)]
        self.prev_fake_keypoints2_3[1:] = tmp_array2_3[0:(batch_size - 1)]
        self.prev_fake_keypoints2_4[1:] = tmp_array2_4[0:(batch_size - 1)]
        self.prev_fake_keypoints1_1[0] = tmp_array1_1[batch_size - 1]
        self.prev_fake_keypoints1_2[0] = tmp_array1_2[batch_size - 1]
        self.prev_fake_keypoints1_3[0] = tmp_array1_3[batch_size - 1]
        self.prev_fake_keypoints1_4[0] = tmp_array1_4[batch_size - 1]
        self.prev_fake_keypoints2_1[0] = tmp_array2_1[batch_size - 1]
        self.prev_fake_keypoints2_2[0] = tmp_array2_2[batch_size - 1]
        self.prev_fake_keypoints2_3[0] = tmp_array2_3[batch_size - 1]
        self.prev_fake_keypoints2_4[0] = tmp_array2_4[batch_size - 1]

        self.loss_G_coherence1 = (self.criterionOP(self.fake_keypoints1_1, self.prev_fake_keypoints1_2) + self.criterionOP(self.fake_keypoints1_2, self.prev_fake_keypoints1_3) + self.criterionOP(self.fake_keypoints1_3, self.prev_fake_keypoints1_4)) / 3
        self.loss_G_coherence = (self.criterionOP(self.fake_keypoints2_1, self.prev_fake_keypoints2_2) + self.criterionOP(self.fake_keypoints2_2, self.prev_fake_keypoints2_3) + self.criterionOP(self.fake_keypoints2_3, self.prev_fake_keypoints2_4)) / 3

        self.loss_G_coherence = (self.loss_G_coherence + self.loss_G_coherence1) * 0.01"""



        self.loss_G_correction1 = (self.criterionOP(self.real_keypoints1, self.fake_keypoints1_1) + self.criterionOP(self.real_keypoints2, self.fake_keypoints1_2) + self.criterionOP(self.real_keypoints3, self.fake_keypoints1_3) + self.criterionOP(self.real_keypoints4, self.fake_keypoints1_4)) / 4
        self.loss_G_correction = (self.criterionOP(self.real_keypoints1, self.fake_keypoints2_1) + self.criterionOP(self.real_keypoints2, self.fake_keypoints2_2) + self.criterionOP(self.real_keypoints3, self.fake_keypoints2_3) + self.criterionOP(self.real_keypoints4, self.fake_keypoints2_4)) / 4

        #self.loss_G_correction = abs((self.loss_G_correction + self.loss_G_correction1) * self.weight_G_correction)
        self.loss_G_correction = (self.loss_G_correction + self.loss_G_correction1) * weight

        self.loss_G_L1 = self.loss_G_L1_1 + loss_G_L2_1
        self.loss_G_GAN = loss_G_GAN1_1 + loss_G_GAN2_1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_correction + self.loss_G_coherence
        self.loss_G.backward(retain_graph=True)
        #print(self.weight_G_correction)
        """with open(output_file_path, "a") as f:  # "a"は追記モードを示します
            f.write(f"weight_G_coherence:[{self.weight_G_coherence}]\n")
            f.write(f"weight_G_correction:[{self.weight_G_correction}]\n")"""

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()  # first call forward to calculate intermediate results
        self.set_requires_grad(self.netC2, False)
        # update D
        self.set_requires_grad(self.netD_i, True)  # enable backprop for D
        self.optimizer_D_i.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D_i.step()  # update D's weights

        self.set_requires_grad(self.netD_v, True)  # enable backprop for D
        self.optimizer_D_v.zero_grad()  # set D's gradients to zero
        self.backward_D_v()  # calculate gradients for D
        self.optimizer_D_v.step()  # update D's weights

        # update G
        self.set_requires_grad(self.netD_i, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD_v, False)
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
    
    def save_image(self, image_numpy):
        """im = image_numpy.astype(np.uint8).copy()
        self.real_image_canvas = copy.deepcopy(im)
        self.real_image_canvas = util.draw_bodypose(im, self.real_iamge_candidate, self.real_image_subset)"""
        self.real_image_canvas_pil = Image.fromarray(self.real_image_canvas)
        self.real_image_canvas_pil.save("result/testtesttest.png")

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
        candidate, subset = self.netBody(im)

        keypoints = np.full((18,2), -1)

        j = 0
        if subset.shape == (0, 20):
            subset = np.full((1,20), -1)
        for i in range(18):
            if not subset[0][i] == -1:
                keypoints[i][0] = candidate[j][0]
                keypoints[i][1] = candidate[j][1]
                j += 1

        """for i in range(18):
            if subset[0, i] == -1:
                candidate = np.insert(candidate, i, [-1, -1, -1, -1], axis=0)
        keypoints = candidate[:, :2]"""

        """if not subset == [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]:
            hands_list = util.handDetect(candidate, subset, im)
            all_hand_peaks = []
            if not hands_list == []:
                for x, y, w, is_lef in hands_list:
                    peaks = self.netHand(im[y:y+w, x:x+w, :])
                    peaks[:, 0] = np.where(peaks[:, 0]==0, -1, peaks[:, 0]+x)
                    peaks[:, 1] = np.where(peaks[:, 1]==0, -1, peaks[:, 1]+y)
                    
                all_hand_peaks.append(peaks)
                all_hand_peaks = np.array(all_hand_peaks)
                all_hand_peaks = all_hand_peaks.reshape(21,2)
                all_hand_peaks = all_hand_peaks.tolist()
            else:
                all_hand_peaks = [[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
        else:
            all_hand_peaks = [[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
        
        keypoints += all_hand_peaks
        num_extract = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(39):
            if not keypoints[i] == [-1, -1]:
                num_extract[i] += 1
        
        canvas = util.draw_handpose(canvas, all_hand_peaks)
        self.save_image(canvas)
        save_image(a)"""
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
