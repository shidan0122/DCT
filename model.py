import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from itertools import chain
from data import load_dataset
from module import Feature_Extractor, Encoder_Decoder, Classifier, Feature_Extractor_IMG
from util import get_graph, to_one_hot, fx_calc_map_label, rbf_affnty, labels_affnty

class Model(object):
    def __init__(self, args):
        self.dataset = args.dataset
        self.dual_ratio = args.dual
        self.only_image_ratio = args.oi
        self.only_text_ratio = 1.0 - args.dual - args.oi
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self._setting(args)
        self._load_data()
        self._build_model()
        self.param_loss_recon = args.param_loss_recon
        self.param_loss_cls_img = args.param_loss_cls_img
        self.param_loss_cls_txt = args.param_loss_cls_txt
        self.param_loss_cross = args.param_loss_cross
        self.param_loss_sim = args.param_loss_sim
        self.param_loss_sim_img = args.param_loss_sim_img
        self.param_loss_sim_txt = args.param_loss_sim_txt
        self.param_loss_sim_it = args.param_loss_sim_it

    def _setting(self, args):
        self.nhead = args.num_heads
        self.nhead_out = args.num_out_heads
        self.encoder_type = args.encoder
        self.decoder_type = args.decoder
        self.activation = args.activation
        self.feat_drop = args.in_drop
        self.attn_drop = args.attn_drop
        self.negative_slope = args.negative_slope
        self.residual = args.residual
        self.norm = args.norm
        self.num_hidden = args.num_hidden
        self.in_dim = 1024
        self.num_layers = args.num_layers
        self.dec_in_dim = self.num_hidden
        self.dec_num_hidden = self.num_hidden // self.nhead_out if self.decoder_type in (
        "gat", "dotgat") else self.num_hidden
        if self.encoder_type in ("gat", "dotgat"):
            self.enc_num_hidden = self.num_hidden // self.nhead
            self.enc_nhead = self.nhead
        else:
            self.enc_num_hidden = self.num_hidden
            self.enc_nhead = 1

    def _load_data(self):
        self.data = load_dataset(self.dataset)
        self.train_size = self.data['train_label'].shape[0]
        self.classes = np.unique(self.data['train_label'])
        self.num_classes = len(self.classes)
        dual_size = math.ceil(self.train_size * self.dual_ratio)
        only_image_size = math.ceil(self.train_size * self.only_image_ratio)
        only_text_size = math.floor(self.train_size * self.only_text_ratio)
        index_train = np.arange(self.train_size).tolist()
        np.random.shuffle(index_train)
        self.dual_index = index_train[:dual_size]
        self.only_image_index = index_train[dual_size:dual_size + only_image_size]
        self.only_txet_index = index_train[dual_size + only_image_size:only_text_size + dual_size + only_image_size]
        self.batch_dual_size = math.ceil(self.batch_size * self.dual_ratio)
        self.batch_only_image_size = math.ceil(self.batch_size * self.only_image_ratio)
        self.batch_only_text_size = self.batch_size - self.batch_dual_size - self.batch_only_image_size

    def _build_model(self):
        self.feature_extractor = {}
        self.feature_extractor['img'] = Feature_Extractor(1024, 1024).cuda()
        self.feature_extractor['txt'] = Feature_Extractor(1024, 1024).cuda()
        self.encoder = {}
        self.encoder['img'] = Encoder_Decoder(m_type=self.encoder_type, enc_dec="encoding", in_dim=self.in_dim,
                                              num_hidden=self.enc_num_hidden, out_dim=self.enc_num_hidden,
                                              num_layers=self.num_layers, nhead=self.enc_nhead,
                                              nhead_out=self.enc_nhead, concat_out=True, activation=self.activation,
                                              dropout=self.feat_drop, attn_drop=self.attn_drop,
                                              negative_slope=self.negative_slope, residual=self.residual,
                                              norm=self.norm).cuda()
        self.encoder['txt'] = Encoder_Decoder(m_type=self.encoder_type, enc_dec="encoding", in_dim=self.in_dim,
                                              num_hidden=self.enc_num_hidden, out_dim=self.enc_num_hidden,
                                              num_layers=self.num_layers, nhead=self.enc_nhead,
                                              nhead_out=self.enc_nhead, concat_out=True, activation=self.activation,
                                              dropout=self.feat_drop, attn_drop=self.attn_drop,
                                              negative_slope=self.negative_slope, residual=self.residual,
                                              norm=self.norm).cuda()
        self.encoder_to_decoder = {}
        self.encoder_to_decoder['img'] = nn.Linear(self.dec_in_dim, self.dec_in_dim, bias=False).cuda()
        self.encoder_to_decoder['txt'] = nn.Linear(self.dec_in_dim, self.dec_in_dim, bias=False).cuda()
        self.decoder = {}
        self.decoder['img'] = Encoder_Decoder(m_type=self.decoder_type, enc_dec="decoding", in_dim=self.dec_in_dim,
                                              num_hidden=self.dec_num_hidden, out_dim=self.in_dim, num_layers=1,
                                              nhead=self.nhead,
                                              nhead_out=self.nhead_out, activation=self.activation,
                                              dropout=self.feat_drop, attn_drop=self.attn_drop,
                                              negative_slope=self.negative_slope, residual=self.residual,
                                              norm=self.norm, concat_out=True).cuda()
        self.decoder['txt'] = Encoder_Decoder(m_type=self.decoder_type, enc_dec="decoding", in_dim=self.dec_in_dim,
                                              num_hidden=self.dec_num_hidden, out_dim=self.in_dim, num_layers=1,
                                              nhead=self.nhead,
                                              nhead_out=self.nhead_out, activation=self.activation,
                                              dropout=self.feat_drop, attn_drop=self.attn_drop,
                                              negative_slope=self.negative_slope, residual=self.residual,
                                              norm=self.norm, concat_out=True).cuda()
        self.classifier = {}
        self.classifier['img'] = Classifier(self.num_hidden, self.num_classes).cuda()
        self.classifier['txt'] = Classifier(self.num_hidden, self.num_classes).cuda()

    def Label_Regression_Loss(self, view1_predict, label_onehot):
        loss = ((view1_predict - label_onehot.float()) ** 2).sum(1).sqrt().mean()
        return loss

    def sce_loss(self, x, y, alpha=3):  # 1,2,3,4
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        loss = loss.mean()
        return loss

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def EntropicConfusion(self, z):
        softmax_out = nn.Softmax(dim=1)(z)
        batch_size = z.size(0)
        loss = torch.mul(softmax_out, torch.log(softmax_out)).sum() * (1.0 / batch_size)
        return loss

    def train(self):
        self.optimizer_feature_extractor_image = optim.Adam(self.feature_extractor['img'].parameters(), lr=1e-3)
        self.optimizer_feature_extractor_text = optim.Adam(self.feature_extractor['txt'].parameters(), lr=1e-3)
        self.optimizer_encoder_image = optim.Adam(self.encoder['img'].parameters(), lr=0.001)
        self.optimizer_encoder_text = optim.Adam(self.encoder['txt'].parameters(), lr=0.001)
        self.optimizer_decoder_image = optim.Adam(chain(*[self.encoder_to_decoder['img'].parameters(), self.decoder['img'].parameters()]), lr=0.001)
        self.optimizer_decoder_text = optim.Adam(chain(*[self.encoder_to_decoder['txt'].parameters(), self.decoder['txt'].parameters()]), lr=0.001)
        self.optimizer_classifier_image = optim.Adam(self.classifier['img'].parameters(), lr=1e-3)
        self.optimizer_classifier_text = optim.Adam(self.classifier['txt'].parameters(), lr=1e-3)

        self.reconstruction_criterion = nn.L1Loss(size_average=False)
        # self.GANcriterion = nn.BCELoss()

        batches = int(math.floor(self.train_size / float(self.batch_size)))  # floor
        for epoch in range(self.epochs):
            loss_arr = 0.0
            np.random.shuffle(self.dual_index)
            np.random.shuffle(self.only_image_index)
            np.random.shuffle(self.only_txet_index)
            for batch in range(batches):
                batch_dual_index = self.dual_index[batch * self.batch_dual_size: (batch + 1) * self.batch_dual_size]
                batch_only_image_index = self.only_image_index[batch * self.batch_only_image_size: (batch + 1) * self.batch_only_image_size]
                batch_only_txet_index = self.only_txet_index[batch * self.batch_only_text_size: (batch + 1) * self.batch_only_text_size]

                batch_dual_images = torch.tensor(self.data['train_img'][batch_dual_index], dtype=torch.float32).cuda()
                batch_dual_texts = torch.tensor(self.data['train_txt'][batch_dual_index], dtype=torch.float32).cuda()
                batch_dual_labels = to_one_hot(self.data['train_label'][batch_dual_index], self.classes)

                batch_only_images = torch.tensor(self.data['train_img'][batch_only_image_index], dtype=torch.float32).cuda()
                self.data['train_txt'][batch_only_image_index] = 0
                batch_only_image_masks = torch.tensor(self.data['train_txt'][batch_only_image_index],dtype=torch.float32).cuda()
                batch_only_image_labels = to_one_hot(self.data['train_label'][batch_only_image_index], self.classes)

                batch_only_texts = torch.tensor(self.data['train_txt'][batch_only_txet_index], dtype=torch.float32).cuda()
                self.data['train_img'][batch_only_txet_index] = 0
                batch_only_text_masks = torch.tensor(self.data['train_img'][batch_only_txet_index], dtype=torch.float32).cuda()
                batch_only_text_labels = to_one_hot(self.data['train_label'][batch_only_txet_index], self.classes)

                batch_images = torch.cat([batch_dual_images, batch_only_images, batch_only_text_masks], dim=0).cuda()
                batch_texts = torch.cat([batch_dual_texts, batch_only_image_masks, batch_only_texts], dim=0).cuda()

                batch_labels = torch.cat([batch_dual_labels, batch_only_image_labels, batch_only_text_labels],dim=0).cuda()
                batch_label = torch.cat([torch.tensor(self.data['train_label'][batch_dual_index]),
                                         torch.tensor(self.data['train_label'][batch_only_image_index]),
                                         torch.tensor(self.data['train_label'][batch_only_txet_index])], dim=0).cuda()

                '''feature extractor'''
                '''batch_dual_image_features = self.feature_extractor['img'](batch_dual_images)
                batch_only_image_features = self.feature_extractor['img'](batch_only_images)
                batch_dual_texts_features = self.feature_extractor['txt'](batch_dual_texts)
                batch_only_texts_features = self.feature_extractor['txt'](batch_only_texts)

                #######xiugai
                batch_image_features = torch.cat([batch_dual_image_features, batch_only_image_features, batch_only_texts_features], dim=0)
                batch_text_features = torch.cat([batch_dual_texts_features, batch_only_image_features, batch_only_texts_features], dim=0)
                batch_images = torch.cat([batch_dual_image_features, batch_only_image_features, batch_only_text_masks], dim=0)
                batch_texts = torch.cat([batch_dual_texts_features, batch_only_image_masks, batch_only_text_masks], dim=0)

                #construct graph data
                image_graph = get_graph(batch_image_features, batch_images)
                text_graph = get_graph(batch_text_features, batch_texts)
                label_graph = labels_affnty(batch_label)'''

                batch_image = torch.cat([batch_dual_images, batch_only_images, batch_only_texts], dim=0).cuda()
                batch_text = torch.cat([batch_dual_texts, batch_only_images, batch_only_texts], dim=0).cuda()
                batch_image_features = self.feature_extractor['img'](batch_image)
                batch_text_features = self.feature_extractor['txt'](batch_text)
                image_graph = get_graph(batch_image_features, batch_image_features)
                text_graph = get_graph(batch_text_features, batch_text_features)
                label_graph = labels_affnty(batch_label)

                '''GAT encoder decoder'''
                encoder_image_features, _ = self.encoder['img'](image_graph, image_graph.ndata['feat'], return_hidden=True)
                mid_image_features = self.encoder_to_decoder['img'](encoder_image_features)
                decoder_image_features = self.decoder['img'](image_graph, mid_image_features)

                encoder_text_features, _ = self.encoder['txt'](text_graph, text_graph.ndata['feat'], return_hidden=True)
                mid_text_features = self.encoder_to_decoder['txt'](encoder_text_features)
                decoder_text_features = self.decoder['txt'](text_graph, mid_text_features)

                '''classifier'''
                predict_image_labels = self.classifier['img'](encoder_image_features)  # mid_image_features, encoder_image_features
                predict_text_labels = self.classifier['txt'](encoder_text_features)  # mid_text_features, encoder_text_features

                self.optimizer_feature_extractor_image.zero_grad()
                self.optimizer_feature_extractor_text.zero_grad()
                self.optimizer_encoder_image.zero_grad()
                self.optimizer_encoder_text.zero_grad()
                self.optimizer_decoder_image.zero_grad()
                self.optimizer_decoder_text.zero_grad()
                self.optimizer_classifier_image.zero_grad()
                self.optimizer_classifier_text.zero_grad()

                '''[ LOSS ]'''
                ''' LRL loss '''
                loss_classifier_image = self.Label_Regression_Loss(predict_image_labels, batch_labels)
                loss_classifier_text = self.Label_Regression_Loss(predict_text_labels, batch_labels)

                ''' reconstruction loss '''
                loss_reconstruction = self.sce_loss(decoder_image_features[:self.batch_dual_size + self.batch_only_image_size], batch_image_features[:self.batch_dual_size + self.batch_only_image_size]) \
                                      + self.sce_loss(decoder_text_features[:self.batch_dual_size], batch_text_features[:self.batch_dual_size]) \
                                      + self.sce_loss(decoder_text_features[-self.batch_only_text_size:], batch_text_features[-self.batch_only_text_size:])

                ''' cross-modal loss '''
                loss_cross = self.sce_loss(encoder_image_features, encoder_text_features)

                ''' similarity loss '''
                label_graph = label_graph.cuda()

                temp_img = F.normalize(batch_image_features)
                # temp_img_similarity = temp_img.mm(temp_img.t())
                temp_img_similarity = temp_img @ temp_img.T
                loss_mse_similarity_img = F.mse_loss(temp_img_similarity, label_graph)

                temp_txt = F.normalize(batch_text_features)
                temp_txt_similarity = temp_txt @ temp_txt.T
                loss_mse_similarity_txt = F.mse_loss(temp_txt_similarity, label_graph)

                loss_cross_similarity = F.mse_loss(temp_img_similarity, temp_txt_similarity)

                temp_img_z = F.normalize(encoder_image_features)
                temp_txt_z = F.normalize(encoder_text_features)
                temp_vt_similarity = temp_img_z @ temp_txt_z.T
                loss_cross_similarity_1 = F.mse_loss(temp_vt_similarity, label_graph)

                loss_mse_similarity = self.param_loss_sim_img * loss_mse_similarity_img + self.param_loss_sim_txt * loss_mse_similarity_txt + self.param_loss_sim * loss_cross_similarity + self.param_loss_sim_it * loss_cross_similarity_1
                # loss_mse_similarity = self.param_loss_sim_img * loss_mse_similarity_img + self.param_loss_sim_txt * loss_mse_similarity_txt + self.param_loss_sim_it * loss_cross_similarity_1

                ''' OVERALL LOSS'''
                loss = self.param_loss_cls_img * loss_classifier_image + self.param_loss_cls_txt * loss_classifier_text + self.param_loss_recon * loss_reconstruction + self.param_loss_cross * loss_cross + loss_mse_similarity
                #loss += loss_mse_similarity

                loss.backward()
                self.optimizer_feature_extractor_image.step()
                self.optimizer_feature_extractor_text.step()
                self.optimizer_encoder_image.step()
                self.optimizer_encoder_text.step()
                self.optimizer_decoder_image.step()
                self.optimizer_decoder_text.step()
                self.optimizer_classifier_image.step()
                self.optimizer_classifier_text.step()

                loss_arr += loss
            print(('[loss %4f]  [icls %4f]  [tcls %4f]  [rec %4f]  [cross %4f] [new %4f]') % (loss, loss_classifier_image, loss_classifier_text, loss_reconstruction, loss_cross, loss_cross_similarity_1))
            # print(('da %4f')%loss_mse_similarity)
            # self.test()

    def test(self):
        test_images = torch.tensor(self.data['test_img'], dtype=torch.float32).cuda()
        test_texts = torch.tensor(self.data['test_txt'], dtype=torch.float32).cuda()
        test_labels = torch.tensor(self.data['test_label']).cuda()

        self.feature_extractor['img'].eval()
        self.feature_extractor['txt'].eval()
        self.encoder['img'].eval()
        self.encoder['txt'].eval()

        with torch.no_grad():
            image_features = self.feature_extractor['img'](test_images)
            text_features = self.feature_extractor['txt'](test_texts)

            # testg = np.ones(image_features.shape)
            image_graph = get_graph(image_features, image_features)
            text_graph = get_graph(text_features, text_features)
            encoder_image_features, _ = self.encoder['img'](image_graph, image_graph.ndata['feat'], return_hidden=True)
            encoder_text_features, _ = self.encoder['txt'](text_graph, text_graph.ndata['feat'], return_hidden=True)
            # encoder_image_features = image_graph.ndata['feat']
            # encoder_text_features = text_graph.ndata['feat']

            test_image_features = encoder_image_features.cpu().numpy()
            test_text_features = encoder_text_features.cpu().numpy()

            # test_text_features = text_features.cpu().numpy()
            # test_image_features = image_features.cpu().numpy()
            test_labels = test_labels.cpu().numpy()

        i_map = fx_calc_map_label(test_image_features, test_text_features, test_labels)
        t_map = fx_calc_map_label(test_text_features, test_image_features, test_labels)
        print('Image to Text: MAP: {:.4f}'.format(i_map))
        print('Text to Image: MAP: {:.4f}'.format(t_map))
        ii_map = fx_calc_map_label(test_image_features, test_image_features, test_labels)
        tt_map = fx_calc_map_label(test_text_features, test_text_features, test_labels)
        print('Image to Image: MAP: {:.4f}'.format(ii_map))
        print('Text to Text: MAP: {:.4f}'.format(tt_map))
        Average_map = (i_map + t_map) / 2.
        print('Average Map: {:.4f}'.format(Average_map))

        self.feature_extractor['img'].train()
        self.feature_extractor['txt'].train()
        self.encoder['img'].train()
        self.encoder['txt'].train()
