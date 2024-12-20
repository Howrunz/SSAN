import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SSAN import SSAN
from utils import gen_counting_label


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growthRate)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1(F.relu(x, inplace=True))
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, use_dropout):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nOutChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out

class DenseNet(nn.Module):
    def __init__(self, params):
        super(DenseNet, self).__init__()
        growthRate = params['densenet']['growthRate']
        reduction = params['densenet']['reduction']
        bottleneck = params['densenet']['bottleneck']
        use_dropout = params['densenet']['use_dropout']

        nDenseBlocks = 16
        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(params['encoder']['input_channel'], nChannels, kernel_size=7, padding=3, stride=2, bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels, use_dropout)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels, use_dropout)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, use_dropout))
            else:
                layers.append(SingleLayer(nChannels, growthRate, use_dropout))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, 2, ceil_mode=True)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        return out

class Attention(nn.Module):
    def __init__(self, params):
        super(Attention, self).__init__()
        self.params = params
        self.hidden = params['decoder']['hidden_size']
        self.attention_dim = params['attention']['attention_dim']
        self.hidden_weight = nn.Linear(self.hidden, self.attention_dim)
        self.attention_conv = nn.Conv2d(1, 512, kernel_size=11, padding=5, bias=False)
        self.attention_weight = nn.Linear(512, self.attention_dim, bias=False)
        self.alpha_convert = nn.Linear(self.attention_dim, 1)

    def forward(self, cnn_features, cnn_features_trans, hidden, alpha_sum, image_mask=None):
        query = self.hidden_weight(hidden)
        alpha_sum_trans = self.attention_conv(alpha_sum)
        coverage_alpha = self.attention_weight(alpha_sum_trans.permute(0,2,3,1))
        alpha_score = torch.tanh(query[:, None, None, :] + coverage_alpha + cnn_features_trans.permute(0,2,3,1))
        energy = self.alpha_convert(alpha_score)
        energy = energy - energy.max()
        energy_exp = torch.exp(energy.squeeze(-1))
        if image_mask is not None:
            energy_exp = energy_exp * image_mask.squeeze(1)
        alpha = energy_exp / (energy_exp.sum(-1).sum(-1)[:,None,None] + 1e-10)
        alpha_mask = alpha > 2e-2
        alpha_sum = alpha[:,None,:,:] + alpha_sum
        context_vector = (alpha[:,None,:,:] * alpha_mask[:,None,:,:] * cnn_features).sum(-1).sum(-1)
        return context_vector, alpha, alpha_sum


class PositionEmbeddingSine(nn.Module):

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class AttDecoder(nn.Module):
    def __init__(self, params):
        super(AttDecoder, self).__init__()
        self.params = params
        self.input_size = params['decoder']['input_size']
        self.hidden_size = params['decoder']['hidden_size']
        self.out_channel = params['encoder']['out_channel']
        self.attention_dim = params['attention']['attention_dim']
        self.dropout_prob = params['dropout']
        self.device = params['local_rank']
        self.word_num = params['alphabet_num']
        self.counting_num = params['counting_decoder']['out_channel']

        self.ratio = params['densenet']['ratio']

        self.init_weight = nn.Linear(self.out_channel, self.hidden_size)
        self.embedding = nn.Embedding(self.word_num, self.input_size)
        self.word_input_gru = nn.GRUCell(self.input_size, self.hidden_size)
        self.word_attention = Attention(params)
        self.encoder_feature_conv = nn.Conv2d(self.out_channel, self.attention_dim,
                                              kernel_size=params['attention']['word_conv_kernel'],
                                              padding=params['attention']['word_conv_kernel'] // 2)

        self.word_state_weight = nn.Linear(self.hidden_size, self.hidden_size)
        self.word_embedding_weight = nn.Linear(self.input_size, self.hidden_size)
        self.word_context_weight = nn.Linear(self.out_channel, self.hidden_size)
        self.counting_context_weight = nn.Linear(self.counting_num, self.hidden_size)
        self.localization_context_weight = nn.Linear(432, self.hidden_size)
        self.word_convert = nn.Linear(self.hidden_size, self.word_num)

        if params['dropout']:
            self.dropout = nn.Dropout(params['dropout_ratio'])

    def forward(self, cnn_features, labels, counting_preds, localization_pred, images_mask, labels_mask, is_train=True):
        batch_size, num_steps = labels.shape
        height, width = cnn_features.shape[2:]
        word_probs = torch.zeros((batch_size, num_steps, self.word_num)).to(device=self.device)
        images_mask = images_mask[:, :, ::self.ratio, ::self.ratio]

        word_alpha_sum = torch.zeros((batch_size, 1, height, width)).to(device=self.device)
        word_alphas = torch.zeros((batch_size, num_steps, height, width)).to(device=self.device)
        hidden = self.init_hidden(cnn_features, images_mask)
        localization_weighted = self.localization_context_weight(localization_pred)
        counting_context_weighted = self.counting_context_weight(counting_preds)

        cnn_features_trans = self.encoder_feature_conv(cnn_features)
        position_embedding = PositionEmbeddingSine(256, normalize=True)
        pos = position_embedding(cnn_features_trans, images_mask[:, 0, :, :])
        cnn_features_trans = cnn_features_trans + pos

        if is_train:
            for i in range(num_steps):
                word_embedding = self.embedding(labels[:, i - 1]) if i else self.embedding(
                    torch.ones([batch_size]).long().to(self.device))
                hidden = self.word_input_gru(word_embedding, hidden)
                word_context_vec, word_alpha, word_alpha_sum = self.word_attention(cnn_features, cnn_features_trans,
                                                                                   hidden,
                                                                                   word_alpha_sum, images_mask)

                current_state = self.word_state_weight(hidden)
                word_weighted_embedding = self.word_embedding_weight(word_embedding)
                word_context_weighted = self.word_context_weight(word_context_vec)

                if self.params['dropout']:
                    word_out_state = self.dropout(torch.maximum(
                        (current_state + word_weighted_embedding + word_context_weighted),
                        localization_weighted) + counting_context_weighted)
                else:
                    word_out_state = torch.maximum((current_state + word_weighted_embedding + word_context_weighted),
                                                   localization_weighted) + counting_context_weighted

                word_prob = self.word_convert(word_out_state)
                word_probs[:, i] = word_prob
                word_alphas[:, i] = word_alpha
        else:
            word_embedding = self.embedding(torch.ones([batch_size]).long().to(device=self.device))
            for i in range(num_steps):
                hidden = self.word_input_gru(word_embedding, hidden)
                word_context_vec, word_alpha, word_alpha_sum = self.word_attention(cnn_features, cnn_features_trans,
                                                                                   hidden,
                                                                                   word_alpha_sum, images_mask)

                current_state = self.word_state_weight(hidden)
                word_weighted_embedding = self.word_embedding_weight(word_embedding)
                word_context_weighted = self.word_context_weight(word_context_vec)

                if self.params['dropout']:
                    word_out_state = self.dropout(torch.maximum(
                        (current_state + word_weighted_embedding + word_context_weighted),
                        localization_weighted) + counting_context_weighted)
                else:
                    word_out_state = torch.maximum((current_state + word_weighted_embedding + word_context_weighted),
                                                   localization_weighted) + counting_context_weighted

                word_prob = self.word_convert(word_out_state)
                _, word = word_prob.max(1)
                word_embedding = self.embedding(word)
                word_probs[:, i] = word_prob
                word_alphas[:, i] = word_alpha
        return word_probs, word_alphas

    def init_hidden(self, features, feature_mask):
        average = (features * feature_mask).sum(-1).sum(-1) / feature_mask.sum(-1).sum(-1)
        average = self.init_weight(average)
        return torch.tanh(average)

class ChannelAtt(nn.Module):
    def __init__(self, channel, reduction):
        super(ChannelAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel//reduction),
                nn.ReLU(),
                nn.Linear(channel//reduction, channel),
                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class CountingDecoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(CountingDecoder, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.trans_layer = nn.Sequential(
            nn.Conv2d(self.in_channel, 512, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(512))
        self.channel_att = ChannelAtt(512, 16)
        self.pred_layer = nn.Sequential(
            nn.Conv2d(512, self.out_channel, kernel_size=1, bias=False),
            nn.Sigmoid())

    def forward(self, x, mask):
        b, c, h, w = x.size()
        x = self.trans_layer(x)
        x = self.channel_att(x)
        x = self.pred_layer(x)
        if mask is not None:
            x = x * mask
        x = x.view(b, self.out_channel, -1)
        x1 = torch.sum(x, dim=-1)
        return x1, x.view(b, self.out_channel, h, w)

class SSAN_CAN_DWAP(nn.Module):
    def __init__(self, params=None):
        super(SSAN_CAN_DWAP, self).__init__()
        print('SSAN_CAN_DWAP')
        self.params = params
        self.use_label_mask = params['use_label_mask']
        self.in_channel = params['counting_decoder']['in_channel']
        self.out_channel = params['counting_decoder']['out_channel']
        self.encoder = DenseNet(params=self.params)

        self.localization_decoder = SSAN(in_ch=[684, 600, 432], out_ch=1, layers_num=[1, 1], drop_rate=0.5)
        self.counting_decoder1 = CountingDecoder(self.in_channel, self.out_channel, 3)
        self.counting_decoder2 = CountingDecoder(self.in_channel, self.out_channel, 5)
        self.decoder = AttDecoder(params=self.params)
        self.cross = nn.CrossEntropyLoss(reduction='none') if self.use_label_mask else nn.CrossEntropyLoss()
        self.counting_loss = nn.SmoothL1Loss(reduction='mean')
        self.spatial_loss = nn.SmoothL1Loss(reduction='mean')

        self.ratio = params['densenet']['ratio']

    def forward(self, images, images_mask, labels, labels_mask, auxiliary, is_train=True):
        cnn_features = self.encoder(images)

        counting_mask = images_mask[:, :, ::self.ratio, ::self.ratio]
        counting_labels = gen_counting_label(labels, self.out_channel, True)
        local_fea, local_pred = self.localization_decoder(cnn_features)
        counting_preds1, _ = self.counting_decoder1(cnn_features, counting_mask)
        counting_preds2, _ = self.counting_decoder2(cnn_features, counting_mask)
        counting_preds = (counting_preds1 + counting_preds2) / 2
        counting_loss = self.counting_loss(counting_preds1, counting_labels) + self.counting_loss(counting_preds2, counting_labels) \
                        + self.counting_loss(counting_preds, counting_labels)
        spatial_loss = self.spatial_loss(local_pred * images_mask[..., ::self.ratio*2, ::self.ratio*2], auxiliary)

        local_fea = (local_fea * images_mask[..., ::self.ratio * 2, ::self.ratio * 2]).sum(-1).sum(-1)
        word_probs, word_alphas = self.decoder(cnn_features, labels, counting_preds, local_fea, images_mask, labels_mask, is_train=is_train)
        word_loss = self.cross((word_probs.contiguous().view(-1, word_probs.shape[-1])), labels.view(-1))
        word_average_loss = (word_loss * labels_mask.view(-1)).sum() / (labels_mask.sum() + 1e-10) if self.use_label_mask else word_loss
        return word_probs, [word_average_loss, counting_loss, spatial_loss], word_alphas
