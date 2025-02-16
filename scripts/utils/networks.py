import torch
import torch.nn as nn
import torch.nn.functional as F

####################################################
#            Contrastive SiameseNN                 #
####################################################
class ContrastiveSiameseNN(nn.Module):
    def __init__(self, input_dim):
        super(ContrastiveSiameseNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Combine out1, out2, |diff|, product, exp(-|diff|) => 5 * 32 = 160
        self.classifier = nn.Sequential(
            nn.Linear(32 * 5, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Linear(32, 1)
        )

    def forward_once(self, x):
        return self.feature_extractor(x)

    def forward(self, emb1, emb2, labels=None):
        out1 = self.forward_once(emb1)
        out2 = self.forward_once(emb2)

        abs_diff = torch.abs(out1 - out2)
        prod     = out1 * out2
        sim      = torch.exp(-abs_diff)

        combined = torch.cat([out1, out2, abs_diff, prod, sim], dim=1)
        logits   = self.classifier(combined)

        outputs = {'logits': logits}
        if labels is not None:
            labels = labels.float().unsqueeze(1)
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            outputs['loss'] = loss
        return outputs

####################################################
#            Triplet SiameseNN                     #
####################################################
class TripletSiameseNN(nn.Module):
    def __init__(self, input_dim):
        super(TripletSiameseNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU()
        )

    def forward_once(self, x):
        return self.feature_extractor(x)

    def forward(self, anchor, positive, negative):
        anchor_out = self.forward_once(anchor)
        pos_out = self.forward_once(positive)
        neg_out = self.forward_once(negative)
        return anchor_out, pos_out, neg_out


####################################################
#            Loss Functions                        #
####################################################
class TripletLoss(nn.Module):
    """
    Margin-based triplet loss:
      L = mean( max(0, d(anchor, pos)^2 - d(anchor, neg)^2 + margin ) ).
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_out, pos_out, neg_out):
        dist_pos = F.pairwise_distance(anchor_out, pos_out)
        dist_neg = F.pairwise_distance(anchor_out, neg_out)

        loss = dist_pos.pow(2) - dist_neg.pow(2) + self.margin
        loss = torch.clamp(loss, min=0.0).mean()
        return loss
