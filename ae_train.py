import argparse
import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, L1Loss, MSELoss, BCELoss
from torch.utils.data import DataLoader

from nn_img_compress.dataset.img_dataset import ImageFolderDataset
from nn_img_compress.models.state_model import StateDAE


def custom_loss(outputs, labels):
    n = torch.abs(outputs - labels)
    loss = n * torch.exp(n)
    return loss


bce = BCELoss()

cos_sim = torch.nn.CosineSimilarity()
mse_loss = MSELoss()

if __name__ == '__main__':
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    default_probe_path = os.path.join(os.path.dirname(__file__), "config/probe.csv")
    parser = argparse.ArgumentParser(description="Trainer for state AutoEncoder model.")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each sample batch")
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")
    parser.add_argument("--probe_path", type=str, default=default_probe_path, help="probe path")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    opt = parser.parse_args()

    epochs = opt.epochs
    batch_size = opt.batch_size
    dataset_path = opt.dataset_path
    probe_path = opt.probe_path

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = StateDAE(coor_size=2, feat_size=3, out_size=3,
                     n_sep=4, n_probe=64,
                     n_enc_layers=3,
                     n_dec_layers=1,
                     drop_p=0.0,
                     dropatt=0.0,
                     d_inner=256, d_model=128, pre_lnorm=True,
                     n_head=8, d_head=16)
    model.to(device)

    if opt.pretrained_weights:
        model.load_state_dict(torch.load(opt.pretrained_weights, map_location=device))

    train_ds = ImageFolderDataset(dataset_path, n_visible=3600, n_inference=25000, is_train=True)
    valid_ds = ImageFolderDataset(dataset_path, n_visible=3600, n_inference=25000, is_train=False)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=opt.n_cpu)
    valid_dataloader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, num_workers=opt.n_cpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    for epoch in range(epochs):
        model.train()

        outputs = None
        t_feats = None

        for i, (coor, feats, t_coor, t_feats, h, w, idx) in enumerate(train_dataloader):
            optimizer.zero_grad()

            coor = coor.to(device)
            feats = feats.to(device)
            t_coor = t_coor.to(device)
            t_feats = t_feats.to(device)

            outputs = model(coor, feats, t_coor)
            outputs = torch.sigmoid(outputs)
            loss = bce(outputs, t_feats)

            log_str = "---- [Epoch %d/%d, Step %d/%d] loss: %.6f ----" % (
                epoch, epochs, i, len(train_dataloader), loss.item())
            logging.info(log_str)

            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, (coor, feats, t_coor, t_feats, h, w, idx) in enumerate(valid_dataloader):
                coor = coor.to(device)
                feats = feats.to(device)
                t_coor = t_coor.to(device)
                t_feats = t_feats.to(device)

                outputs = model(coor, feats, t_coor)
                outputs = torch.sigmoid(outputs)
                loss = bce(outputs, t_feats)

                total_loss += loss.item()
            total_loss /= len(valid_dataloader)

            log_str = "---- [Epoch %d ] eval loss: %.6f ----\n" % (
                epoch, total_loss)

            logging.info(log_str)

            if outputs is not None:
                t_feats = t_feats.cpu().detach().numpy() * 255
                outputs = outputs.cpu().detach().numpy() * 255

                image = np.zeros((int(h[0]), int(w[0]), 3), dtype=np.uint8)
                t_coor = t_coor[0].cpu()
                image[(t_coor[:, 0] * h[0]).int(), (t_coor[:, 1] * w[0]).int(), :] = outputs[0].astype(np.uint8)


                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # plt.savefig(f"output/ae_ckpt_%d_%.6f.png" % (epoch, total_loss))
                plt.show()
                plt.clf()
                plt.imshow(cv2.cvtColor(cv2.imread(valid_ds.img_paths[int(idx[0])]), cv2.COLOR_BGR2RGB))
                plt.show()
                plt.clf()
                plt.close()

        torch.save(model.state_dict(),
                   f"checkpoints/ae_ckpt_%d_%.6f.pth" % (epoch, total_loss))
