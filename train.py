from PIA import PIA
import torch
import numpy as np
from utils import get_batch
import argparse


parser = argparse.ArgumentParser(description='Training PIA for diffusion-relaxation model')
parser.add_argument('--predictor_depth', type=int, default=2, help='Depth of the predictor network')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=500000, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model =  PIA(predictor_depth=args.predictor_depth, device=device)
    params = list(model.encoder.parameters()) + list(model.v_predictor.parameters()) + list(model.D_predictor.parameters()) + list(model.T2_predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    ctr = 1
    total_loss = 0.0

    for ep in range(args.num_epochs):
        x, _, _, _, y = get_batch(args.batch_size)
        x , y = x.to(device), y.to(device)
        optimizer.zero_grad()
        D, T2, v = model.encode(x)        
        recon = model.decode(D, T2, v)
        loss = model.loss_function(recon, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if not ep % 10:
            print(f'{(total_loss/ctr):.2f}',end ="\r")
        ctr += 1

    PATH = 'pia_model2.pt'
    torch.save(model, PATH)
    print('Done')

if __name__ == '__main__':

    args = parser.parse_args()
    main(args)
