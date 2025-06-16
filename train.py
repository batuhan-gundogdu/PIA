"""
PIA Non-commercial License
© 2023-2025 The University of Chicago.
Author: Batuhan Gundogdu


Redistribution and use for noncommercial purposes in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. The software is used solely for noncommercial purposes. It may not be used indirectly for commercial use, such as on a website that accepts advertising money for content. Noncommercial use does include use by a for-profit company in its research. For commercial use rights, contact The University of Chicago, Polsky Center for Entrepreneurship, and Innovation, at polskylicensing@uchicago.edu or call 773-702-1692 and inquire about Tech ID XX-X-XXX project.
2. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
3. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
4. Neither the name of The University of Chicago nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY OF CHICAGO AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF CHICAGO OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
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
