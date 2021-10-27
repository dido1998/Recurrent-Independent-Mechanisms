import torch.distributions as dist
import torch

def main():
    probs = torch.rand(5,2)
    dist_A = dist.bernoulli.Bernoulli(probs=probs[:,0])
    dist_B = dist.bernoulli.Bernoulli(probs=probs[:,1])
    KL_AB = dist.kl.kl_divergence(dist_A,dist_B)
    KL_defined_AB = KL_bern(probs[0,0], probs[0,1])
    print(f'torch: {KL_AB}, defined: {KL_defined_AB}')

def KL_bern(p, q):
    kl = p*(p/q).log() + (1-p)*((1-p)/(1-q)).log()
    return kl

if __name__ == "__main__":
    main()