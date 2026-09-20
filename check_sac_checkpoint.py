import torch
ckpt = torch.load('models/sac_agent.pt', map_location='cpu')
print('keys:', list(ckpt.keys()))
if 'actor' in ckpt:
    s = ckpt['actor']
    print('\nActor param shapes:')
    for k,v in s.items():
        try:
            print(k, tuple(v.shape))
        except:
            print(k, type(v))
else:
    print('actor not in checkpoint')

if 'critic' in ckpt:
    s = ckpt['critic']
    print('\nCritic param shapes:')
    for k,v in s.items():
        try:
            print(k, tuple(v.shape))
        except:
            print(k, type(v))
