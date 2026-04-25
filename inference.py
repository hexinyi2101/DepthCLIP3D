import torch
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    clip_type = ('depth',)
    model = LanguageBind(clip_type=clip_type, cache_dir='/home/hexinyi/LanguageBind-new/pretrained_model/')
    model = model.to(device)
    model.eval()
    pretrained_ckpt = f'/home/hexinyi/LanguageBind-new/pretrained_model/LanguageBind/LanguageBind_Depth'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt) #, cache_dir='./cache_dir/tokenizer_cache_dir')
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type}

 
    depth = ["/home/hexinyi/LanguageBind-new/downstream_datasets/Depth/shapenet12/data/val/rocket/04099429-1a3ef9b0c9c8ae6244f315d1e2c21ef_r_000_depth0001.png","/home/hexinyi/LanguageBind-new/downstream_datasets/Depth/shapenet12/data/val/rocket/04099429-1a3ef9b0c9c8ae6244f315d1e2c21ef_r_024_depth0001.png","/home/hexinyi/LanguageBind-new/downstream_datasets/Depth/shapenet12/data/val/rocket/04099429-1a3ef9b0c9c8ae6244f315d1e2c21ef_r_096_depth0001.png","/home/hexinyi/LanguageBind-new/downstream_datasets/Depth/shapenet12/data/val/rocket/04099429-1a3ef9b0c9c8ae6244f315d1e2c21ef_r_168_depth0001.png","/home/hexinyi/LanguageBind-new/downstream_datasets/Depth/shapenet12/data/val/rocket/04099429-1a3ef9b0c9c8ae6244f315d1e2c21ef_r_336_depth0001.png"]
    #thermal = ['assets/thermal/0.jpg', 'assets/thermal/1.jpg']
    language = ["airplane", "bench", "bicycle", "bottle", "bus", "car", "chair", "display", "lamp", "loudspeaker", "rifle", "sofa", "table", "telephone", "vessel"]


    inputs = {
        # 'image': to_device(modality_transform['image'](image), device),
        #'video': to_device(modality_transform['video'](video), device),
        #'audio': to_device(modality_transform['audio'](audio), device),
        'depth': to_device(modality_transform['depth'](depth), device),
        #'thermal': to_device(modality_transform['thermal'](thermal), device),
    }
    inputs['language'] = to_device(tokenizer(language, max_length=77, padding='max_length',
                                             truncation=True, return_tensors='pt'), device)

    print(f"inputs:{inputs}")
    with torch.no_grad():
        embeddings,before_proj = model(inputs)
    print("Depth x Text: ",
          torch.softmax(embeddings['depth'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
