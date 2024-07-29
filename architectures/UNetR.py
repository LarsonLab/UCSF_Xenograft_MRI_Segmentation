import torch 
import torch.nn as nn 

class PatchCreator(nn.Module):
    def __init__(self,config):
        super(PatchCreator,self).__init__()
        self.patch_size= config['patch_size']
        self.unfold = nn.Unfold(kernel_size=(self.patch_size,self.patch_size),stride=self.patch_size)

    def forward(self,x):
        #shape = [B,C,H,W]
        patches = self.unfold(x)
        #shape = [B,patch height * patch width, num patches]
        patches = patches.transpose(1,2)

        return patches



class ConvBlock(nn.Module):
    def __init__(self,in_c,out_c,kernel_size=3,padding=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=kernel_size,padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.4)
        )

    def forward(self,x):
        return self.layers(x)
    
class DeconvBlock(nn.Module): 

    def __init__(self,in_c,out_c,):
        super(DeconvBlock,self).__init__()
        
        self.deconv = nn.ConvTranspose2d(in_c,out_c,kernel_size=2,stride=2,padding=0)

    def forward(self,x):
        return self.deconv(x)
    

class UNetR(nn.Module):

    #cf refers to the cofiguration 
    def __init__(self,cf):
        super().__init__()
        self.cf = cf 
        self.patch_creator = PatchCreator(cf)

        #Patch + position embeddings
        self.patch_embed = nn.Linear(
            cf["patch_size"]*cf["patch_size"]*cf["num_channels"],
            cf["hidden_dim"])
        self.positions = torch.arange(start=0,end=cf['num_patches'],step=1,dtype=torch.int32)
        self.pos_embed = nn.Embedding(cf['num_patches'],cf['hidden_dim'])
        self.trans_encoder_layers = []
        
        for i in range(cf['num_layers']):
            layer = nn.TransformerEncoderLayer(
                d_model=cf['hidden_dim'],
                nhead = cf['num_heads'],
                dim_feedforward = cf['mlp_dim'],
                dropout= cf['dropout_rate'],
                activation = nn.GELU(),
                batch_first=True
            )
            self.trans_encoder_layers.append(layer)

        #cnn decoder 
        self.d1 = DeconvBlock(cf['hidden_dim'],512)
        self.s1 = nn.Sequential(
            DeconvBlock(cf['hidden_dim'],512),
            ConvBlock(512,512)
        )
        self.c1 = nn.Sequential(
            ConvBlock(512+512,512),
            ConvBlock(512,512)
            )
        self.d2 = DeconvBlock(512,256)
        self.s2 = nn.Sequential(
            DeconvBlock(cf['hidden_dim'],256),
            ConvBlock(256,256),
            DeconvBlock(256,256),
            ConvBlock(256,256)
        )
        self.c2 = nn.Sequential(
            ConvBlock(256+256,256),
            ConvBlock(256,256)
        )
        self.d3 = DeconvBlock(256,128)
        self.s3 = nn.Sequential(
            DeconvBlock(cf['hidden_dim'],128),
            ConvBlock(128,128),
            DeconvBlock(128,128),
            ConvBlock(128,128),
            DeconvBlock(128,128),
            ConvBlock(128,128)
        )
        self.c3 = nn.Sequential(
            ConvBlock(128+128,128),
            ConvBlock(128,128)
        )

        self.d4 = DeconvBlock(128,64)
        self.s4 = nn.Sequential(
            ConvBlock(1,64),
            ConvBlock(64,64)
        )
        self.c4 = nn.Sequential(
            ConvBlock(64+64,64),
            ConvBlock(64,64)
        )
        #output

        self.output = nn.Conv2d(64,1,kernel_size=1,padding=0)



        




    def forward(self,inputs): 
        #patch and position embeddings
        device = inputs.device 
        batch_size = inputs.shape[0]
        for layer in self.trans_encoder_layers:
            layer.to(device)
        patches = self.patch_creator(inputs) # [2,144,256]
        patch_embed = self.patch_embed(patches) # [2,144,768]
        positions = self.positions.to(device)
        pos_embed = self.pos_embed(positions) # [144,768]
        x = patch_embed + pos_embed # [2,144,768]
        #transformer encoder 

        skip_connection_index = [3,6,9,12]
        skip_connections = []
        for i in range(self.cf['num_layers']):
            layer = self.trans_encoder_layers[i]
            x = layer(x) # [2,144,768]
            if (i+1) in skip_connection_index:
                skip_connections.append(x)

        #cnn decoder 
        z3, z6, z9, z12 = skip_connections 

        #reshaping 
        batch = patches.shape[0]
        z0 = patches.reshape(batch,self.cf['num_channels'],self.cf['image_size'],self.cf['image_size'])
        shape = (batch,self.cf['hidden_dim'],self.cf['image_size']//self.cf['patch_size'],self.cf['image_size']//self.cf['patch_size'])
 
        z3 = z3.view(shape)
        z6 = z6.view(shape)
        z9 = z9.view(shape)
        z12 = z12.view(shape)

        x = self.d1(z12)
        s = self.s1(z9)
        x = torch.cat([x,s],dim=1)
        x = self.c1(x)

        x = self.d2(x)
        s = self.s2(z6)
        x = torch.cat([x,s],dim=1)
        x = self.c2(x)

        x = self.d3(x)
        s = self.s3(z3)
        x = torch.cat([x,s],dim=1)
        x = self.c3(x)

        x = self.d4(x)
        s = self.s4(z0)
        x = torch.cat([x,s],dim=1)
        x = self.c4(x)

        #output 
        output = self.output(x)

        return output 








def get_default_config():
    return {
        'image_size': 192,
        'num_layers': 12,
        'hidden_dim': 768,
        'mlp_dim': 3072,
        'num_heads': 12,
        'dropout_rate': 0.1,
        'num_patches': 144,
        'patch_size': 16,
        'num_channels': 1
    }


if __name__ == '__main__':
    config = {
        'image_size': 192,
        'num_layers': 12,
        'hidden_dim': 768,
        'mlp_dim': 3072,
        'num_heads': 12,
        'dropout_rate': 0.1,
        'num_patches': 144,
        'patch_size': 16,
        'num_channels': 1
    }

    #test input 
    x = torch.randn((
        2,config['num_patches']*config['patch_size']*config['num_channels']
    ))

def create_UNetR():
    config = get_default_config()
    return UNetR(config)









