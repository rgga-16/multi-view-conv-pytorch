import torch 
from torch import nn 
from torch.nn import functional as F
from utils import device

class Encoder(nn.Module):

    def encoder_layer(self,in_c,out_c):
        return nn.Sequential(
                nn.Conv2d(in_c,out_c,kernel_size=4,stride=2,padding=1,padding_mode='reflect'), 
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(),
            )

    def __init__(self, in_c=1):
        super(Encoder,self).__init__()
        self.in_c = in_c

        self.e1 = self.encoder_layer(in_c,64)
        self.e2 = self.encoder_layer(64,128)
        self.e3 = self.encoder_layer(128,256)
        self.e4 = self.encoder_layer(256,512)
        self.e5 = self.encoder_layer(512,512)
        self.e6 = self.encoder_layer(512,512)
        # self.e7 = self.encoder_layer(512,512)


    def forward(self,input):
        assert input.shape[1]==self.in_c #Assume input=(b,1,256,256) 128
        
        layer_feats = {}
        ef1 = self.e1(input) #ef1 = (b,64,128,128) 64
        layer_feats['ef1']=ef1 
        ef2 = self.e2(ef1) #ef2 = (b,128,64,64) 32
        layer_feats['ef2']=ef2
        ef3 = self.e3(ef2) #ef3 = (b,256,32,32) 16
        layer_feats['ef3']=ef3
        ef4 = self.e4(ef3) #ef4 = (b,512,16,16) 8 
        layer_feats['ef4']=ef4
        ef5 = self.e5(ef4) #ef5 = (b,512,8,8) 4
        layer_feats['ef5']=ef5
        ef6 = self.e6(ef5) #ef6 = (b,512,4,4) 2
        layer_feats['ef6']=ef6
        # ef7 = self.e7(ef6) #ef7 = (b,512,2,2)
        # layer_feats['ef7']=ef7
        ef7=ef6
        return ef7,layer_feats


class Decoder(nn.Module):

    class DecoderLayer(nn.Module):
        def __init__(self,in_c,out_c) -> None:
            super(Decoder.DecoderLayer,self).__init__()
            self.deconv_layer = nn.Conv2d(in_c,out_c,kernel_size=3,stride=1,padding=1,padding_mode='reflect')
            self.batchnorm = nn.BatchNorm2d(out_c)
            self.leakyrelu = nn.LeakyReLU(0.2)
        
        def forward(self,input):
            _,c,h,w = input.shape
            upsampled = F.interpolate(input,size=(h*2,w*2))
            output = self.leakyrelu(self.batchnorm(self.deconv_layer(upsampled)))
            return output


    def __init__(self,out_c=5):
        super(Decoder,self).__init__()
        # self.l7=self.DecoderLayer(512,512)
        self.l6=self.DecoderLayer(1024,512)
        self.l5 = self.DecoderLayer(1024,512)
        self.l4 = self.DecoderLayer(1024,256)
        self.l3 = self.DecoderLayer(512,128)
        self.l2 = self.DecoderLayer(256,64)
        self.l1 = self.DecoderLayer(128,out_c)
        self.tanh = nn.Tanh()


    def forward(self,ef7,layer_feats):
        assert ef7.shape[1]==512
        # df6 = F.dropout(self.l7(ef7)) # df6=(b,512,4,4) 
        df6=ef7
        df5 = F.dropout(self.l6(torch.cat((df6,layer_feats['ef6']),dim=1))) # df5=(b,512,8,8) 4
        df4 = F.dropout(self.l5(torch.cat((df5,layer_feats['ef5']),dim=1))) # df4=(b,512,16,16) 8
        df3 = self.l4(torch.cat((df4,layer_feats['ef4']),dim=1)) #df3=(b,256,32,32) 16
        df2 = self.l3(torch.cat((df3,layer_feats['ef3']),dim=1)) #df2=(b,128,64,64) 32
        df1 = self.l2(torch.cat((df2,layer_feats['ef2']),dim=1)) #df1=(b,64,128,128) 64
        output = self.tanh(self.l1(torch.cat((df1,layer_feats['ef1']),dim=1))) #output=(b,5,256,256) 128

        return output


class MonsterNet(nn.Module):
    def __init__(self,n_target_views,in_c=4,out_c=5):
        super(MonsterNet,self).__init__()
        self.encoder = Encoder(in_c=in_c).to(device)
        self.decoders = [Decoder(out_c=out_c).to(device) for _ in range(n_target_views)]

    

    def forward(self,input):

        views = []
        last_layer_feat,feats = self.encoder(input)
        for dec in self.decoders:
            views.append(dec(last_layer_feat,feats))
        pred = torch.stack(views,dim=0)
        pred = torch.reshape(pred,(pred.shape[0]*pred.shape[1],pred.shape[2],pred.shape[3],pred.shape[4]))
        return pred


class Discriminator(nn.Module):
    def __init__(self,in_c=1) -> None:
        super(Discriminator,self).__init__()

        self.encoder = Encoder(in_c=in_c).to(device)
        self.fc_layer = nn.Linear(2048,1).to(device)
        self.sigmoid = nn.Sigmoid().to(device)
    
    def forward(self,input):
        last_layer_feat,_ = self.encoder(input)
        flat_feats = torch.flatten(last_layer_feat,start_dim=1)


        probs = self.sigmoid(self.fc_layer(flat_feats)).squeeze()
        
        return probs 
if __name__ =='__main__':

    net = MonsterNet(3)
    noise = torch.rand((1,1,256,256),device=device)
    views = net(noise)

    