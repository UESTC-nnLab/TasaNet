import torch
import torch.nn as nn
from darknet import BaseConv
import torch.nn.functional as F
from postion_encoding import PositionEmbeddingLearned
###记录每个版本的区别###

class Neck_V3(nn.Module): 
    def __init__(self, channels=[128,256,512] ,num_frame=5):
        super().__init__()
        self.num_frame = num_frame
        #  关键帧与参考帧融合
        self.conv_ref = nn.Sequential(
            BaseConv(channels[0]*(self.num_frame-1), channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1, act='sigmoid')
        )
        self.conv_cur = BaseConv(channels[0], channels[0],3,1)
        # 参考帧分别与关键帧融合
        self.conv_gl = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
        # 最终融合
        self.conv_gl_mix = nn.Sequential(
            BaseConv(channels[0]*(self.num_frame-1), channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
        self.conv_cr_mix = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )

    def forward(self, feats):
        f_feats = []
        r_feat = torch.cat([feats[j] for j in range(self.num_frame-1)],dim=1)
        r_feat = self.conv_ref(r_feat)
        c_feat = self.conv_cur(r_feat*feats[-1])
        
        r_feats = []
        for i in range(self.num_frame-1):
            r_feats.append(self.conv_gl(torch.cat([feats[i], feats[-1]], dim=1)))
        # r_feat = self.conv_gl_mix(torch.cat([r_feats[i] for i in range(self.num_frame-1)],dim=1))
        r_feat = self.conv_gl_mix(torch.cat(r_feats,dim=1))
        c_feat = self.conv_cr_mix(torch.cat([r_feat, c_feat], dim=1))
        
        f_feats.append(c_feat)
            
        return f_feats
    
class Neck_V6(nn.Module):       # 主干！！！！
    def __init__(self, channels=[128,256,512] ,num_frame=5):
        super().__init__()
        self.num_frame = num_frame
        self.weight = nn.ParameterList(torch.nn.Parameter(torch.tensor([0.25]), requires_grad=True) for _ in range(num_frame))
        #  关键帧与参考帧融合
        self.conv_ref = nn.Sequential(
            BaseConv(channels[0]*(self.num_frame-1), channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1,act='sigmoid')
        )
        self.conv_cur = nn.Sequential(
            BaseConv(channels[0], channels[0],3,1),
            BaseConv(channels[0], channels[0],3,1)
        )
        
        # 参考帧分别与关键帧融合
        self.conv_gl = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
        # 最终融合
        self.conv_gl_mix = nn.Sequential(
            BaseConv(channels[0], channels[0],3,1),
            BaseConv(channels[0],channels[0],3,1)
        )
        self.conv_cr_mix = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
        self.conv_final = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )

    def forward(self, feats):
        f_feats = []
        r_feat = torch.cat([feats[j] for j in range(self.num_frame-1)],dim=1)
        r_feat = self.conv_ref(r_feat)
        c_feat = self.conv_cur(r_feat*feats[-1])
        c_feat = self.conv_cr_mix(torch.cat([c_feat, feats[-1]], dim=1))
        
        r_feats = torch.stack([self.conv_gl(torch.cat([feats[i], feats[-1]], dim=1))*self.weight[i] for i in range(self.num_frame-1)], dim=0)
        r_feat= self.conv_gl_mix(torch.sum(r_feats, dim=0))
        
        c_feat = self.conv_final(torch.cat([r_feat,c_feat], dim=1))
        
        f_feats.append(c_feat)
            
        return f_feats

class Neck_V7(nn.Module):  # 加了一个余弦相似度
    def __init__(self, channels=[128,256,512] ,num_frame=5):
        super().__init__()
        self.num_frame = num_frame
        #  关键帧与参考帧融合
        self.conv_ref = nn.Sequential(
            BaseConv(channels[0]*(self.num_frame-1), channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1,act='sigmoid')
        )
        self.conv_cur = nn.Sequential(
            BaseConv(channels[0], channels[0],3,1),
            BaseConv(channels[0], channels[0],3,1)
        )
        
        # 参考帧分别与关键帧融合
        self.conv_gl = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
        # 最终融合
        self.conv_gl_mix = nn.Sequential(
            BaseConv(channels[0], channels[0],3,1),
            BaseConv(channels[0],channels[0],3,1)
        )
        self.conv_cr_mix = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
        self.conv_final = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )

    def forward(self, feats):
        f_feats = []
        r_feat = torch.cat([feats[j] for j in range(self.num_frame-1)],dim=1)
        r_feat = self.conv_ref(r_feat)
        c_feat = self.conv_cur(r_feat*feats[-1])
        c_feat = self.conv_cr_mix(torch.cat([c_feat, feats[-1]], dim=1))
        
        r_feats = torch.stack([self.conv_gl(torch.cat([feats[i], feats[-1]], dim=1))
                               * torch.reciprocal(torch.abs(F.cosine_similarity(
                                        feats[i].reshape(-1), feats[-1].reshape(-1), dim=0
                                   )))
                               for i in range(self.num_frame-1)], dim=0)
        r_feat= self.conv_gl_mix(torch.sum(r_feats, dim=0))
        
        c_feat = self.conv_final(torch.cat([r_feat,c_feat], dim=1))
        
        f_feats.append(c_feat)
            
        return f_feats
    
class Sap_Tem_Fusion(nn.Module):
    def __init__(self,in_channel, out_channel):
        super().__init__()
        self.conv_1 = nn.Sequential(
            BaseConv(in_channel, in_channel//2, 1, 1),
            BaseConv(in_channel//2, in_channel, 1, 1)
        )
        self.conv_2 = nn.Sequential(
            BaseConv(in_channel, in_channel//2, 1, 1),
            BaseConv(in_channel//2, in_channel, 1, 1, act="sigmoid")
        )
        self.conv = nn.Sequential(
            BaseConv(in_channel, in_channel//2, 1, 1),
            BaseConv(in_channel//2, out_channel, 1, 1)
        )
        
    def forward(self, r_feat, c_feat):
        m_feat = r_feat + c_feat
        m_feat = self.conv_2(self.conv_1(m_feat))
        m_feat = self.conv(c_feat*m_feat + r_feat*(1-m_feat))
        
        return m_feat
    
class Neck_V8(nn.Module):  # 加了一个余弦相似度,特征融合模块
    def __init__(self, channels=[128,256,512] ,num_frame=5):
        super().__init__()
        self.num_frame = num_frame
        #  关键帧与参考帧融合
        self.conv_ref = nn.Sequential(
            BaseConv(channels[0]*(self.num_frame-1), channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1,act='sigmoid')
        )
        self.conv_cur = nn.Sequential(
            BaseConv(channels[0], channels[0],3,1),
            BaseConv(channels[0], channels[0],3,1)
        )
        
        # 参考帧分别与关键帧融合
        self.conv_gl = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
        # 最终融合
        self.conv_gl_mix = nn.Sequential(
            BaseConv(channels[0], channels[0],3,1),
            BaseConv(channels[0],channels[0],3,1)
        )
        self.conv_cr_mix = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
        self.conv_final = Sap_Tem_Fusion(channels[0], channels[0])
    def forward(self, feats):
        f_feats = []
        r_feat = torch.cat([feats[j] for j in range(self.num_frame-1)],dim=1)
        r_feat = self.conv_ref(r_feat)
        c_feat = self.conv_cur(r_feat*feats[-1])
        c_feat = self.conv_cr_mix(torch.cat([c_feat, feats[-1]], dim=1))
        
        r_feats = torch.stack([self.conv_gl(torch.cat([feats[i], feats[-1]], dim=1))
                               * torch.reciprocal(torch.abs(F.cosine_similarity(
                                        feats[i].reshape(-1), feats[-1].reshape(-1), dim=0
                                   )))
                               for i in range(self.num_frame-1)], dim=0)
        r_feat= self.conv_gl_mix(torch.sum(r_feats, dim=0))
        
        c_feat = self.conv_final(r_feat, c_feat)
        
        f_feats.append(c_feat)
            
        return f_feats
    
class MSA(nn.Module):
    def __init__(self, channels=[128,256,512], num_frame=5, dim=1024):
        super().__init__()
        self.num_frame = num_frame
        self.conv_ref = nn.Sequential(
            BaseConv(channels[0]*(self.num_frame-1), channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
        self.K = BaseConv(channels[0], channels[0], 3, 2)
        self.V = BaseConv(channels[0], channels[0], 3, 2)
        self.Q = nn.Sequential(
            BaseConv(channels[0], channels[0],3,2),
            BaseConv(channels[0], channels[0],1,1)
        )
        self.position = PositionEmbeddingLearned(num_pos_feats=64)
        self.attn = nn.MultiheadAttention(embed_dim=1024, num_heads=8)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Linear(dim, dim)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            BaseConv(channels[0], channels[0], 3, 1)
        )
        
    def forward(self, ref, cur):
        B, C, H, W = cur.shape
        ref = self.conv_ref(ref)
        K, V = self.K(ref), self.V(ref)
        Q = self.Q(cur)
        pos = self.position(Q).reshape(B,C,-1)
        attn, _ = self.attn(Q.reshape(B,C,-1)+pos, K.reshape(B,C,-1)+pos, V.reshape(B,C,-1)+pos)
        attn = self.norm(attn+Q.reshape(B,C,-1))
        attn = self.norm(attn + self.ffn(attn)).reshape(B,C,H//2,W//2)
        attn = self.upsample(attn)
        return attn
        
    
class Neck_V9(nn.Module):  # 加了一个余弦相似度,特征融合模块, self-attention
    def __init__(self, channels=[128,256,512] ,num_frame=5):
        super().__init__()
        self.num_frame = num_frame
        #  关键帧与参考帧融合
        self.attn = MSA(channels=channels,num_frame=5,dim=1024)
        
        # 参考帧分别与关键帧融合
        self.conv_gl = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
        # 最终融合
        self.conv_gl_mix = nn.Sequential(
            BaseConv(channels[0], channels[0],3,1),
            BaseConv(channels[0],channels[0],3,1)
        )
        self.conv_cr_mix = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
        self.conv_final = Sap_Tem_Fusion(channels[0], channels[0])
    def forward(self, feats):
        f_feats = []
        r_feat = torch.cat([feats[j] for j in range(self.num_frame-1)],dim=1)
        c_feat = self.attn(r_feat, feats[-1])
        c_feat = self.conv_cr_mix(torch.cat([c_feat, feats[-1]], dim=1))
        
        r_feats = torch.stack([self.conv_gl(torch.cat([feats[i], feats[-1]], dim=1))
                               * torch.reciprocal(torch.abs(F.cosine_similarity(
                                        feats[i].reshape(-1), feats[-1].reshape(-1), dim=0
                                   )))
                               for i in range(self.num_frame-1)], dim=0)
        r_feat= self.conv_gl_mix(torch.sum(r_feats, dim=0))
        
        c_feat = self.conv_final(r_feat, c_feat)
        
        f_feats.append(c_feat)
            
        return f_feats

class Neck_V10(nn.Module):  # 加了一个余弦相似度,特征融合模块, self-attention
    def __init__(self, channels=[128,256,512] ,num_frame=5):
        super().__init__()
        self.num_frame = num_frame
        #  关键帧与参考帧融合
        self.conv_ref = nn.Sequential(
            BaseConv(channels[0]*(self.num_frame-1), channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1,act='sigmoid')
        )
        self.conv_cur = nn.Sequential(
            BaseConv(channels[0], channels[0],3,1),
            BaseConv(channels[0], channels[0],3,1)
        )
        self.attn = MSA(channels=channels,num_frame=5,dim=1024)
        # 参考帧分别与关键帧融合
        self.conv_gl = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
        # 最终融合
        self.conv_gl_mix = nn.Sequential(
            BaseConv(channels[0], channels[0],3,1),
            BaseConv(channels[0],channels[0],3,1)
        )
        self.conv_final = Sap_Tem_Fusion(channels[0], channels[0])
    def forward(self, feats):
        f_feats = []
        r_feat = torch.cat([feats[j] for j in range(self.num_frame-1)],dim=1)
        r_feat = self.conv_ref(r_feat)
        c_feat = self.conv_cur(r_feat*feats[-1])
        c_feat = self.attn(c_feat, feats[-1])  # MSA会做修改
        
        r_feats = torch.stack([self.conv_gl(torch.cat([feats[i], feats[-1]], dim=1))
                               * torch.reciprocal(torch.abs(F.cosine_similarity(
                                        feats[i].reshape(-1), feats[-1].reshape(-1), dim=0
                                   )))
                               for i in range(self.num_frame-1)], dim=0)
        r_feat= self.conv_gl_mix(torch.sum(r_feats, dim=0))
        
        c_feat = self.conv_final(r_feat, c_feat)
        
        f_feats.append(c_feat)
            
        return f_feats

class MSA_(nn.Module):
    def __init__(self, channels=[128,256,512], num_frame=5, dim=1024):
        super().__init__()
        self.num_frame = num_frame
        self.K = nn.Sequential(
            BaseConv(channels[0], channels[0],3,2),
            BaseConv(channels[0], channels[0],1,1)
        )
        self.V = nn.Sequential(
            BaseConv(channels[0], channels[0],3,2),
            BaseConv(channels[0], channels[0],1,1)
        )
        self.Q = nn.Sequential(
            BaseConv(channels[0], channels[0],3,2),
            BaseConv(channels[0], channels[0],1,1)
        )
        self.position = PositionEmbeddingLearned(num_pos_feats=64)
        self.attn = nn.MultiheadAttention(embed_dim=1024, num_heads=8)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Linear(dim, dim)

    def forward(self, ref, cur):
        B, C, H, W = cur.shape
        K, V = self.K(ref), self.V(ref)
        Q = self.Q(cur)
        pos = self.position(Q).reshape(B,C,-1)
        attn, _ = self.attn(Q.reshape(B,C,-1)+pos, K.reshape(B,C,-1)+pos, V.reshape(B,C,-1)+pos)
        attn = self.norm(attn+Q.reshape(B,C,-1))
        attn = self.norm(attn + self.ffn(attn)).reshape(B,C,H//2,W//2)
        return attn
class Neck_V11(nn.Module):  
    def __init__(self, channels=[128,256,512] ,num_frame=5):
        super().__init__()
        self.num_frame = num_frame
        #  关键帧与参考帧融合
        self.conv_ref = nn.Sequential(
            BaseConv(channels[0]*(self.num_frame-1), channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1,act='sigmoid')
        )
        self.conv_cur = nn.Sequential(
            BaseConv(channels[0], channels[0],3,1),
            BaseConv(channels[0], channels[0],3,1)
        )
        
        # 参考帧分别与关键帧融合
        for i in range(1,num_frame):
            self.__setattr__("attn_%d"%i, MSA_(channels=channels,num_frame=5,dim=1024)) 

        # 最终融合
        self.conv_gl_mix = nn.Sequential(
            BaseConv(channels[0], channels[0],3,1),
            BaseConv(channels[0],channels[0],3,1)
        )
        self.conv_cr_mix = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
        self.conv_final = Sap_Tem_Fusion(channels[0], channels[0])

    def forward(self, feats):
        f_feats = []
        r_feat = torch.cat([feats[j] for j in range(self.num_frame-1)],dim=1)
        r_feat = self.conv_ref(r_feat)
        c_feat = self.conv_cur(r_feat*feats[-1])
        c_feat = self.conv_cr_mix(torch.cat([c_feat, feats[-1]], dim=1))
        
        r_feats = torch.cat([self.__getattr__("attn_%d"%i)(feats[i-1], feats[-1]) 
                             for i in range(1, self.num_frame)], dim=1)
        r_feat= self.conv_gl_mix(r_feats)
        
        c_feat = self.conv_final(r_feat,c_feat)
        
        f_feats.append(c_feat)
            
        return f_feats

if __name__ == "__main__":
    a = [torch.randn([4,128,64,64]) for _ in range(5)]
    print(Neck_V11(channels=[128],num_frame=5)(a))