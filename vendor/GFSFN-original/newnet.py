import torch
import torch.nn as nn
import torch.nn.functional as F
class lush(nn.Module):
    def __init__(self):
        super(lush, self).__init__()
    def forward(self, x):
        return torch.where(x < 0, x * torch.tanh(F.softplus(x)), x)
class CTEM(nn.Module):
    def __init__(self,inchanle):
        super(CTEM, self).__init__()
        self.ln=nn.BatchNorm1d(inchanle)#nn.LayerNorm([inchanle,25])
        self.c=nn.Sequential(nn.Conv1d(inchanle,inchanle,1),nn.Mish(),)
        self.c2 = nn.Sequential(nn.Conv1d(inchanle, inchanle, 1,),nn.BatchNorm1d(inchanle))
        self.dwc=nn.Sequential(nn.Conv1d(inchanle,inchanle,3,1,1,groups=inchanle),
                               nn.BatchNorm1d(inchanle),)#
        self.dc=nn.Sequential(nn.Conv1d(inchanle,inchanle,1,dilation=1),
                              nn.Mish()
                              )
    def forward(self,x):
        x1=self.ln(x)
        x2=self.c(x1)
        x3=self.dwc(x2)
        x0=self.dc(x1)
        x4=x0*x3
        x5=self.c2(x4)
        return x5+x1
class MixTrans(nn.Module):
    def __init__(self,dim=224,dim2=224):
        super(MixTrans,self).__init__()
        self.a=nn.Sequential(nn.BatchNorm1d(dim), SFA(dim), nn.Mish(),)
        self.a2 = nn.Sequential(nn.BatchNorm1d(dim), CTEM(dim),nn.Mish())
        self.b=nn.Sequential(nn.BatchNorm1d(dim))
        self.p=nn.Parameter(torch.ones(1))
    def forward(self,x):#-1 224 25
        x1 = self.a2(x)
        x2=self.a(x)
        x3=x2*x1
        ou2 = self.b(x3)
        return ou2
class CMSF(nn.Module):
    def __init__(self,):
        super(CMSF,self).__init__()
        self.c1 = nn.Sequential(nn.Conv3d(32, 32, (1, 3, 3)), lush(), nn.BatchNorm3d(32), )
        self.c2 = nn.Sequential(nn.Conv3d(64, 64, (1, 3, 3)), lush(), nn.BatchNorm3d(64), )
        self.c3 = nn.Sequential(nn.Conv3d(128, 128, (1, 3, 3)),lush(), nn.BatchNorm3d(128), )
        self.c10 = nn.Sequential(nn.Conv3d(32, 32, (1, 3, 3)),lush(), nn.BatchNorm3d(32), )
        self.c20 = nn.Sequential(nn.Conv3d(64, 64, (1, 3, 3)),lush(), nn.BatchNorm3d(64), )
        self.c30 = nn.Sequential(nn.Conv3d(128, 128, (1, 3, 3)), lush(), nn.BatchNorm3d(128), )
        self.c11 = nn.Sequential(nn.Conv3d(32, 32, (1, 3, 3)), lush(), nn.BatchNorm3d(32), )
        self.c21 = nn.Sequential(nn.Conv3d(64, 64, (1, 3, 3)), lush(), nn.BatchNorm3d(64), )
        self.c31 = nn.Sequential(nn.Conv3d(128, 128, (1, 3, 3)),lush(), nn.BatchNorm3d(128), )
    def forward(self,x,x2,i):#-1 224 81
       if(i==2):
           x1=self.c1(x)
           x22=self.c10(x2)
           out=x1 + x22
           out=self.c11(out)
       if (i == 5):
           x1 = self.c2(x)
           x22 = self.c20(x2)
           out = x1 + x22
           out = self.c21(out)
       if (i == 8):
           x1 = self.c3(x)
           x22 = self.c30(x2)
           out = x1 + x22
           out = self.c31(out)
       return out
class MLGF(nn.Module):
    def __init__(self,in1,in2,in3,ou):# 32*7
        super(MLGF, self).__init__()
        self.aa=nn.Sequential(lush())#nn.Tanh()PLush()lush()
        self.ab=nn.Sequential(nn.Sigmoid())    #-1 224 81
        self.c1=nn.Sequential(nn.Conv1d(in1,ou,1,),nn.BatchNorm1d(ou))
        self.c2 = nn.Sequential(nn.Conv1d(in2, ou, 1,),nn.BatchNorm1d(ou))
        self.c3 = nn.Sequential(nn.Conv1d(in3, ou, 1),nn.BatchNorm1d(ou))
        self.ac=nn.Sequential(lush())#
    def forward(self, x1,x2,x3):#
        x1 = self.c1(x1)
        x2 = self.c2(x2)
        x3 = self.c3(x3)
        x11 = self.aa(x1)
        x12 = self.ab(x1)
        x21 = self.aa(x2)
        x22 = self.ab(x2)
        x31 = self.aa(x3)
        x32 = self.ab(x3)
        x10 = x11 * x12
        x20 = x21 * x22
        x30 = x31 * x32
        ok = x10+x20+x30
        return self.ac(ok)
class SFA(nn.Module):
    def __init__(self,inc):
        super(SFA,self).__init__()
        self.c0 = nn.Sequential(nn.Conv1d(inc, inc,3,1,1,),nn.Mish())
        self.c1=nn.Sequential(nn.Conv1d(inc, inc, 3,1,1,),nn.Mish())
        self.sm = nn.Sequential(nn.Softmax(dim=-1))
        self.p=nn.Parameter(torch.ones(1))
    def forward(self,x):
        x1=self.c0(x)
        x2= self.c1(x)
        x3 = x
        f = torch.fft.fft(x1)
        fshift = torch.fft.fftshift(f)
        f2 = torch.fft.fft(x2)
        fshift2 = torch.fft.fftshift(f2)
        fn=fshift*fshift2
        ishift = torch.fft.ifftshift(fn)
        back = torch.abs(torch.fft.ifft(ishift))
        back = self.sm(back)
        back=back*x3
        return self.p*back+x
class GFSFN(nn.Module):
    def __init__(self,  num_classes=6,dim=224,band1=30,band2=1):
        super(GFSFN, self).__init__()
        self.name='GFSFN'
        self.band2=band2
        self.conv01 = nn.Sequential(nn.Conv3d(1, 32, (1, 3, 3), (1, 1, 1), (0, 0, 0),dilation=1),  #
                                    nn.BatchNorm3d(32),
                                    nn.Mish(),
                                    nn.Conv3d(32, 64, (1, 1, 1), (1, 1, 1), (0, 0, 0), dilation=1),  #
                                    nn.BatchNorm3d(64),
                                    nn.Mish(),
                                    nn.Conv3d(64, 128, (band1, 1, 1), (1, 1, 1), (0, 0, 0), dilation=1),  #
                                    nn.BatchNorm3d(128),
                                    nn.Mish()
                                    )
        self.cl1=nn.Sequential(nn.Conv3d(1, 32, (1, 3, 3), (1, 1, 1), (0, 0, 0),dilation=1 ),  #
                                    nn.BatchNorm3d(32),
                                    nn.Mish(),
                                    nn.Conv3d(32, 64, (1, 1, 1), (1, 1, 1), (0, 0, 0), dilation=1),  #
                                    nn.BatchNorm3d(64),
                                    nn.Mish(),
                                    nn.Conv3d(64, 128, (band2, 1, 1), (1, 1, 1), (0, 0, 0), dilation=1),  #
                                    nn.BatchNorm3d(128),
                                    nn.Mish()
                               )
        self.Mt=MixTrans(dim=dim)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.fs=MLGF(32,64,128,dim)
        self.fffu=CMSF()
        self.full_connection = nn.Sequential(
            nn.Linear(dim, num_classes),
        )
    def forward(self, x,x2):
        x1 = x.unsqueeze(1)
        x02=x2.unsqueeze(1)
        a=x1
        b=x02
        xs=[]
        for i in range(9):
            a=self.conv01[i](a)
            b=self.cl1[i](b)
            if (i==2 or i==5 or i==8):
               a1,_=torch.max(a,dim=2,keepdim=True)
               a2 = torch.mean(a, dim=2, keepdim=True)
               b1,_=torch.max(b,dim=2,keepdim=True)
               b2 = torch.mean(b, dim=2, keepdim=True)
               aa=self.fffu(a1+b1,a2+b2,i)
               m_batchsize, c,channle, height, width = aa.size()
               aa = aa.view(m_batchsize, c, -1)
               xs.append(aa)
        xo=self.fs(xs[0],xs[1],xs[2])
        out= self.Mt(xo)
        out=self.global_pooling(out).squeeze().squeeze()
        out1=self.full_connection(out)
        return out1

