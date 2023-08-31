# Deep Snake代码阅读记录3

这个代码里面涉及到的函数太多，且都没有注释，需要对此进行一个详细而完整的梳理。

## 1 核心网络结构代码

```python
class Network(nn.Module):`

  `def __init__(self, num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):`

​    `super(Network, self).__init__()`



​    `\# 参数 num_layers=34, heads=CfgNode({'act_hm': 8, 'awh': 2}),` 

​    `\# head_conv=256, down_ratio=4,` 

​    `\# det_dir='/media/disk/exp/snake-master/snake-master/data/model/rcnn_snake/'`

​    `\# 这三个组件的作用依次是：`

​    `\#` 

​    `self.dla = DLASeg('dla{}'.format(num_layers), heads,`

​             `pretrained=True,`

​             `down_ratio=down_ratio,`

​             `final_kernel=1,`

​             `last_level=5,`

​             `head_conv=head_conv)`

​    `self.cp = ComponentDetection()`

​    `self.gcn = Evolution()`



​    `\# 注意cfg.det_mode是定义在相关的yaml配置文件中，而不是在总配置文件里面`

​    `\# cfg.det_dir则是模型目录 + 任务名 + 参数args.det`

​    `det_dir = os.path.join(os.path.dirname(cfg.model_dir), cfg.det_model)`

​    `\# 非严格加载网络net的检查点文件`

​    `\# net_utils.load_network(self, det_dir, strict=False)`



  `def decode_detection(self, output, h, w):`

​    `ct_hm = output['act_hm']`

​    `wh = output['awh']`

​    `\# 为什么事先需要对热图使用sigmoid呢`

​    `ct, detection = snake_decode.decode_ct_hm(torch.sigmoid(ct_hm), wh)`

​    `detection[..., :4] = data_utils.clip_to_image(detection[..., :4], h, w)`

​    `output.update({'ct': ct, 'detection': detection})`

​    `return ct, detection`



  `def forward(self, x, batch=None): # {'act_hm': torch.Size([1, 8, 304, 608]), 'awh': torch.Size([1, 2, 304, 608])}`

​    `output, cnn_feature = self.dla(x)` 

​    `with torch.no_grad(): # 这里是将函数的返回值添加到了output的键值中`

​      `self.decode_detection(output, cnn_feature.size(2), cnn_feature.size(3))`

​    `output = self.cp(output, cnn_feature, batch)`

​    `output = self.gcn(output, cnn_feature, batch)`

​    `return output`
```

## 2  网络从输入到输出的分析

### 2.1 特征提取网络 DLASeg层

#### 2.1.1 网络结构：

```
class DLASeg(nn.Module):
    # final_kernel,
    # last_level, head_conv, out_channel=0
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0):
        super(DLASeg, self).__init__()
        
        # 这里应该保证都是按照2的倍数来进行缩小的
        assert down_ratio in [2, 4, 8, 16]
        # 因为每层缩小2倍，直接对缩小的倍数求log即可知连续降采样的层数
        self.first_level = int(np.log2(down_ratio)) # 2
        self.last_level = last_level   # 5
        
        
        # base_name 'dla34'  pretraine=True
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels # [16,32,64,128, 256, 512]
        scales = [2 ** i for i in range(len(channels[self.first_level:]))] # [1,2,4,8]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales) # 2 【64,128, 256, 512】 [1,2,4,8]
        # 如果没有指定输出的维度，那么就指定输出维度为主干提取网络的first_level处的维度
        if out_channel == 0:
            out_channel = channels[self.first_level] # [64]

        # 又进行了一次IDAUp，【proj(128,64),node(64,64),up(64,64,f=2)] , 【proj(256,64),node(64,64),up(64,64,f=4)】
        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], # 64  [64, 128, 256]  [1, 2, 4]
                            [2 ** i for i in range(self.last_level - self.first_level)])

        self.heads = heads # CfgNode({'act_hm': 8, 'awh': 2})
        # 1这里的两头是什么意思 对应的类的个数又是什么含义？
        # 感觉第一个好像heatmap中对应8个类的对象，每个对象对应宽和高坐标共需要两位来存储
        # 2 为什么hm对应的最后的卷积层需要将bias初始化为负数
        # 而宽高则将bias初始化为0呢？
        # 3 head_conv大于小于0会影响到初始通道数到head_conv这一中间过度卷积层的存在
        #   那什么时候会需要head_conv呢？
        for head in self.heads:
            classes = self.heads[head] # 8 2 
            if head_conv > 0: # 256
                fc = nn.Sequential(
                    nn.Conv2d(channels[self.first_level], head_conv, # 64 256
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,   # 256 8   | 256 2
                              kernel_size=final_kernel, stride=1,
                              padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(channels[self.first_level], classes,
                               kernel_size=final_kernel, stride=1,
                               padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)
```

说明：网络使用dla34来作为主干特征提取网络，然后一层DLA_UP层，一层IDA_UP层，以及两类预测头act_hm以及awh。

#### 2.2.2 输入输出分析

```python
`output, cnn_feature = self.dla(x)`
```

```
 def forward(self, x):
       
        x = self.base(x)  
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return z, y[-1]
```

#### 2.2.2.1 输入

​       输入x是批处理的图片，比如说[3, 3, 1216, 2432];**

#### 2.2.2.2 dla34网络处理**

（1）**处理结果：** [3, 3, 1216, 2432]  ====》x = [3, 16, 1216, 2432]

​                                                     ====>y = [ [3, 16, 1216，2432]、[3, 32, 608，1216]、

​                                                                       **[3, 64，304，608 ]、[3, 128，152，304]、[3, 256, 76、152]、[3, 512, 38, 76]]** 

（2）**处理过程代码：**

```python
def forward(self, x):
        y = []  # torch.Size([3, 8, 1216, 2432])
        x = self.base_layer(x) # torch.Size([3, 16, 1216, 2432])
   
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y
```

（3）**网络结构总结：**

​      首先第一步使用base网络进行处理，这里的base指的是dla34网络，dla34网络结构是DLA结构的一种特殊情况，也就是说DLA网络中的后6层结构中每层结构对应的网络层数levels是[1, 1, 1, 2, 2, 1]、每层结构对应的维度数channels分别是[16,32,64,128,256,512]。



​       dla34的网络结构为：

​      1）1层基础卷积：ConvBnReLu结构base（维度从3变为16，图片大小不变）、

​      2）2层普通卷积层：0层卷积level0的结果图片维度和大小都保持不变（16-》16，1-》1），相当于是做了一层特征增强； 1层卷积level1降采样2倍、维度增加2倍（16-》32，1-》1/2）

​      3）4层树结构卷积：每棵树的对应的通道数都翻倍，图片大小都缩减为一半 (32-》512，1/2-》1/32)、

```
# 构建dla34模型并使用了预训练模型
def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model
```



​       DLA从通用的角度来说共包含7层结构，第一层是基础特征提取层base_layer，维度数从3变成channels[0]；然后是两层普通卷积层level0和level1，level0对应levels[0]个ConvBnReLu层以实现维度从channels[0]转向channels[0]（其中的第一个允许stride不等同于1，也就是说可以进行下采样，后面的则不可以进行下采样，stride通通默认为1；第一个的维度转化是channels[0]->channels[0], 后面的是channels[0]->channels[0]），level1对应levels[1]个ConvBnReLu层以实现维度从channels[0]转向channels[1]（第一个的维度转化是channels[0]->channels[1], 后面的是channels[1]->channels[1]）；最后4层树结构卷积，每棵树对应的深度分别为levels[2]、levels[3]、levels[4]、levels[5]，对应的通道数变换分别是channels[1]->channels[2]、channels[2]->channels[3]、channels[3]->channels[4]

channels[4]->channels[5]。需要指出一点的是，这里的7层对应的卷积核的个数都设置为3，每层可以设置不同的stride大小来控制图片尺寸的大小变换。

```
class DLA(nn.Module):
    # [1, 1, 1, 2, 2, 1] levels
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels # [16,32,64,128,256,512]
        self.num_classes = num_classes
        # DLA的基础层是一层普通卷积ConvBnReLu结构
        # 主要是将图片的channel数从3变换到16
        # 不改变图片的大小，因为（x+3*2 - 7)/1 + 1 = x
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        # 16, 16, 1
        # levels [1,1,1,2,2,1]
        # 1层卷积 通道数从16-》16，stride=1即没有下采样
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])           
        # 16,32,1   1216->608
        # 1层卷积 通道数从16-》32，stride=2进行了一次下采样
        # 通常都是图片下采样但是通道数增加
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
   
        # 连续4棵树，只有中间的是2层结构
        # 这4棵树的通道数都发生了变化，所以都包含映射层
        # stride都不为1，即都发生了降采样，所以都包含downsample模块   
        # 这四棵树的总体效果是每棵树的对应的通道数都翻倍，图片大小都缩减为一半            
        # 32-》64   608->304
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        # 64 -》 128  304->152
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        # 128 -》 256 152->76
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        # 256 -》 512  76->38
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    
    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    # 此函数即制造连续的几个卷积层
    # 其中第一个卷积层的通道数从inplanes变化到planes，其后面的卷积层保持通道数为planes不变
    # 其中第一个卷积层允许进行下采样，其余的不运行进行下采样
    # 卷积层的个数由参数convs来决定
    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)
```

#### 2.2.2.3 DLA_UP网络对不同尺度的图片特征进行融合**

（1）**处理结果：**layers = y = [ [3, 16, 1216，2432]、[3, 32, 608，1216]、

​                                                                       **[3, 64，304，608 ]、[3, 128，152，304]、[3, 256, 76、152]、[3, 512, 38, 76]]** 

====> 【（64，1/4)，(128, 1/8)，(256，1/16），（512，1/32)】

​          j即  [ [3, 64，304，608], [3, 128, 152，304]、[3, 256, 76、152]、[3, 512, 38, 76]]** (全部是layers[5]变换得来)

（2）**处理过程代码：**

```python
def forward(self, layers):

​    # layers到底指的是什么？

​    print('layers is:', layers)

​    out = [layers[-1]] 

​    for i in range(len(layers) - self.startp - 1):

​      ida = getattr(self, 'ida_{}'.format(i))

​      ida(layers, len(layers) - i - 2, len(layers))   # 每次只改变layers中的部分结果

​      out.insert(0, layers[-1])

​    print('out is:', out)

​    return out
```

（3）**DLAUP网络结构分析总结：**

​        DLAUP层其实就是利用多个IDA_UP层来对从dla34获取的多尺度特征进行有效融合。

【1】DLAUP网络中的startp对应于输入中的first_level，其也就是我们需要进行下采样的层数，这里是需要下采样4倍，也就是2层；

```
self.first_level = int(np.log2(down_ratio)) # 2 因为每层缩小2倍，直接对缩小的倍数求log即可知连续降采样的层数
self.last_level = last_level   # 5
```



【2】channels和scales是一一对应的，scales[i] = channes[i] // channes[0], 也就是通道数相对于第一个通道的倍数关系；同时可以发现channels中值的大小按照顺序依次增大2倍；

 注：容易发现这里面的channels和scales对应的是first_level后面的几层，而非first_level这些层。

```
 channels = self.base.channels # [16,32,64,128, 256, 512]
 scales = [2 ** i for i in range(len(channels[self.first_level:]))] # [1,2,4,8]
 self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales) # 2 【64,128, 256, 512】 [1,2,4,8]
```

```
scales = np.array(scales, dtype=int) # 要求必须是整数倍的尺度
```



【3】DLA_UP网络包含的IDA_UP层的个数等于len(channels[self.first_level:]))-1，也就是我们需要进行上采样的层数。

​		这里的channels[self.first_level:]数组用于辅助我们实现相应的通道数变化关系，这里是[16,32,64,128, 256, 512] [2:] = 【64,128, 256, 512】，我们所用的IDA_UP层数总是固定为该辅助数组的长度减1。



​        在本问题中，我们已经通过dla34网络获取了关于图片通道数和图片尺寸依次为（16，1）、（32，1/2）、（64，1/4)、（128，1/8)、（256，1/16)、（512，1/32)的6种尺度特征，而这里的DLA_UP主要是对后面4种尺度的特征进行融合。这里的【64,128, 256, 512】也可以看做是layers[2]、layers[3]、layers[4]、layers[5]当前对应的通道数，后面会随着更新发生相应的改变。



​       **具体的融合方式可以描述为：**

​       **（1）对layers[5]特征进行更新：**首先对layers[5]进行通道数减半512->256、图片大小上采样2倍 1/32->1/16的转换（依次使用proj(512, 256)、upsample(256,256,f=2) ），然后与同样通道数和尺寸layers[4]相加进行融合（node(256,256)) 得到更新的layers[5]；

​         layers[5] = [3, 512, 38, 76] ===》 layers[5] = [3, 256, 38, 76]

​                                                    ===》 layers[5] = [3, 256, 76, 152]   

​                                                   ===》  layers[5] = [3, 256, 76, 152]  +  layers[4] =  [3, 256, 76, 152]   

​												    ===》 layers[5] = [3, 256, 76, 152]

​         【64,128, 256, 512】 ==》 【64,128, 256, 256】

​     **（2） 对layers[4]和layers[5]特征依次进行更新：**首先对layers[4]进行通道数减半256->128、图片大小上采样2倍 1/16->1/8转换（依次使用proj(256 128)、upsample(128，128,  f=2) ），然后与同样通道数和尺寸layers[3]相加进行融合（node(128,128)) **得到更新的layers[4]**；然后对layers[5]进行通道数减半256->128、图片大小上采样2倍 1/16->1/18的转换（依次使用proj(256, 128)、upsample(128，128,  f=2)），然后与同样通道数和尺寸layers[4]相加进行融合（node(128,128)) **得到更新的layers[5]**

​		layers[4] = [3, 256, 76、152]  ===》 layers[4] = [3, 128,  76、152]

 							 ===》 layers[4] = [3, 128, 152, 304]

​														   ===》 layers[4] = [3, 128, 152, 304]  +  layers[3] = [3, 128，152，304]

​														   ===》 layers[4] = [3, 128, 152, 304]

​        layers[5] = [3, 256, 76, 152] ===》 layers[5] = [3, 128, 76, 152]

​                                                    ===》 layers[5] = [3, 128, 152,  304]   

​                                                   ===》  layers[5] = [3, 128, 152,  304]  +  layers[4] =  [3, 128, 152, 304]   

​												    ===》 layers[5] = [3, 128, 152,  304]

​       【64,128, 256, 256】 ==》 【64,128, 128, 256】==》 【64,128, 128, 128】

​     **（3）对layers[3]、layers[4]和layers[5]特征依次进行更新：方法同上**

​         【64,128, 128, 128】==》 【64,64, 128, 128】==》 【64, 64，64, 128】 ==》 【64, 64，64, 64】



【4】out数组是依次往0位置处追加元素的，所以其依次追加了的layer[5]通道数和图片大小依次为（256，1/16）、(128, 1/8)、

（64，1/4)。最终的out数组为【（64，1/4)，(128, 1/8)，(256，1/16），（512，1/32)】

#### 2.2.2.4  再使用一层IDA_UP

```
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))
```

 【1】首先这里的last_level等于5,  first_level=2, 代码中的y只存储了out中的前3个元素, 也就是layers[2]  layers[3] layers[4]的结果,

 y= [   [3, 64，304，608], [3, 128, 152，304]、[3, 256, 76、152]   ]

【2】 再使用一层IDA_UP网络进行处理即可, 这里需要注意一下构造IDA_UP网络的3个参数,  这里的y对应的是不同通道数和图片尺寸的图片特征数组,  0表示进行IDA_UP操作的起点  startp ,  len(y)表示IDA_UP操作的终点endp(终点位置不进行操作, 终点前一个位置可操作).

 y= [   [3, 64，304，608], [3, 128, 152，304]、[3, 256, 76、152]   ]     =====>

​                      startp= 0               startp + 1 = 1                                          endp = 3

 y= [   [3, 64，304，608], [3, 64，304，608]、[3, 256, 76、152]   ]    =====>

y=  [   [3, 64，304，608], [3, 64，304，608]、[3, 64，304，608]   ] 

​       IDA_UP操作是从startp + 1位置开始计算的,  然后每个位置index首先进行1次proj操作将通道数转换为前1个位置的通道数目,  再进行1次upsample操作将图片大小转化为前1个位置对应的图片大小, 最后将 这两步骤得到的更新的y[index] 与前1个位置的y[index-1]进行混合,得到 当前位置更新后的y[index], 整个过程可以用数学表达为: y[index] =  node(  upsmaple(proj( y[index] ))   +   y[index-1] )

​      从startp + 1位置到 endp - 1位置都进行同样类似的操作。

```python
# 1.IDAUp包含channels-1个层
# 每层都是由两个变形卷积层和一个上采样的反卷积组成
# 2.变形卷积层中一个将特征从高纬转化为低纬，即c->o
#   一个的特征维度始终保持o不变 o->o
#   上采样本的反卷积层同样保持通道数不变 o->o
class IDAUp(nn.Module): # 64  [64, 128, 256]  [1, 2, 4]
    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]  # [512]
            f = int(up_f[i]) # [2]
            proj = DeformConv(c, o) # [512,256]  
            node = DeformConv(o, o) # [256, 256]   
            # 这里的反卷积对应的图片大小到底是怎么变化的呢？
            # 如果按照卷积的公式，则图片大小不变
            # 而反卷积的计算公式为:output=(input-1) * stride + output_padding - 2 * padding + kernel_size
            # 则这里是 (x-1) * f + 0 - 2 * (f // 2) + 2 * f = x  * f
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        # 先把每层的特征利用变形卷积从c映射为o，再利用反卷积上采样图像f倍，
        # 最后将前一层特征和本层特征相加后进行一次通道数不变的变形卷积操作
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])
```

#### 2.2.2.4 两类预测head

​         预测act_hm ，即每个对象中心的热力图，shape为（3，8，1216/4， 2432/4 ）；

​        预测awh, 即每个中心点对应的宽高， width为（3，2，1216/4，2432/4）。

​         注意，这里可以发现预测head回归的两种特征图的大小与1/4 大小一致，其中的8对应预测所涉及到的所有类别数目，2分别对应宽和高两个值。

#### 2.2.2.5 DLASEG网络的返回结果

​       返回的output是一个字典，主要包含了两类预测head的预测结果；返回的cnn_feature对应于y[-1],  是被更新多次的layers[4]特征，shape为（3，64，1216/4，2432/4)



### 2.2 检测框的解码

#### 2.2.1 主要作用

​         主要是对上一步得到的两类预测head进行进一步的解码处理，其解码包含两步，首先解码热力图和宽高图，得到每张图片的K个中心点以及检测框信息；其次对检测框的坐标进行越界的修正处理。

```python
decode_detection(output, cnn_feature.size(2), cnn_feature.size(3))
```

```python
 def decode_detection(self, output, h, w):
        ct_hm = output['act_hm']
        wh = output['awh']
        # 为什么事先需要对热图使用sigmoid呢
        ct, detection = snake_decode.decode_ct_hm(torch.sigmoid(ct_hm), wh)
        detection[..., :4] = data_utils.clip_to_image(detection[..., :4], h, w)
        output.update({'ct': ct, 'detection': detection})
        return ct, detection
```

#### 2.2.2  snake_decode.decode_ct_hm   解码热力图和宽高图

 【1】：通过对热力图进行nms处理以及得分排序操作，得到每张图片对应的K个中心点的坐标信息、类别信息、得分信息

【2】：结合回归的宽高图，得到中心点对应的bbox信息

```python
def decode_ct_hm(ct_hm, wh, reg=None, K=100):
    batch, cat, height, width = ct_hm.size()
    ct_hm = nms(ct_hm)

    scores, inds, clses, ys, xs = topk(ct_hm, K=K)
    wh = transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)

    if reg is not None:
        reg = transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1)
        ys = ys.view(batch, K, 1)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    ct = torch.cat([xs, ys], dim=2)  # 这里的ct指的是每个物体中心点的坐标 （3，100，2）
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    detection = torch.cat([bboxes, scores, clses], dim=2) # （3，100, 6)  
    # 每个中心点对应的检测框的左上角和右下角坐标顶点、中心点对应的预测得分、中心点对应的预测类别
    return ct, detection
```

（1）输入： 将DLASEG网络回归得到的ct_hm通过sigmoid函数转化，使得值的分布在0-1之间，然后作为解码函数输入中的ct_hm；

（2）首先进行nms处理：这里使用的（2，2）池化层能够找到每个局部的最大值，设置了相应大小的pad能够使得图片的大小保持不变；hmax == heat能够找到图片中那些是局部最大值的像素点位置；heat * keep保留了回归热力图中代表局部最大值的像素点的热力值，同时将那些不是局部最大值的像素点的热力值也都设定为了0，返回的结果也可以看作是图片上每个像素点在每个类别上（0-7）上的得分。

```
def nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep
```

（3）对热力图中的得分进行排序并找到K个最大值来作为最终每张图片中检测到的中心点：

​             首先在8个类别通道上的每个通道层面对应的2维像素矩阵上，我们找到其中得分排名前K的像素点位置（这里的位置从2个角度来刻画，一是使用在1维展开的2维矩阵的下标来表示 topk_inds， 二是从2维矩阵的横向坐标topk_xs和纵向坐标topk_ys来表示）、像素点得分topk_scores（对应于topk_inds的排列方式）；然后从这8个类别对应的8 * K个像素点的总集合中找到得分排名前K的像素点，这K个像素点就作为每张图片的中心点，我们需要记录这K个点的得分topk_score、类别topk_clses、1维展开中的像素点位置topk_inds、像素点的横向坐标位置topk_xs、像素点的纵向坐标位置topk_ys，这些信息的维度均为（batch_size, K)。

```python
# 对经过nms处理后的heatmap也就是scores

def topk(scores, K=40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K) # （3，8， 100）

    topk_inds = topk_inds % (height * width) # （3，8， 100）
    topk_ys = (topk_inds / width).int().float()  # 图片矩阵中第几行
    topk_xs = (topk_inds % width).int().float()  # 图片矩阵中第几列

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K) # （3，800）
    topk_clses = (topk_ind / K).int()
    #  （3，100，1） =》 （3，100）
    #  对（3，800，1）进行（3，100，1）维度上的index，
    #  返回的是8 * 100 个值中前100个最大值的原始图片像素的索引index
    #  需要进行这一步转换的原因是通过在（3，800）这种形式的矩阵里面获取前100个元素
    #     得到的结果的索引是0-799
    #    而我们需要的是其在原始图片大小中对应的索引
    topk_inds = gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
    #   （3，100）、（3，100）、（3，100）、（3，100）、（3，100）
```

（4）通过在回归出的宽高awh特征图上进行索引，得到每张图片对应的K个中心点对应的检测框的宽度和高度： 

```python
# 【3，800，1】 【3，100】
def gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim) # （300，50）
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

# （300，2，14，56） （300，50）
def transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()  
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat
```

(5) 通过中心点的横向坐标位置topk_xs、像素点的纵向坐标位置topk_ys得到每张图片对应的K个中心点的坐标表示（batch, K, 2)：

​              即将两者进行一个维度的拼接。

```
 ct = torch.cat([xs, ys], dim=2) 
```

(6) 通过中心点的坐标表示以及前面获取的bbox宽高，得到每个中心点对应bbox的左上角和右下角坐标的四点表示：

```
 bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
```

（7）综合每张图片对应的K个中心点的bbox四点坐标表示、预测得分、预测类别，得到解码结果之detection(batch, K, 6):

```
 detection = torch.cat([bboxes, scores, clses], dim=2) # （3，100, 6)
```

#### 2.2.3  data_utils.clip_to_image 对中心点的bbox坐标信息进行越界处理

```
# 对检测框的四个坐标顶点进行裁剪，这是因为宽高、以及中心点的位置都是预测出来的，所以有可能在尺度上是不合理的
def clip_to_image(bbox, h, w):
    bbox[..., :2] = torch.clamp(bbox[..., :2], min=0) # 对边框左上角的横纵坐标约定最小值为0
    bbox[..., 2] = torch.clamp(bbox[..., 2], max=w-1) # 对边框右下角的横坐标约定为最大值为w-1
    bbox[..., 3] = torch.clamp(bbox[..., 3], max=h-1) # 对边框右下角的纵坐标约定为最大值为h-1
    return bbox
```

#### 2.2.4 返回结果

往output里面增添了两个键值，ct （batch,K,2)代表中心点坐标， detection （batch,K,6)代表检测框信息。



### 2.3 ComponentDetection组件部分

#### 2.3.1 主要作用

```
output = self.cp(output, cnn_feature, batch)
```

#### 2.3.2  核心网络结构

```
class ComponentDetection(nn.Module):
    def __init__(self):
        super(ComponentDetection, self).__init__()
         
        # （7.28）
        # ROIAlign的作用和相关参数的具体含义是什么呢  7，28
        # ROIAligh就是从同一特征图上不同尺寸的ROI得到相同尺寸的特征表示，
        # 不需要进行ROIPool的两次量化，只需按照输出尺寸的要求进行等大小的区域划分然后再每个区域使用双线性插值即可
        # 所有ROI对齐（7，28）的尺寸
        self.pooler = ROIAlign((rcnn_snake_config.roi_h, rcnn_snake_config.roi_w))

        # 多层卷积堆叠的融合模块
        # 由于stride=1，所以每个卷积都只是改变了通道数，并未改变图片大小
        # 第一层卷积进行通道数增加，其他层卷积相当于只进行了特征的整合
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.heads = {'cp_hm': 1, 'cp_wh': 2}
        # 两类头 分别用于热力图和宽高任务
        # 头结构都是由一层上采样和一层卷积组成, 
        # 这里的上采样将图像扩大为2倍，只是通道数256-》256-》classes
        for head in self.heads:
            # head为字典的关键字，而不是 关键字 + 值
            classes = self.heads[head]
            fc = nn.Sequential(
                # y = (x-1) * 2 + 0 - 2*0 + 2 = 2x
                nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                nn.Conv2d(256, classes, kernel_size=1, stride=1)
            )
            # hm对应的偏置值填复值，wh的偏置值填写0
            # 疑惑：为什么这里偏偏要对偏置进行初始化呢？
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)
```

#### 2.3.3 整个执行流程

```
    def forward(self, output, cnn_feature, batch=None):
        z = {}
        # 用于训练目的
        if batch is not None and 'test' not in batch['meta']:
            roi = self.prepare_training(cnn_feature, output, batch)
            roi = self.fusion(roi)
            for head in self.heads:
                z[head] = self.__getattr__(head)(roi)
        # 用于测试目的
        if not self.training:
            with torch.no_grad():
                roi = self.prepare_testing(cnn_feature, output) # torch.Size([300, 64, 7, 28])
                roi = self.fusion(roi) # torch.Size([300, 256, 7, 28])
                cp_hm = self.cp_hm(roi) # torch.Size([300, 1, 14, 56])
                cp_wh = self.cp_wh(roi) # torch.Size([300, 2, 14, 56])
                self.decode_cp_detection(cp_hm, cp_wh, output)

        output.update(z)

        return output
```

#### 2.3.3.1 测试过程的核心步骤

（1）通过原始的图片特征以及中心点信息，获取所有图片中所有中心点对应的固定大小的ROI特征图

```
 roi = self.prepare_testing(cnn_feature, output) # torch.Size([300, 64, 7, 28])
```

```
def prepare_testing(self, cnn_feature, output):
        if rcnn_snake_config.nms_ct: # 如果使用nms来进行处理的话
            detection, ind = self.nms_abox(output)
        else:
            ind = output['detection'][..., 4] > rcnn_snake_config.ct_score
            detection = output['detection'][ind]
            ind = torch.cat([torch.full([ind[i].sum()], i) for i in range(len(ind))], dim=0)

        ind = ind.to(cnn_feature.device) # 300 torch.Size([300, 5])
        abox = detection[:, :4]
        roi = torch.cat([ind[:, None], abox], dim=1) # roi对应batch个图片中的所有中心点  其中第1列对应中心点所属的图片、后4列对应中心点的box
        # (3,64,304,608)  (300, 5)  ROI池化的输入是批量的图片特征（只不过每个图片像素点的特征维度是64） 所有中心点对应的边框信息
        roi = self.pooler(cnn_feature, roi) # torch.Size([300, 64, 7, 28])
        output.update({'detection': detection, 'roi_ind': ind})

        return roi
```

【1】：nms_abox对批量的图片进行处理。首先对于其中的每一张图片我们都通过nms_class_box函数进行处理，利用list(zip(*box_score_cls_))将得到类似于【（box_m,box_n，box_t,... ）（score_m，score_n, score_t, ...）,（ label_m, label_n, label_t, ...）】这样一个列表，然后将这个列表中的每个元素在第0维度进行拼接，就能得到按照类别序号排序好的关于该图片所有中心点的【box、score、label】；将所有图片的【box、score、label】都存储进box_score_cls列表中，就能得到一个形如【【box_img1、score_img1、label_img1】、【box_img2、score_img2、label_img2】、【box_img3、score_img3、label_img3】、...】的2维列表，再利用 *list(zip(  *box_score_cls))得到【（box_img1,box_img2, ...)、（score_img1、score_img2、...)、（label_img1、label_img2、...)】，再将这个列表中的每个元素在第0维度进行拼接，就能得到按照图片序号排序好的所有中心点的box、score、cls（shape分别为（filter(batch * K)，4）、（filter(batch * K)，1）、（filter(batch * K)，1）），进行拼接后得到detection检测框（filter(batch * K)，6）并返回。注意这里还返回了一个ind，其与（filter(batch * K)，6）中的filter(batch * K)相对应，用来记录每张图片经过nms过滤后剩下的中心点检测框的个数, 这里的filter表示经过nms过滤处理后的中心点个数。

```
def nms_abox(self, output):
        box = output['detection'][..., :4] # 中心点检测框的四个点坐标 （3，100，4）
        score = output['detection'][..., 4]  # 中心点的得分 （3，100）
        cls = output['detection'][..., 5]  # 中心点的类别 （3，100）

        batch_size = box.size(0) # 3
        cls_num = output['act_hm'].size(1) # 8

        box_score_cls = []
        # 这里是按照每一张图片来单独依次处理的
        for i in range(batch_size):
            # 一句话总结nms_class_box函数的作用
            # 函数的输入是该图片对应的100个中心点的检测框的4个点坐标、
            # 100个中心点的得分
            # 100个中心点的类别
            # 以及类别总数
            box_score_cls_ = self.nms_class_box(box[i], score[i], cls[i], cls_num) # 对于中心点检测框进行了一个nms过滤，并且中心点按照类进行排序
            box_score_cls_ = [torch.cat(d, dim=0) for d in list(zip(*box_score_cls_))] # 在0维按照类的方向拼接起来
            box_score_cls.append(box_score_cls_)

        box, score, cls = list(zip(*box_score_cls)) # 【(（100，4）, （100，4）, （100，4）)】 【（100） × 3】 【 （100）× 3】
        ind = torch.cat([torch.full([len(box[i])], i) for i in range(len(box))], dim=0) # （300） 100个0、100个1、100个2
        box = torch.cat(box, dim=0) # torch.Size([300, 4]) 
       
        # score = torch.stack(score, dim=1) # torch.Size([100, 3])
        # cls = torch.stack(cls, dim=1) # torch.Size([100, 3])
        
        score = torch.cat(score, dim=0).unsqueeze(1) # torch.Size([300, 1]) 
        cls = torch.cat(cls, dim=0).unsqueeze(1) # torch.Size([300, 1]) 
        # 这里很纳闷 维度不统一要怎么直接相连接
        # 可是之前直接使用cuda运行的时候结果都能运行得出来  没有任何问题啊
        detection = torch.cat([box, score, cls], dim=1)  # # torch.Size([300, 6]) 
        # 包含每张图片中所有中心点的box左上角和右下角坐标、中心点对应的得分、中心点的类别 
        return detection, ind
```

nms_class_box函数主要是对1张图片中包含的所有中心点检测框bbox进行处理，主要步骤是首先按照类别的编号依次找到该图片中该类别对应的所有中心点检测框bbox，然后通过设定的rcnn_snake_config.max_ct_overlap（最大重叠的IOU阈值）去除掉重复多余、通过设定的 rcnn_snake_config.ct_score（最低置信度）来对bbox进行过滤，最后结合bbox的预测得分、类别标签信息得到该图片该类别的有效bbox的信息表示[box_, score_, label_]。最后返回的box_score_cls应该是1个2维数组，其中的每个元素是关于某个类别的1个1维数组[box_, score_, label_]。

```
# 一张图片对应的100个中心点对应的box的左上角和右下角坐标、100个中心点对应的得分以及类别
    def nms_class_box(self, box, score, cls, cls_num): 
        box_score_cls = [] # 应该是按照中心点的类别对这100个中心点进行一个划分
                           
        for j in range(cls_num):
            ind = (cls == j).nonzero().view(-1) # 使用nonzero得到的是1个2维数组，使用view(-1)可以直接将其转化为1维数组
            if len(ind) == 0:
                continue

            box_ = box[ind]   # 当前类别的所有中心点检测框对应的左上角和右下角坐标
            score_ = score[ind]  # 当前类别的所有中心点对应的预测得分
            # 下面这个调用命令必须要在GPU上运行
            # ind = _ext.nms(box_, score_, rcnn_snake_config.max_ct_overlap)
            # ind = ind.cpu()
            
            # 自己重写上述功能，其实是错误的写法，只是为了方便cpu顺利执行
            # 首先计算box_中的总数  这可以通过获取ind的长度得知
            # 然后按顺序依次判断对应的score是否大于rcnn_snake_config.max_ct_overlap
            new_idx = []
            for i in range(len(ind)):
                if score_[i] < rcnn_snake_config.max_ct_overlap:
                    new_idx.append(i)
            ind = torch.LongTensor(new_idx)
            
            box_ = box_[ind]
            score_ = score_[ind]

            ind = score_ > rcnn_snake_config.ct_score
            box_ = box_[ind] # （14，4）
            score_ = score_[ind] # （14）
            label_ = torch.full([len(box_)], j).to(box_.device).float() # （14）
            # 每个类别所对应的所有中心点的边框左上角和右下角坐标点、对应的得分、对应的列表标签
            # 需要注意这里理论上应该是【【box、score、label】，【box、score、label】，
            # 【box、score、label】，【box、score、label】】
            box_score_cls.append([box_, score_, label_]) 

        return box_score_cls
```

【2】：将图片特征cnn_feature和roi（ 形状为（filter(batch * K)，5），其中（filter(batch * K)，0）是中心点的图片序号，filter(batch * K)，1：5）是中心点的bbox） 作为C++拓展的函数ROI_Align的输入，得到所有图片中所有中心点对应的固定大小的ROI特征图

（filter(batch * K)，64，7，28）。



（2）对ROI进行特征融合，并获取对应的组件热力图和宽高图表示

特征融合维度从64变成256；

其中self.cp_hm和self.cp_wh两个预测head除了变换维度到1和2，还都上采样了2倍

```
 roi = self.fusion(roi) # torch.Size([300, 256, 7, 28])
 cp_hm = self.cp_hm(roi) # torch.Size([300, 1, 14, 56])
 cp_wh = self.cp_wh(roi) # torch.Size([300, 2, 14, 56])
```



（3）对组件热力图和宽高图表示进行解码

这里需要明确组件热力图torch.sigmoid(cp_hm)和abox的大小关系和区别，这里面每个cp_hm的shape为（1, 14, 56），而这里的abox来自于output['detection'] [..., :4]，adet来自output['detection']，而在nms_abox(self, output)函数中box、score、cls这三个output键值的引用被更新了，abox此时的形状为（filter(batch * K)，4）、adet此时的形状为（filter(batch * K)，6），对应的图片大小仍然是（

1216/4，2432/4）。

```
 box = output['detection'][..., :4] # 中心点检测框的四个点坐标 （3，100，4）
 score = output['detection'][..., 4]  # 中心点的得分 （3，100）
 cls = output['detection'][..., 5]  # 中心点的类别 （3，100）
        
  box = torch.cat(box, dim=0) # torch.Size([300, 4]) 
       
 # score = torch.stack(score, dim=1) # torch.Size([100, 3])
 # cls = torch.stack(cls, dim=1) # torch.Size([100, 3])

score = torch.cat(score, dim=0).unsqueeze(1) # torch.Size([300, 1]) 
cls = torch.cat(cls, dim=0).unsqueeze(1) # torch.Size([300, 1]) 
```

这里得到的组件热力图为（filter(batch * K)，1，14, 56），对应的类别数目只有1。这里的decode_cp_detection函数是解码的关键步骤。首先在每个filter(batch * K)个中心点对应的组件热力图上找到得分排名前M内（此处M=rcnn_snake_config.max_cp_det）的M个像素点的信息，并找到这些像素点所对应的宽高图上的宽和高，然后根据这M个像素点在cp_hm即（14，56）中的坐标偏移程度来计算这些像素点隐射到原图大小（1216/4，2432/4）中的位置xs和ys，然后根据回归的宽高图得到在原图像上的M个检测框bbox, 这M个bbox相当于是原来图像上某个中心点的M个候选检测框。boxes形状为（filter(batch * K)， M， 4）。将类别1和类别2中的第0个候选框设置为原始中心点对应的检测框，并将类别1和类别2中M个候选框中的第0个候选框得分设定为1，其余(M-1)个候选框的得分都设定为0，其实也就是在说这两个类别无需使用多个候选框来进行处理。然后我们对filter(batch * K)个中心点中的每个中心点对应的M个候选框进行处理，使用

rcnn_snake_config.max_cp_overlap进行去重过滤、rcnn_snake_config.cp_score进行置信度过滤，通常是这样处理之后每个中心点对应的候选框就只剩下一个或者0个，其中boxes [i] [cp_ind] [cp_01]形状为【【A,B,C,D】】或者 【【】】的形状，boxes_则是包含上述形状的数组，cp_ind对应原始每个中心点经过删选后剩下的box框的个数，其中为【【】】的会被自动跳过，如下图所示，最后的boxes对boxes__ 进行了0维上的拼接，其形状为（NUM, 4），NUM是所有图片所有中心点经过筛选后剩余的box的坐标：



cp_box键对应的值就是上述的boxes（NUM,4），

cp_ind键对应的值就是每个中心点所对应的筛选过后的检测框的数目，比如说cp_ind =【  【0，0，0】，【2，2】，【5，5，5，5】】,中心点的总数假设为6，也就是在说0号中心点还存在3个筛选过后的剩余检测框，2号中心点还存在2个筛选过后的剩余检测框，5号中心点还存在4个筛选过后的剩余检测框，而其余的1号、3号、4号中心点经过筛选之后已经没有剩余任何检测框了；

输出字典output中有1个很容易混淆的键为roi_ind，其在函数prepare_testing---》nms_abox中取得，其对应于不同图片经过nms过滤之后剩余的中心点的个数，比如说roi_ind = 【 【0，0，0，..., 0】(共计20个0)、【1，1，1，..., 1】（共计30个1）、【2，2，2，..., 2】（共计40个2）】,  这个也就是说0号图片经过nms过滤之后还保留有20个中心点、1号图片经过nms过滤之后还保留有30个中心点，2号图片经过nms过滤之后还保留有40个中心点。

roi_ind是中心点与图片序号的对应关系，cp_ind是所剩余的所有中心点候选框与中心点的实际编号的对应关系，cp_ind通常来说其包含的中心点是要少于roi_ind所包含的中心点的个数的。



疑惑：获取的cp_box即检测框到底是在哪一个图片尺度上呢？ （1/4,1/4)

[image-20210530114411908](C:\Users\levin\AppData\Roaming\Typora\typora-user-images\image-20210530114411908.png)

```
 self.decode_cp_detection(cp_hm, cp_wh, output)
```

```
 def decode_cp_detection(self, cp_hm, cp_wh, output):
        abox = output['detection'][..., :4]  # torch.Size([300, 4])
        adet = output['detection']  # torch.Size([300, 6])
        ind = output['roi_ind']   # torch.Size([300])
        box, cp_ind = rcnn_snake_utils.decode_cp_detection(torch.sigmoid(cp_hm), cp_wh, abox, adet)
        output.update({'cp_box': box, 'cp_ind': cp_ind})
```

```python
def decode_cp_detection(cp_hm, cp_wh, abox, adet):
    batch, cat, height, width = cp_hm.size()
    if rcnn_snake_config.cp_hm_nms:
        cp_hm = nms(cp_hm)

    abox_w, abox_h = abox[..., 2] - abox[..., 0], abox[..., 3] - abox[..., 1]

    scores, inds, clses, ys, xs = topk(cp_hm, rcnn_snake_config.max_cp_det) # （300，50）
    cp_wh = transpose_and_gather_feat(cp_wh, inds) # torch.Size([300, 50, 2])
    cp_wh = cp_wh.view(batch, rcnn_snake_config.max_cp_det, 2)

    cp_hm_h, cp_hm_w = cp_hm.size(2), cp_hm.size(3) # 14，56

    xs = xs / cp_hm_w * abox_w[..., None] + abox[:, 0:1]  # (300, 50)
    ys = ys / cp_hm_h * abox_h[..., None] + abox[:, 1:2]  # (300, 50)
    boxes = torch.stack([xs - cp_wh[..., 0] / 2,
                         ys - cp_wh[..., 1] / 2,
                         xs + cp_wh[..., 0] / 2,
                         ys + cp_wh[..., 1] / 2], dim=2) # (300, 50, 4)

    ascore = adet[..., 4]  # [300]
    acls = adet[..., 5]    # [300]
    excluded_clses = [1, 2] # 这里要排除在外的类是1和2
    for cls_ in excluded_clses:
        boxes[acls == cls_, 0] = abox[acls == cls_]
        scores[acls == cls_, 0] = 1
        scores[acls == cls_, 1:] = 0

    ct_num = len(abox) # 300
    boxes_ = []  
    for i in range(ct_num): # 对每1个中心点进行处理
        # cp_ind = _ext.nms(boxes[i], scores[i], rcnn_snake_config.max_cp_overlap)
        
        # 上面的写法必须使用gpu,这是自己改的，只是为了不报错且跳过上述函数，实现并不正确
        cp_ind = torch.arange(len(scores[i]))
        # for index in range(len(boxes[i])):
        #     if scores[i] <= rcnn_snake_config.max_cp_overlap:
        #         cp_ind.append(index)
        cp_ind = torch.LongTensor(cp_ind)
        
        cp_01 = scores[i][cp_ind] > rcnn_snake_config.cp_score
        boxes_.append(boxes[i][cp_ind][cp_01])

    cp_ind = torch.cat([torch.full([len(boxes_[i])], i) for i in range(len(boxes_))], dim=0)
    cp_ind = cp_ind.to(boxes.device)
    boxes = torch.cat(boxes_, dim=0)

    return boxes, cp_ind   # [171,4]   [171]
```

#### 2.3.3.2 训练过程的核心步骤



### 2.4  Evolution()

#### 2.4.1 主要作用

该模块主要是为了利用上一函数组件获得的所有图片对应的中心点及其bbox信息，来进行对象轮廓多边形的初始化以及演进工作。

```python
# 模型分为三个核心模块
# 第一个模块是作为初始gcn的Snake
# 第二个模块是作为演变gcn的snake
# 第三个模块是evolve_gcn_0, evolve_gcn_1
# 三个snake的特征维度都是66，中间维度都是128
# 可以看出来模型其实完全就是由很多个Snake模型块结构组合而成的
# 只是包含的SNake的个数是不一定的
# 起码包含两个Snake模型块，一个作为初始化，一个作为演变
# 剩余Snake模块的个数完全由迭代次数来决定
class Evolution(nn.Module):
    def __init__(self):
        super(Evolution, self).__init__()

        self.fuse = nn.Conv1d(128, 64, 1)
        self.init_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='dgrid')
        self.evolve_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='dgrid')
        self.iter = 2
        for i in range(self.iter):
            evolve_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='dgrid')
            self.__setattr__('evolve_gcn'+str(i), evolve_gcn)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
```

#### 2.4.2 主要执行过程

#### 2.4.2.1 执行代码

```python
def forward(self, output, cnn_feature, batch=None):
        ret = output  # ret主要用于训练中，在测试中没有用到

        if batch is not None and 'test' not in batch['meta']:
            with torch.no_grad():
                init = self.prepare_training(output, batch)

            ex_pred = self.init_poly(self.init_gcn, cnn_feature, init['i_it_4py'], init['c_it_4py'], init['4py_ind'])
            ret.update({'ex_pred': ex_pred, 'i_gt_4py': output['i_gt_4py']})

            py_pred = self.evolve_poly(self.evolve_gcn, cnn_feature, init['i_it_py'], init['c_it_py'], init['py_ind'])
            py_preds = [py_pred]
            for i in range(self.iter):
                py_pred = py_pred / snake_config.ro
                c_py_pred = snake_gcn_utils.img_poly_to_can_poly(py_pred)
                evolve_gcn = self.__getattr__('evolve_gcn'+str(i))
                py_pred = self.evolve_poly(evolve_gcn, cnn_feature, py_pred, c_py_pred, init['py_ind'])
                py_preds.append(py_pred)
            ret.update({'py_pred': py_preds, 'i_gt_py': output['i_gt_py'] * snake_config.ro})

        if not self.training:
            with torch.no_grad():
                init = self.prepare_testing_init(output)
                ex = self.init_poly(self.init_gcn, cnn_feature, init['i_it_4py'], init['c_it_4py'], init['ind']) # torch.Size([240, 16, 2])
                ret.update({'ex': ex})

                evolve = self.prepare_testing_evolve(output, cnn_feature.size(2), cnn_feature.size(3))
                py = self.evolve_poly(self.evolve_gcn, cnn_feature, evolve['i_it_py'], evolve['c_it_py'], init['ind']) # torch.Size([243, 120, 2])
                pys = [py / snake_config.ro]
                # 应该是通过多次迭代来演化多边形顶点坐标
                for i in range(self.iter):
                    py = py / snake_config.ro  # torch.Size([240, 120, 2])
                    c_py = snake_gcn_utils.img_poly_to_can_poly(py) # torch.Size([240, 120, 2])
                    evolve_gcn = self.__getattr__('evolve_gcn'+str(i))
                    py = self.evolve_poly(evolve_gcn, cnn_feature, py, c_py, init['ind']) # torch.Size([240, 120, 2])
                    pys.append(py / snake_config.ro)
                ret.update({'py': pys})

        return output
```

#### 2.4.2.2 测试过程中的具体步骤

【1】：**测试的准备工作prepare_testing_init， 主要是获取四菱形顶点并进行采样**。

```
init = self.prepare_testing_init(output)
```

```
def prepare_testing_init(self, output): # output['cp_box'] torch.Size([282, 4])
        i_it_4py = snake_decode.get_init(output['cp_box'][None]) # torch.Size([1, 282, 4, 2])
        # i_it_4py = snake_gcn_utils.uniform_upsample(i_it_4py, snake_config.init_poly_num) # torch.Size([1, 282, 40, 2])
        i_it_4py = i_it_4py.repeat(1,1,snake_config.init_poly_num//4,1)
        c_it_4py = snake_gcn_utils.img_poly_to_can_poly(i_it_4py) # torch.Size([1, 282, 160, 2])

        i_it_4py = i_it_4py[0]  # torch.Size([282, 160, 2])
        c_it_4py = c_it_4py[0]  # torch.Size([282, 160, 2])
        ind = output['roi_ind'][output['cp_ind'].long()]  # 计算每个中心点所对应的图片编号
        init = {'i_it_4py': i_it_4py, 'c_it_4py': c_it_4py, 'ind': ind}
        output.update({'it_ex': init['i_it_4py']}) # torch.Size([282, 160, 2]) 名称含义貌似图片初始的值点

        return init
```

```
def get_init(box):
    if snake_config.init == 'quadrangle':
        return get_quadrangle(box)
    else:
        return get_box(box)
```

```
def get_quadrangle(box):
    x_min, y_min, x_max, y_max = box[..., 0], box[..., 1], box[..., 2], box[..., 3]  # torch.Size([1, 282])
    quadrangle = [   # torch.Size([1, 282])
        (x_min + x_max) / 2., y_min,
        x_min, (y_min + y_max) / 2.,
        (x_min + x_max) / 2., y_max,
        x_max, (y_min + y_max) / 2.
    ]
    quadrangle = torch.stack(quadrangle, dim=2).view(x_min.size(0), x_min.size(1), 4, 2) # torch.Size([1, 282, 8])===> torch.Size([1, 282, 4, 2])
    return quadrangle
```

​       output是一个用于存储各个组件调用后相关结果的字典，一方面可以很方便地为后续组件所使用，另一方面使用字典来作为该网络模型的输出结果，既可以保存尽可能多的中间结果，也方便通过键值直接获取中间结果（从总体上来看，也就是达到了通过1个变量获取多个变量值的效果。）

​       output['cp_box']对应形状为（M,4），也就是说经过前两个组件筛选后所有图片对应的所有中心点对应的目标检测框的四点坐标表示（这个四点坐标是由模型预测出来的值，虽然说一般的bbox是由左上角和右下角的两点来表示的，但是这里的四点实际上也有可能是左下角和右上角的两点）。我们首先按照  上中==》 中左 == 》下中 == 》中右 的顺序获取每个bbox的四条边上的中点，也可以是说获取四菱形，得到 i_it_4py （1,M,4,2); 然后我们对四菱形的4边进行点采样，使得采样后的总顶点个数为snake_config.init_poly_num = 160，具体的采样方式主要是依据四菱形的4条边的长度来分配采样点的个数，得到 i_it_4py （1, M, snake_config.init_poly_num, 2); 最后归一化得到 c_it_4py （1, M, snake_config.init_poly_num, 2)；最后我们去点不必要的0维，即这里的1，得到init = {'i_it_4py': i_it_4py, 'c_it_4py': c_it_4py, 'ind': ind}，ind是M个中心点所对应的实际图片编号 （M,) 。

​      这一步返回了init字典，并在output中添加键{'it_ex': init['i_it_4py']} 来表示M个中心点所对应的初始多边形

​    （M, snake_config.init_poly_num, 2）。

【2】：**init_poly使用一层Snake网络从采样后的4菱形演变得到初始的多边形轮廓。**主要是首先从提取的图片特征中去获取多边形顶点对应的特征图，然后利用多边形顶点的特征作为Snake模型的输入得到多边形上每个顶点的预测坐标偏移量，最后使用这些偏移量来修正多边形顶点的坐标。**将4点菱形进行变形后得到的新的4个点作为ex键存储在output中**，这里并没有保存归一化的顶点坐标。

```
 ex = self.init_poly(self.init_gcn, cnn_feature, init['i_it_4py'], init['c_it_4py'], init['ind']) # torch.Size([240, 16, 2])
 ret.update({'ex': ex})  # 这里的ret是output的1个引用，改变ret同样会改变output
```

​          注意这里获取多边形顶点特征的方式get_gcn_feature，首先我们通过torch.nn.functional.grid_sample这个双线性采样函数来获取顶点的初步特征，我们首先需要将多边形顶点的坐标值转换到【-1，1】上，然后我们通过线性坐标映射关系可以将转化后的每个多边形顶点坐标映射到原始图片的特征图坐标范围内，通常得到的是浮点值（X,Y)，最后我们在原始图片特征图坐标空间内使用双线性插值得到多边形顶点(X,Y)的特征表示；然后我们找到多边形顶点中横坐标的最小值x_min和最大值x_max, 纵坐标的最小值y_min和最大值y_max，我们将（ （x_min+x_max）/2， （y_min+y_max)/2 ) 作为每个多边形的中心点坐标，并同样使用grid_sample方法来获取每个多边形中心点的特征表示；然后我们将多边形顶点的特征表示和其对应中心点的特征表示拼接起来，并使用融合层fuse进行融合，得到更新后的多边形顶点特征表示；最后我们将多边形顶点的特征表示和每个顶点的归一化坐标连接起来，作为每个多边形顶点的最终特征表示init_input。

​         Snake模型主要是利用环状图上的循环卷积来更新每个多边形顶点的特征表示，并使用这些特征来预测每个多边形顶点的横纵坐标偏移。因为循环卷积需要知道环状图上每个顶点左右adj_num=4个邻居，如果我们直接按照顶点编号的顺序来处理的话，第0号顶点的左边邻居为【snake_config.init_poly_num-4，snake_config.init_poly_num-3，snake_config.init_poly_num-2，snake_config.init_poly_num-1】，而最后1个顶点的右邻居为【0，1，2，3】，为了方便后面的卷积过程，我们构建了一个矩阵adj=(M, adj_num=4)来记录每个顶点的左邻居顶点的编号，但是在代码中可以发现该adj并未被使用，可能在测试过程中不需要使用adj，在训练中可能需要。

​        在得到adj矩阵和每个多边形顶点的最终特征表示init_input后，snake模型首先使用1维循环卷积更新1次顶点特征，因为卷积作用在首尾相连的环形图上，所以特征表示对应的顶点总个数始终是不变的；然后使用7层不同空洞大小的残差循环卷积来得到7种包含不同尺度融合信息的顶点特征表示，7层对应的空洞大小为[1, 1, 1, 2, 2, 4, 4]，可以看出前面几层提取近距离的特征，后面几层保留了近距离特征的同时提取出了长距离信息；再然后将前面8种特征进行维度拼接，并连接了8种特征经过融合层合并后的融合特征的最大值，得到最终更新的顶点特征（这里面可以发现很多模型都将多种特征的最大值拼接起来，因为特征每个维度上的最大值可能来自于不同的顶点，所以这样可能能够进一步捕获不同顶点之间的关系）。最后我们使用连续的3层1维卷积，依次将顶点的特征维度从 融合维度  转化到 256，64，2 ，最后的2代表预测的坐标偏移值。

​        将多边形顶点坐标的原本值加上偏移值，得到更新后的多边形顶点坐标。显而易见，这里得到的多边形对应的顶点个数应该是采样后的顶点个数M。但是注意我们这里实际返回的值对于每个多边形来说是4个顶点，可以这样理解，4菱形顶点进行了1次变形，得到了变形后的4顶点，采样只是为了让形状变化的程度相对缓和，不会大幅度跑偏。

```python
    # Snake(state_dim=128, feature_dim=64+2, conv_type='dgrid')
    # 这里应该主要是完成论文中所指的轮廓初始化工作，因为这里的轮廓使用的就是多边形表示的，所以这里也可以理解为是多边形顶点坐标相关的初始化工作
    # i_it_poly c_it_poly【中心点个数、采样个数、2】 torch.Size([240, 160, 2])
    def init_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros([0, 4, 2]).to(i_it_poly) 

        h, w = cnn_feature.size(2), cnn_feature.size(3) # 304 608
        init_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w) # torch.Size([240, 64, 160])
        center = (torch.min(i_it_poly, dim=1)[0] + torch.max(i_it_poly, dim=1)[0]) * 0.5  # torch.Size([240, 2])
        ct_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, center[:, None], ind, h, w) # torch.Size([240, 64, 1])
        init_feature = torch.cat([init_feature, ct_feature.expand_as(init_feature)], dim=1) # torch.Size([240, 64, 160]) + torch.Size([240, 64, 160])
        init_feature = self.fuse(init_feature) # torch.Size([240, 128, 160]) ======> torch.Size([240, 64, 160])
        # torch.Size([240, 66, 160]) 在此处的特征中直接加入了多边形顶点的坐标
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1) #torch.Size([240, 64, 160]) + torch.Size([240, 2, 160,])
        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, init_input.size(2), init_input.device) # torch.Size([160, 4])
        i_poly = i_it_poly + snake(init_input, adj).permute(0, 2, 1)  # torch.Size([240, 160, 2]) + torch.Size([240, 160, 2]) ===>torch.Size([240, 160, 2])
        i_poly = i_poly[:, ::snake_config.init_poly_num//4] # 160 // 4 == 40 torch.Size([240, 4, 2])

        return i_poly
```

```python
# 这里的snake结构复杂
# 第一个部分是DilatedCircConv-ReLu-BN的BasicBlock结构，主要利用空洞的图卷积获取特征，
#     特征维度从feature_dim到state_dim
# 第二个部分是7层的不同空洞大小的连续BasicBlock结构，特征维度不变，均是state_dim
#     后面将利用其进行残差连接f
# 第三个是融合不同感受野特征的1维卷积融合层
# 第四个是模型预测层，主要包含三个连续的1维卷积层 （疑问：这里为什么不需要BN结构了？）
class Snake(nn.Module):
    def __init__(self, state_dim, feature_dim, conv_type='dgrid'):
        super(Snake, self).__init__()

        self.head = BasicBlock(feature_dim, state_dim, conv_type)

        self.res_layer_num = 7
        dilation = [1, 1, 1, 2, 2, 4, 4]
        for i in range(self.res_layer_num):
            conv = BasicBlock(state_dim, state_dim, conv_type, n_adj=4, dilation=dilation[i])
            self.__setattr__('res'+str(i), conv)

        fusion_state_dim = 256
        self.fusion = nn.Conv1d(state_dim * (self.res_layer_num + 1), fusion_state_dim, 1)
        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 1)
        )

    # snake算法的执行过程
    # （1）首先利用头部图卷积更新初始的输入特征x1, 将x1存放进states数组；
    # （2）然后堆叠7层感受野分别为[1, 1, 1, 2, 2, 4, 4]的残差空洞图卷积层，
    #      即每个残差层都是一个空洞图卷积，
    #  且把当前空洞图卷积层的输出和上一层的输出连接起来作为下一层的输入；
    #     这样每个残差层就能得到一种特征表示xi(i=2,3,4,5，6，7，8)，同样存放进states数组；
    # （3）将states数组中的特征按照维度方向进行拼接，
    #       然后利用1维卷积融合模块得到更新后的特征x (256d)
    #  (4) 这里的求最大值是求每个像素点的各个维度上特征的最大值吗？
    #      还是说求在每个维度层面上像素点的最大值
    def forward(self, x, adj):
        print('the shape of x is:', x.shape) # torch.Size([240, 66, 160])
        states = []

        x = self.head(x, adj) # torch.Size([240, 128, 160])
        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__('res'+str(i))(x, adj) + x
            states.append(x)

        state = torch.cat(states, dim=1) # torch.Size([240, 1024, 160]) = torch.Size([240, 128 * 8, 160])
        # torch.max返回的值包含values和indices，对应下标分别为0和1
        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0] # torch.Size([240, 256, 1])
        global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2)) # torch.Size([240, 256, 160])
        state = torch.cat([global_state, state], dim=1) # torch.Size([240, 1280, 160])
        x = self.prediction(state) # torch.Size([240, 2, 160])

        return x
```

【3】：在上一步得到初始的多边形四点轮廓后，我们**进一步对多边形轮廓进行变形，也即对4点坐标再次进行修正或者说演化**。主要包括

**多边形轮廓演化的准备工作和具体的演化过程**。

​          轮廓演化的准备工作包括两点，首先**对上一步变形得到的4点坐标进行越界修正**，也就是说限定横纵坐标的最小值和最大值，如果超出或者低于要用相应的极值进行替换，这里面传入的h和w对应于图片提取的特征图cnn_feature的宽和高，这也说明之前获取的bbox对应的图片大小与cnn_feature是一致的；其次是**从修正的4点坐标中获取8边形并对8边形进行采样**，修正的4点坐标是从矩形检测框bbox的规则四中点菱形变形而来，其仍然具备菱形的大体形态但是可能是不太规则的，其对应的外接矩阵可能不是原来的bbox框了，我们可以通过构造其对应的新的外接矩阵，从而方便8边形对应的12个顶点坐标的定位，采样方式仍然是依据边长来加权采样，采样的个数在代码中设定为snake_config.poly_num。

​        在得到了8边形采样后的多边形顶点坐标(M,  snake_config.poly_num, 2) 后，**我们进行具体的演化过程**。首先我们同样使用双线性采样从图片的提取特征图cnn_feature中获取多边形顶点的特征表示，并连接上snake_config.ro=4倍的多边形顶点的归一化坐标，得到初步的多边形顶点特征init_input （这里的归一化坐标应该是比较小的，乘以snake_config.ro倍应该能够放大顶点坐标在顶点特征表示中的影响吧）；然后同样构造(snake_config.poly_num, 4)的邻接矩阵adj，结合adj和init_input作为Snake模型的输入我们能够得到坐标偏移值

(M,  snake_config.poly_num, 2)；最后将未归一化的顶点坐标 乘以 snake_config.ro倍 并加上偏移值 得到 演化后的多边形顶点坐标值

(M,  snake_config.poly_num, 2)。

```python
 evolve = self.prepare_testing_evolve(output, cnn_feature.size(2), cnn_feature.size(3))
  py = self.evolve_poly(self.evolve_gcn, cnn_feature, evolve['i_it_py'], evolve['c_it_py'], init['ind']) # torch.Size([243, 120, 2])
```

~~~python
 # 首先对使用1层snake演变后的多边形顶点进行越界修正
    def prepare_testing_evolve(self, output, h, w):
        ex = output['ex'] # torch.Size([240, 4, 2])
        ex[..., 0] = torch.clamp(ex[..., 0], min=0, max=w-1)
        ex[..., 1] = torch.clamp(ex[..., 1], min=0, max=h-1)
        evolve = snake_gcn_utils.prepare_testing_evolve(ex)
        output.update({'it_py': evolve['i_it_py']}) # torch.Size([1, 240, 120, 2])
        return evolve
~~~

~~~python
# ex  torch.Size([240, 4, 2])  获取八边形后并进行了一次上采样
def prepare_testing_evolve(ex):
    if len(ex) == 0:
        i_it_pys = torch.zeros([0, snake_config.poly_num, 2]).to(ex)
        c_it_pys = torch.zeros_like(i_it_pys)
    else:
        i_it_pys = snake_decode.get_octagon(ex[None]) # ex:torch.Size([240, 4, 2])  ===> torch.Size([1, 240, 12, 2])
        # i_it_pys = uniform_upsample(i_it_pys, snake_config.poly_num)[0]
        i_it_pys = i_it_pys.repeat(1,1,snake_config.poly_num//12,1)[0] # torch.Size([1, 240, 120, 2])[0]
        c_it_pys = img_poly_to_can_poly(i_it_pys) # torch.Size([240, 120, 2])
    evolve = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys}
    return evolve
~~~

~~~python
 # i_it_poly torch.Size([240, 120, 2]) ind torch.Size([240]) cnn_feature torch.Size([3, 64, 304, 608])
    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3) # 304，608
        init_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w) # torch.Size([240, 64, 120])
        c_it_poly = c_it_poly * snake_config.ro # torch.Size([240, 120, 2])
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1) # torch.Size([240, 66, 120])
        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, init_input.size(2), init_input.device) # torch.Size([120, 4])
        i_poly = i_it_poly * snake_config.ro + snake(init_input, adj).permute(0, 2, 1) # torch.Size([240, 120, 2])
        return i_poly
~~~

【4】：不断通过Snake来演变顶点坐标。pys数组用于存储所有演变过程中的多边形顶点坐标，pys[0]存储的是上一步得到的顶点坐标且进行了snake_config.ro倍的缩小。

~~~python
pys = [py / snake_config.ro]
# 应该是通过多次迭代来演化多边形顶点坐标
for i in range(self.iter):
    py = py / snake_config.ro  # torch.Size([240, 120, 2])
    c_py = snake_gcn_utils.img_poly_to_can_poly(py) # torch.Size([240, 120, 2])
    evolve_gcn = self.__getattr__('evolve_gcn'+str(i))
    py = self.evolve_poly(evolve_gcn, cnn_feature, py, c_py, init['ind']) # torch.Size([240, 120, 2])
    pys.append(py / snake_config.ro)
    ret.update({'py': pys})
~~~





![image-20210602110613023](C:\Users\levin\AppData\Roaming\Typora\typora-user-images\image-20210602110613023.png)






