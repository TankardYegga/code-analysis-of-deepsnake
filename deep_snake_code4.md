# Deep Snake代码阅读记录4

看懂了最后1部分的实行轮廓演变的子网络结构，并对deep_snake代码种的相关技巧进行了总结。

（编号续上周报告）

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



## 3 代码技巧总结

### 3.1 使用cfg并结合ArgumentParser来表示模型及其训练测试过程中相关参数的设定

（1）CfgNode是**由FaceBook开源的YACS API**（Yet Another Configuration System，专门用于商业或者科研中的参数配置管理）中**提供的参数设定的模板类**，可以很方便地**使用cfg.X = Y的方式来给一系列的参数设定初始的默认值**，这时候的参数相当于是这个类的一些属性但是可以随时新增。

（2）其用法具有3点特殊性：1）**本身是可以递归使用的**，也就是说cfgNode对象的参数本身可以是另外一个CfgNode对象，比如代码中的cf.heads = CN(), cf.train = CN(), cf.test = CN()，分别为模型的预测头、训练参数、测试参数创建了一个子CfgNode对象。**好处是可以使得参数的层次结构比较清晰。 **2）**从yaml配置文件中直接导入外部配置选项**，直接调用CfgNode对象的merge_from_file（filename)方法就可以实现。3) **可以从一个数组列表中直接导入外部配置选项，**直接调用CfgNode对象的merge_from_list（var name)即可以实现。

​      通常来说，cfgNode是作为一个全局的变量来使用的，其中所有模型都涉及到的参数设定写在全局的code当中，而模型间有区别的参数则放置在ArgumentParser的参数选项中，后来再从其中导入。

```python
from .yacs import CfgNode as CN
import argparse
import os

cfg = CN()

# model
cfg.model = 'hello'
cfg.model_dir = '/media/disk/exp/snake-master/snake-master/data/model'

# network
cfg.network = 'dla_34'

# network heads
cfg.heads = CN()

# task
cfg.task = ''

# gpus
cfg.gpus = [3]

# if load the pretrained network
cfg.resume = True


# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = 'CocoTrain'
cfg.train.epoch = 140
cfg.train.num_workers = 8

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 5e-4

cfg.train.warmup = False
cfg.train.scheduler = ''
cfg.train.milestones = [80, 120, 200, 240]
cfg.train.gamma = 0.5

cfg.train.batch_size = 4

# test
cfg.test = CN()
cfg.test.dataset = 'CocoVal'
cfg.test.batch_size = 1
cfg.test.epoch = -1

# recorder
cfg.record_dir = '/media/disk/exp/snake-master/snake-master/data/record'

# result
cfg.result_dir = '/media/disk/exp/snake-master/snake-master/data/result'

# evaluation
cfg.skip_eval = False

cfg.save_ep = 5
cfg.eval_ep = 5

cfg.use_gt_det = False

# -----------------------------------------------------------------------------
# snake
# -----------------------------------------------------------------------------
cfg.ct_score = 0.05
cfg.demo_path = ''

# 利用args来复制增添cfg的相关属性
def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # assign the gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])

    cfg.det_dir = os.path.join(cfg.model_dir, cfg.task, args.det)

    # assign the network head conv
    # 第一个卷积层的维度
    cfg.head_conv = 64 if 'res' in cfg.network else 256

    # 模型文件目录设置 = 模型根目录 + 任务名 + 任务中使用到的模型（在对应的任务.yaml配置文件中使用到）
    cfg.model_dir = os.path.join(cfg.model_dir, cfg.task, cfg.model)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.model)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.model)

def make_cfg(args):
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    parse_cfg(cfg, args)
    return cfg

# python run.py --type  --cfg_file configs/default.yaml
parser = argparse.ArgumentParser()
# 配置文件保存的地方
parser.add_argument("--cfg_file", default="/media/disk/exp/snake-master/snake-master/configs/city_rcnn_snake.yaml", type=str)
# 这里action的含义是说如果参数中存在--test选项(不需要再额外赋值了）那么对应的值就为True
# 如果参数中不存在--test选项的话那么默认为False
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="evaluate")
parser.add_argument('--det', type=str, default='')
parser.add_argument('-f', type=str, default='')
# 将所有剩余的参数组成一个列表赋成此项
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
# 只要指明了类型，那么就一定有task为run
if len(args.type) > 0:
    cfg.task = "run"
cfg = make_cfg(args)
```

### 3.2  使用imp包来方便地调用不同层级目录的功能模块

 deep snake的目录结构非常复杂，主要是涉及到的实验内容和模型结构很多。

 **imp包的load_source(module_name, module_file_path)方法**可以从module_file_path所代表的文件路径中加载module_name所对应的module，module_name注意也是包含了模块的目录层次关系的，比如lib.networks.rcnn_snake。

好处是**提供了一个统一的接口**，能够调用具有相似功能、且目录结构上相对接近的模块。

~~~python
# 从cfg的任务中选择任务对应的相应模型
def make_network(cfg):
    print('cfg is',cfg)
    print('cfg task is:', cfg.task)
    module = '.'.join(['lib.networks', cfg.task])
    path = os.path.join('/media/disk/exp/snake-master/snake-master/lib/networks', cfg.task, '__init__.py')
    print('module is',module)
    print('path is',path)
    return imp.load_source(module, path).get_network(cfg)
~~~

### 3.3  对网络的训练和测试使用高级别的封装，使得调用代码十分简洁

（1）**高级别的调用过程：**

~~~python
def train(cfg, network):
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)

    begin_epoch = load_model(network, optimizer, scheduler, recorder, cfg.model_dir, resume=cfg.resume)
    set_lr_scheduler(cfg, scheduler)

    train_loader,train_dataset = make_data_loader(cfg, is_train=True)
    val_loaderm,test_dataset = make_data_loader(cfg, is_train=False)

    # 从预训练模型迭代次数 + 1 次迭代开始 一直到 总迭代次数
    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0:
            save_model(network, optimizer, scheduler, recorder, epoch, cfg.model_dir)

        if (epoch + 1) % cfg.eval_ep == 0:
            trainer.val(epoch, val_loader, evaluator, recorder)

    return network
~~~

（2）高层次封装可以分3个方向：

【1】：关于网络训练和测试的封装：make_trainer

​          下面的3段代码展示了对最基础的Network进行封装的过程。首先我们利用NetWrapper类将基本的Network架构和对应的损失函数封装起来（注意其也是nn.Module的子类）得到1个NetWorkWrapper对象，其前向传播函数除了计算网络的输出外还计算了不同的损失，这里能计算损失的前提是输入的batch数据中同时也包含了数据的label；然后我们使用Trainer类来对NetWorkWrapper对象进一步封装，主要实现了其train和val/test方法，这两个方法的输入除了数据还需要其他模块（对于train来说需要训练优化器trainer和相关结果保存器recorder，对于val/test来说需要训练评估器evaluator以及保存器recorder），注意两个方法都是只训练和测试1个epoch。

​        这样我们就得到lTrainer对象。

```python
# 对网络再进行深层次封装
# 包括模型的部署、训练和验证过程
def make_trainer(cfg, network):
    # 根据cfg来确定对应的网络封装类这一模块的位置
    network = _wrapper_factory(cfg, network)
    return Trainer(network)
```

 ~~~python
 class NetworkWrapper(nn.Module):
     def __init__(self, net):
         super(NetworkWrapper, self).__init__()
 
         self.net = net
 
         self.act_crit = net_utils.FocalLoss()
         self.awh_crit = net_utils.IndL1Loss1d('smooth_l1')
         self.cp_crit = net_utils.FocalLoss()
         self.cp_wh_crit = net_utils.IndL1Loss1d('smooth_l1')
 
     def forward(self, batch):
         output = self.net(batch['inp'], batch)
 
         scalar_stats = {}
         loss = 0
 
         act_loss = self.act_crit(net_utils.sigmoid(output['act_hm']), batch['act_hm'])
         scalar_stats.update({'act_loss': act_loss})
         loss += act_loss
 
         awh_loss = self.awh_crit(output['awh'], batch['awh'], batch['act_ind'], batch['act_01'])
         scalar_stats.update({'awh_loss': awh_loss})
         loss += 0.1 * awh_loss
 
         act_01 = batch['act_01'].byte()
 
         cp_loss = self.cp_crit(net_utils.sigmoid(output['cp_hm']), batch['cp_hm'][act_01])
         scalar_stats.update({'cp_loss': cp_loss})
         loss += cp_loss
 
         cp_wh, cp_ind, cp_01 = [batch[k][act_01] for k in ['cp_wh', 'cp_ind', 'cp_01']]
         cp_wh_loss = self.cp_wh_crit(output['cp_wh'], cp_wh, cp_ind, cp_01)
         scalar_stats.update({'cp_wh_loss': cp_wh_loss})
         loss += 0.1 * cp_wh_loss
 
         scalar_stats.update({'loss': loss})
         image_stats = {}
 
         return output, loss, scalar_stats, image_stats
 ~~~

```python
# 对训练网络的进一步封装
# 包括将网络放置在gpu显卡上并行处理
# 并定义了其上的训练、测试函数
class Trainer(object):
    def __init__(self, network):
        network = network.cuda()
        network = DataParallel(network)
        self.network = network

    # 对每个键值都进行了平均，实现了降维
    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    # 对batch的元素经过分析后放置在gpu显卡上
    # 比如meta就不放、元组取出来每个都放、其余的直接放
    def to_cuda(self, batch):
        for k in batch:
            if k == 'meta':
                continue
            if isinstance(batch[k], tuple):
                batch[k] = [b.cuda() for b in batch[k]]
            else:
                batch[k] = batch[k].cuda()
        return batch

    # 训练网络
    def train(self, epoch, data_loader, optimizer, recorder):
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()
        # 这里知道数据中batch是什么很重要
        # 这里的recorder是什么数据类型
        # 感觉像是一个类？因为每次训练过程中都对其step属性进行计数操作
        # 还调用了update_loss_stats等方法
        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1
            recorder.step += 1

            # batch = self.to_cuda(batch)
            # 网络有四类输出，分别正常输出、损失、损失状态、图像状态
            output, loss, loss_stats, image_stats = self.network(batch)

            # training stage: loss; optimizer; scheduler
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
            optimizer.step()

            # data recording stage: loss_stats, time, image_stats
            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            # batch_time记录的是本轮训练结束距离上轮训练结束的时间
            # data_time记录的是本轮训练开始距离上轮训练结束的时间
            # 如果data_time为0，即上一轮训练结束后没有输出训练的状态
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)
           
            # 每20次或者最后一次迭代打印一下训练状态
            if iteration % 20 == 0 or iteration == (max_iter - 1):
                # print training state
                # 计算剩余的大概时间？用每批次的平均训练时间 乘上 剩余的迭代次数
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                lr = optimizer.param_groups[0]['lr']
                # 将返回此程序开始以来的Tensor的峰值分配内存
                # 原始单位是字节
                # 这里应该是MB
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                # 这里recorder的字符串表示是什么
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

                # record loss_stats and image_dict
                # 调用了recorder的两个方法：更新图像状态和记录
                # 图像状态是网络的输出结果之一
                recorder.update_image_stats(image_stats)
                recorder.record('train')

    # 验证或者说是测试网络
    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        self.network.eval()
        # 执行完下面这条命令，在nivid-smi中才会同步减去被释放的内存
        # 如果直接使用cpu()释放，在nivida-smi中不会减去
        torch.cuda.empty_cache()

        # 记录测试过程的相关状态
        val_loss_stats = {}
        data_size = len(data_loader)
        for batch in tqdm.tqdm(data_loader):
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].cuda()

            with torch.no_grad():
                output, loss, loss_stats, image_stats = self.network(batch)
                # 通过模型的输入和输出来计算评价指标
                if evaluator is not None:
                    evaluator.evaluate(output, batch)
            
            # 损失状态进行降维
            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                # 将val_loss_stats同样键的值设定为0，然后重新赋值
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        # 保存平均后的损失结果
        print(loss_state)

        if evaluator is not None:
            # summarize和evaluate函数的区别
            result = evaluator.summarize()
            val_loss_stats.update(result)

        # 只需记录迭代次数、验证损失情况、最后一次验证的图像状态结果
        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)
```

【2】：**构建其他相关的功能构建**

包括make_data_loader(cfg, is_train=True) 分别用于训练和测试的数据加载器，以及

 optimizer = make_optimizer(cfg, network)

  scheduler = make_lr_scheduler(cfg, optimizer)

  recorder = make_recorder(cfg)

  evaluator = make_evaluator(cfg)。

【3】：**模型保存、加载参数的utils**

~~~python
# 从相关的文件中加载模型，包括网络net还有optim、scheduler、recorder、
def load_model(net, optim, scheduler, recorder, model_dir, resume=True, epoch=-1):
    # 如果没有恢复或者说不打算使用预训练的模型参数
    # 那么就直接删除模型文件夹
    if not resume:
        os.system('rm -rf {}'.format(model_dir))
        return 0

    if not os.path.exists(model_dir):
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0

    # 查找目录下的检查点文件
    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
    if len(pths) == 0:
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0
    # 如果没有指定迭代次数的话，那么就以检查点文件群中的最大迭代次数作为选择
    # 否则的话就直接使用函数参数中指定的迭代次数
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    print('load model: {}'.format(os.path.join(model_dir, '{}.pth'.format(pth))))
    
    # 根据选定的迭代次数来加载相应的模型检查点文件
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    # 可以看出预训练模型是分为网络、优化器、学习率调控器、记录器这4个部分来分别加载的
    # 这种加载方式与amp的模型保存方式有关
    net.load_state_dict(pretrained_model['net'])
    optim.load_state_dict(pretrained_model['optim'])
    scheduler.load_state_dict(pretrained_model['scheduler'])
    recorder.load_state_dict(pretrained_model['recorder'])
    # 返回预训练模型的迭代次数 + 1
    return pretrained_model['epoch'] + 1

# 创建相应的模型检查点保存目录，并将模型检查点文件保存在内
def save_model(net, optim, scheduler, recorder, epoch, model_dir):
    os.system('mkdir -p {}'.format(model_dir))
    torch.save({
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, '{}.pth'.format(epoch)))

    # remove previous pretrained model if the number of models is too big
    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
    if len(pths) <= 200:
        return
    # 如果预训练模型的检查点文件超过200个，则删除其中迭代次数最少的
    os.system('rm {}'.format(os.path.join(model_dir, '{}.pth'.format(min(pths)))))


# 加载网络net，其他load_model函数中加载的这里不需要加载 
# model_dir '/media/disk/exp/snake-master/snake-master/data/model/rcnn_snake/long_rcnn'
def load_network(net, model_dir, resume=True, epoch=-1, strict=True):
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0

    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir) if 'pth' in pth]
    if len(pths) == 0:
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    print('load model: {}'.format(os.path.join(model_dir, '{}.pth'.format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    net.load_state_dict(pretrained_model['net'], strict=strict)
    return pretrained_model['epoch'] + 1
~~~



