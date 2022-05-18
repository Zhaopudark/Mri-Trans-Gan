"""
helper build conv layers
众所周知 一个卷积过程是
由一个输入(input) 经过一个卷积核(filter) 以步长(stride)-膨胀率(dilation_rate)-补齐方式(padding)-等基本参数 的作用 获得输出(output)的过程
我们关注卷积过程前后的size的改变情况 改变的过程 因此将卷积过程中的size变化记为: 
input_length-->conv(filter_size,stride,dilation_rate,padding,..)-->output_length 是一个多对一的映射
而卷积的反向过程的size变化记为
output_length-->deconv(filter_size,stride,dilation_rate,padding,..)-->input_length 是一个一对多的映射
SECTION 以下过程将探讨不同padding方式下output_length input_length的计算过程 
    ANCHOR 约定如下
        定义 padding 表示卷积操作内对输入的自动补齐方式名字 有 valid same full causal等值
            直接记为 valid same full causal 
        定义 output_padding 表示反卷积的输出补齐方式 一般是人为给定的
        定义 padding_behavior 表示padding操作 是封闭在conv内部的操作
            有  valid_behavior 
                same_behavior 
                full_behavior 
                causal_behavior
        定义 output_padding_behavior 表示output_padding操作 是封闭在deconv内部的操作
        定义 artificial_padding 表示卷积操作外对输入的手动补齐方式名字 有 valid same full causal等值
            记为 artificial_valid artificial_same artificial_full artificial_causal
        定义 artificial_padding_behavior 表示artificial_padding操作 是在conv外部的操作 统称为 外部padding
            有  artificial_valid_behavior 
                artificial_same_behavior 
                artificial_full_behavior 
                artificial_causal_behavior
        定义 artificial_output_padding_behavior 表示artificial_output_padding操作 是在deconv外部的操作
        定义 padding_conv 表示基于某个补齐方式的卷积操作 是封闭的内部过程
            有 valid_conv same_conv full_conv causal_conv
        定义 padding_deconv 表示基于某个补齐方式的反卷积操作 是封闭的内部过程
            有 valid_deconv same_deconv full_deconv causal_deconv
        定义 artificial_padding_conv 表示先在卷积操作外进行人为的补齐操作artificial_padding_behavior 再进行valid_conv的过程
            有 artificial_valid_conv artificial_same_conv artificial_full_conv artificial_causal_conv
        ANCHOR 设"+"表示若干行为的组合 则
        定义 函数 output_length(·)表示由一个确定的输入length计算输出length的过程 返回输出length即output_length
        定义 函数 input_length(·,p) 表示由一个确定的输出length计算一个满足要求p的输入length的过程 返回输入length即input_length
        定义 函数 input_length_range(·) 表示由一个确定的输出length计算输入length可行域的过程 返回输入length可行域input_length_range 是output_length()的反函数
            NOTE 不存在 padding+valid_padding_deconv 因为反卷积是不存在输入padding的
                 不存在 output_length(*deconv*) input_length(*deconv*)  因为反卷积就是卷积的矩阵形式的卷积核转置运算 视为同一个过程的正序逆序 避免混淆
        定义 === 表示恒等 用于结论或者待证明的结论中 用以凸显
        定义 == 表示相等 用于证明过程中间 用于强调必要前提
        定义 = 表示相等 也表示赋值 用于过程中间 不需要强调凸显的地方

    ANCHOR 为了分析问题,对某个具体的框架(tf,pytorch,paddle等)做如下假设
        假设1. 封闭的所有conv是基于同一个后端的(一般的开发模式皆如此,可以提高代码复用率)
        假设2. 封闭的valid_conv是完全和定义等效的(操作结果是用户完全知晓且确定的,'等效'表示运行内部或者底层的优化但必须维持操作结果的等效)
        假设3. 内部的valid_behavior full_behavior causal_behavior是完全依据定义执行的
        假设4. 内部的same_behavior也是完全依据定义执行的一个确定的无歧义行为
                由于same_behavior是output_length==ceil(input_length/stride)的padding行为
                本身是存在不确定性的 即 假设按照要求需要padding总数为3 那么左右的padding可以分别是[0,3] [1,2] [2,1] [3,0] 而皆满足定义要求
                但同时 same_behavior 只采用一套行为准则 即对于同一批输入参数 same_behavior只给出唯一确定的padding结果
                不过由于该行为封装在内部 用户一般是无法知晓的 只能揣测
    ANCHOR 假设不成立的若干指南
        若 假设1 不成立 (翻看源码可知)
            需要考察不同后端在操作结果上的区别 用一套规则引导这个差异 相应地修改部分结论
        若 假设2 不成立 (实验与手动计算比较可知)
            则框架本身就缺乏数学计算结果上的正确性,弃之
        若 假设3 不成立 (用artificial_padding_behavior+valid_conv的结果与手动计算比较可知)
            则框架的可拓展性太差(只能用其提供的实例而无法做太多细节修改),建议弃之
        若 假设4 不成立 (用artificial_same_behavior+valid_conv尝试模拟same_conv可知)
            则框架的可拓展性太差(只能用其提供的实例而无法做太多细节修改),建议弃之
    ANCHOR 所以有前提性的结论如下:
        NOTE 一般的 框架皆满足如上4点假设 
        因此 valid full causal是按照其原始定义的padding行为 虽然封装在conv内部 但已知且确定的
            same也是按照其原始定义的padding行为 是封装在内部的确定但位置的行为
        虽然 valid_conv same_conv full_conv causal_conv的具体后端是未知的
            但是都可以用内部操作和valid_conv模拟 即 padding_conv = padding_behavior+valid_conv
        
        有 valid_conv = valid_behavior + valid_conv 
        有 same_conv = same_behavior + valid_conv 
        有 full_conv = full_behavior + valid_conv 
        有 causal_conv = causal_behavior + valid_conv 

        依据定义 必定有 artificial_valid_conv = artificial_valid_behavior + valid_conv
        依据定义 必定有 artificial_same_conv = artificial_same_behavior + valid_conv
        依据定义 必定有 artificial_full_conv = artificial_full_behavior + valid_conv 
        依据定义 必定有  artificial_causal_conv = artificial_causal_behavior + valid_conv 

        显然有 artificial_valid_conv = valid_conv
        至于其余 artificial_padding_conv 与 padding_conv的关系则需要视内 padding_behavior和artificial_padding_behavior是否一致而决定

考察六个问题
ANCHOR 问题1 artificial_padding_conv===padding_conv ?
    在满足4点假设的情况下
    由于artificial_padding_conv和padding_conv最终的后端都是valid_conv 
    因此他们有一样的对待数据的模式 一样的卷积核移动方向 只需要考虑padding的问题
    不妨设输入Tensor的每个维度都是由左至右的顺序模式 如果有时间顺序 则是由左至右的时间先后顺序(casual面向时间序列https://arxiv.org/abs/1609.03499)
    不妨设卷积核是从左至右的顺序移动计算 
    假设一个基本的1D-conv过程 假设 filter_size 已经经过了dilation 
        那么有 output_length = (input_length+left_pad+right_pad-filter_size+1+stride-1) // stride 
        解释如下
            输入经过左右padding后
            卷积核从初始位置先计算一次 往后每次按stride步长向右移动 每移动一次进行一次计算
            观察卷积核最右侧一列,记为L  并记L扫过的总列数(包括每步之间跳过的列数)记为ALL_L
            output_length == 1+(ALL_L-1)//stride  即初始位置+后续可行位置能被stride划分的区间数
        ALL_L == all_l_func(input_length,left_pad,right_pad,filter_size) 
              == all_l_func2(padded_input_length,filter_size) 
              == padded_input_length-filter_size+1 即摆放第一个卷积核后剩余的位置+初始位置
        padded_input_length == input_length+left_pad+right_pad
    当 padding==valid 时
        只要满足 外部padding按照valid的定义进行 则就与内部的padding行为一致
        即left_pad=left_pad=0
        就有 artificial_valid_behavior==valid_behavior
    当 padding==full 时
        只要满足 外部padding按照full的定义进行 则就与内部的padding行为一致
        即 left_pad=left_pad=filter_size-1
        就有 artificial_full_behavior==full_behavior 
    当 padding==causal 时
        只要满足 外部padding按照causal的定义进行 则就与内部的padding行为一致
        即 left_pad=filter_size-1 right_pad=0
        就有 artificial_causal_behavior==causal_behavior 
    当 padding==same 时 
        由于same不是直接定义了left_pad和right_pad
        而是以 output_length = ceil(input_length/stride)作为唯一的定义依据 NOTE 前三个padding是padding驱动 same padding是output_length驱动
        假设以 left_pad和right_pad进行外部padding 并记 pad_all = left_pad+right_pad  满足
        output_length == 1+(ALL_L-1)//stride == ceil(input_length/stride)
        那么依据
            ALL_L == all_l_func(input_length,left_pad,right_pad,filter_size)
                == all_l_func2(padded_input_length,filter_size)
            padded_input_length == input_length+left_pad+right_pad
        只要 pad_all = left_pad+right_pad 总和不变 所有非负整数的 left_pad right_pad 组合皆可行
        而这其中 有且只有一个组合使得 artificial_same_behavior==same_behavior
        所以仅仅确保外部padding按照same的定义进行 是无法保证与内部的same padding 行为一致的

        当满足 外部padding按照same的定义进行 并且与内部的padding行为一致时 
        有 artificial_same_behavior==same_behavior 
        当满足 外部padding按照same的定义进行 并且与内部的padding行为不一致时 
        有 artificial_same_behavior!=same_behavior 

    所以
    在4点基本假设的前提下
        artificial_valid_conv === valid_conv 在外部padding按照valid的定义进行时成立
        artificial_full_conv === full_conv 在外部padding按照full的定义进行时成立
        artificial_causal_conv === causal_conv 在外部padding按照causal的定义进行时成立
        artificial_same_conv == same_conv 在外部padding按照same的定义进行 并且与内部的padding行为一致时成立(artificial_same_behavior==same_behavior)
    
    所以
    在4点基本假设的前提下
    存在一个artificial_padding_conv 对应着的 artificial_padding_behavior
        当 padding == valid或者full或者causal时 
            artificial_padding_behavior 与定义完全一致
        当 padding == same时 
            artificial_padding_behavior==same_behavior 与封闭在conv内部的same padding 行为一致
        使得 artificial_padding_conv=== artificial_padding_behavior + valid_conv === padding_conv
    以上是用artificial_padding_conv模拟padding_conv的理论基础
    可以用于拓展既有的封闭起来的conv操作的padding方式

ANCHOR 问题2 output_length(artificial_padding_conv)===output_length(padding_conv) ?
    由问题1的结论
        满足artificial_padding_conv===padding_conv要求的 artificial_padding_behavior记为artificial_padding_behavior_1
    定义 artificial_padding_behavior_2 是artificial_padding_behavior_1的变体
        artificial_padding_behavior_2 在padding==same时 会给出与artificial_padding_behavior_1的pad_left pad_right相反的两个pad结果
        NOTE artificial_padding_behavior_2的pad_left pad_right依旧可以满足output_length = ceil(input_length/stride)要求
    那么尽管artificial_padding_conv===padding_conv在artificial_padding_behavior_2时不再恒成立
        但此时output_length(artificial_padding_conv)===output_length(padding_conv)恒成立
    
    更普遍的
    如果定义artificial_padding_behavior_2  
        在 padding == valid或者full或者causal时 与artificial_padding_behavior_1行为一致
        在 padding == same时满足 output_length = ceil(input_length/stride)定义的行为 与artificial_padding_behavior_1行为不一定一致
            设此时满足output_length = ceil(input_length/stride)定义的行为有M种(包括与artificial_padding_behavior_1一致的在内)
    
    那么
    在4点基本假设的前提下
    存在至少M种artificial_padding_behavior 
    使得output_length(artificial_padding_conv)===output_length(padding_conv)
        '至少M种':指的是在padding == valid或者full或者causal时 也许存在更多的
                满足output_length(artificial_padding_conv)==output_length(padding_conv)
                的artificial_padding_behavior 不一定要与artificial_padding_behavior_1完全一致 
                但是讨论这部分行为的意义不大,'至少M种'这一结论足以支持后续的函数定义
    
        '至少M种': 放宽了artificial_padding_behavior 在padding == same时的操作空间 有利于找到简介一致的output_length计算形式
        这是函数conv_output_length()的基础前提

ANCHOR 问题3 input_length_range(artificial_padding_conv)===input_length_range(padding_conv) ?
    由问题2可知 
    存在至少M种 artificial_padding_behavior 
    使得
        output_length(artificial_padding_conv)===output_length(padding_conv) 恒成立
    对于其中某一种确定的 artificial_padding_behavior_X
        input_length_range(·) 是 output_length()反函数
        始终有 output_length = (input_length+left_pad+right_pad-filter_size+1+stride-1)//stride #基于卷积定义的计算形式是必定存在的 artificial_padding_behavior改变的只是padding方式与行为 
        将该计算逆向 固定output_length 反求input_length 
        由于存在"//"等操作 反求的input_length是不唯一的 即存在多个input_length可以在正向计算中输出相同的output_length
        求出满足input_length->output_length映射的所有input_length值 即可获得input_length_range
    所以
    在4点基本假设的前提下
    存在至少M种 artificial_padding_behavior 使得 input_length_range(artificial_padding_conv)===input_length_range(padding_conv) 
        计算input_length_range 就是将artificial_padding_behavior下的 
        output_length = (input_length+left_pad+right_pad-filter_size+1+stride-1)//stride
        计算逆向进行
    这是deconv_output_length()函数的基础前提之一

ANCHOR 问题4 input_length(artificial_padding_conv,p)===input_length(padding_conv,p) ?
    由问题3可知
    在4点基本假设的前提下
    存在至少M种 artificial_padding_behavior 使得 input_length_range(artificial_padding_conv)===input_length_range(padding_conv) 
        计算input_length_range 就是将artificial_padding_behavior下的 
        output_length = (input_length+left_pad+right_pad-filter_size+1+stride-1)//stride
        计算逆向进行
    在某个 artificial_padding_behavior 下
        如果规定 p 为 "使返回的input_length为input_length_range的最小(最大)"
            因为 input_length_range(artificial_padding_conv)===input_length_range(padding_conv) 
            所以 input_length_range是确定的  其最值也是确定的
            所以必然有 input_length(artificial_padding_conv,p)===input_length(padding_conv,p) 成立 
        如果规定 p 为 "padding==valid或者same或者causal时返回最接近output_length*stride的input_length padding==full时返回最小input_length"
            因为 input_length_range(artificial_padding_conv)===input_length_range(padding_conv) 
            所以 input_length_range是确定的 其最值也是确定的 
            同时 input_length*stride也是确定的  
            所以必然有 input_length(artificial_padding_conv,p)===input_length(padding_conv,p) 成立 
        要计算input_length 即通过input_length(artificial_padding_conv,p)计算input_length
        就是将artificial_padding_behavior下的 
        output_length = (input_length+left_pad+right_pad-filter_size+1+stride-1)//stride
        计算逆向进行
        找到符合 'p' 要求的input_length
    这是函数conv_input_length()的基础前提
ANCHOR 问题5 deconv(·)过程如何确定 output_length
    deconv(·)在输入输出shape上 可以视为conv(·)的逆过程
    问题3 问题4可知
    在4点基本假设的前提下
        存在至少M种 artificial_padding_behavior 使得 input_length_range(artificial_padding_conv)===input_length_range(padding_conv)成立
        选定一个合适的 artificial_padding_behavior
        再规定一个条件 p 
        即可有 input_length(artificial_padding_conv,p)===input_length(padding_conv,p)
    input_length就是deconv(·)过程的output_length
    input_length会依据条件p而改变
    所以
    deconv(·)的output_length也不是唯一确定的
    要使得deconv(·)的output_length确定 没有歧义 
    可以同input_length(artificial_padding_conv,p)一样计算
        选定一个合适的 artificial_padding_behavior 
        再规定一个条件 p 
        以计算 input_length(artificial_padding_conv,p)
        即将 conv_output_length = (conv_input_length+left_pad+right_pad-filter_size+1+stride-1)//stride
        计算逆向进行
        找到符合 'p' 要求的 conv_input_length
        获得 output_length = conv_input_length

    事实上 对于 conv_output_length = (conv_input_length+left_pad+right_pad-filter_size+1+stride-1)//stride
        存在 stride个conv_input_length使得 (conv_input_length+left_pad+right_pad-filter_size+1+stride-1)%stride ∈[0,stride-1]
        条件 p就是规定了stride个conv_input_length中哪一个作为输出
    NOTE 不过一个已有的deconv(·)函数  artificial_padding_behavior和条件p是封装于内部的
    不同的条件p 会导致不同的 input_length结果
    要考虑input_length(artificial_padding_conv,p)结果是否满足用户的预期 
    因此将条件p用新的变量 output_padding代替
    一个合理的 deconv(·)计算output_length过程应当可以由用户进行控制 满足如下的逻辑
    设置一个新的变量 output_padding 接受用户控制
        方法1:
            当用户有output_padding需求时 
                先规定一个artificial_padding_behavior
                给出最小的output_length作为正向卷积的input_length
                    即 input_length(artificial_padding_conv,p)中 p条件设置为取最小值
                再依据output_padding 还原出用户希望的 output_length
                可以理解为在最小的conv_input_length基础上补充一个用户指定的值
            当用户没有output_padding需求时 
                先规定一个artificial_padding_behavior
                依据一般经验猜测一个合适的output_length
                    即 input_length(artificial_padding_conv,p)中 p条件设置为某个约束
        方法2:
            ...
        ...
        由于p条件是不唯一的 
        当用户没有output_padding需求时
            这些方法可以设置五花八门的p条件而致使output_length千奇百怪 尽管输出的output_length处于正确范围
        因此存在多种具体的deconv(·)计算output_length的方法
    函数deconv_output_length()就是以此为基础的
ANCHOR 问题6 反卷积过程中的padding是否可等价?如何进行等价
    一般的 将输入输出都扁平化为一个向量 则任意的卷积 都可以等效为一个矩阵乘法
    for example
    consider a 1-D conv with input X=[x1,x2,x3,x4,x5], kernel=[w1,w2,w3], stride = 1, padding='VALID',so 
        W = [[w1,0 ,0 ,],
             [w2,w1,0 ,],
             [w3,w2,w1,],
             [0 ,w3,w2,],
             [0 ,0 ,w3,]] #r5x3
        X = [[x1,x2,x3,x4,x5]] #r1x5 do not pad since  padding='VALID'
        Y = X@W = [[x1w1+x2w2+x3w3,x2w1+x3w2+x4w3,x3w1+x4w2+x5w3]] #r1x3
    consider a 1-D conv with input X=[x1,x2,x3], kernel=[w1,w2,w3], stride = 1, padding='same',so 
        W = [[w1,0 ,0 ,],
             [w2,w1,0 ,],
             [w3,w2,w1,],
             [0 ,w3,w2,],
             [0 ,0 ,w3,]] #r5x3
        X = [[0,x1,x2,x3,0]] #r1x5 pad since padding='same'
        Y = X@W = [[x1w2+x2w3,x1w1+x2w2+x3w3,x2w1+x3w2]] #r1x3
    而一个反卷积 就可以视为 反转卷积的输入输出尺度 并将卷积的卷积核稀疏矩阵转置后的矩阵乘法
    由问题5可知 
    deconv(·)必须确定output_length 而确定output_length的方法与结果是不唯一的
    问题5 给出了一个通过设置 output_padding变量计算output_length的方法 而更普遍的 直接规定output_length也是可行的
    以下将基于output_padding 进行计算
    for example if output_padding=0
    consider a 1-D deconv with input X=[x1,x2,x3], kernel=[w1,w2,w3], stride = 1, padding='VALID',so 
        consider 
        NMW.T = [[w1,w2,w3, 0, 0,...],
                 [0 ,w1,w2,w3, 0,...],
                 [0 ,0 ,w1,w2,w3,...],
                 ...,
                 ...,] #rMxN
        since input X is 3-D tensor
        from output_padding=0
            get output_length = (3 - 1) * 1 + 3 = 5
            get temp_output_length = output_length+pad_left+pad_right = 5 #get the padded length
            pad_left = 0
            pad_right = 0
        grab r3x5 matrix from NMW.T
        W.T = [[w1,w2,w3, 0, 0,],
               [0 ,w1,w2,w3, 0,],
               [0 , 0,w1,w2,w3,]] #r3x5
        X = [[x1,x2,x3]] #r1x3 deconv never pad inputs 
        temp_output = Y_ = X@W.T = [[x1w1,x1w2+x2w1,x1w3+x2w2+x3w1,x2w3+x3w2,x3w3]] # r1x5
        Y  = slice(Y_,pad_left,pad_right) = [[x1w1,x1w2+x2w1,x1w3+x2w2+x3w1,x2w3+x3w2,x3w3]]  # r1x5
    consider a 1-D deconv with input X=[x1,x2,x3], kernel=[w1,w2,w3], stride = 1, padding='same',so
        consider 
        NMW.T = [[w1,w2,w3, 0, 0,...],
                 [0 ,w1,w2,w3, 0,...],
                 [0 , 0,w1,w2,w3,...],
                 ...,
                 ...,] #rMxN
        since input X is 3-D tensor
        from output_padding=0
            get output_length = 3*1 = 3
            get temp_output_length = output_length+pad_left+pad_right = 5 #get the padded length
            pad_left = 3//2
            pad_right = 3-3//2 -1
        grab r3x5 matrix from NMW.T
        W.T = [[w1,w2,w3, 0, 0,],
               [ 0,w1,w2,w3, 0,],
               [ 0, 0,w1,w2,w3,]] #r3x5 
        temp_output = Y_ =  X@W.T = [[w1,x1w2+x2w1,x1w3+x2w2+x3w1,x2w3+x3w2,w3]] # r1x5
        Y  = slice(Y_,pad_left,pad_right) = [[x1w2+x2w1,x1w3+x2w2+x3w1,x2w3+x3w2]] # r1x3
        NOTE In above case, W.T was sliced to calculated with X for correct output Y(correct output_length and correct value). 
        Actully, specific deconv op is determined by c++ ops, such as 'Conv2DBackpropInput', 'Conv3DBackpropInput', etc.
        This case is not the only explanation. Another explanation is to slice Y for correct output. 
        But the principle is, deconv never pad inputs. 
    由此可见 deconv(·)的过程是conv(·)过程的完全反向
        conv(·) input -> padded_input -> @W -> output
        deconv() output(sliced) <- padded_output <- @W.T <- input
        先依据output_padding 确定 output_length
        再依据padding反卷积 确定 pad_left,pad_right 值与conv的padding_behavior中的规定相同
            那么 就得到了 temp_output_length
        由于X@W.T 就是为了得到 同temp_output_length的维度的向量即 padded_output
            所以 将依据卷积核和stride密铺的 NMW.T从头攫取对应的维度(length(X) and temp_output_length)
        由 X@W.T 得到 padded_output
        再 依据 pad_left,pad_right 得到 padded_output切片 即output(sliced) 就是转置卷积的输出
    因此 deconv 的padding output_padding 都并非指示一个 tf.pad()之类的过程 
    因此 反卷积过程中的padding无法也不必做等价
"""
import logging
import tensorflow as tf 
from functools import wraps
from typeguard import typechecked
from typing import Iterable
# def typeguard_for_conv_helper(func):
#     def _arg_trans(input_arg):
#         """
#         specify for conv helper
#         trans input_shape,filter,kernel_size to int 
#         trans padding to str in lowercase
#         """
#         if isinstance(input_arg,list) or isinstance(input_arg,tuple) or isinstance(input_arg,tf.TensorShape):
#             if len(input_arg)>1 or len(input_arg)<=0:
#                 raise ValueError(f"input arg must be 1-dim, not {input_arg}.")
#             else:
#                 if hasattr(input_arg,'as_list'):
#                     out_put = input_arg.as_list()[0]
#                 else:
#                     out_put =  input_arg[0]
#             if out_put is None: # To avoid None in [None,shape1,shape2]
#                 raise ValueError("input_arg[0] must be a element but not a None")
#             else:
#                 return int(out_put)
#         elif isinstance(input_arg,int) or isinstance(input_arg,float) or isinstance(input_arg,tf.Tensor):
#             return  int(input_arg)
#         elif isinstance(input_arg,str):
#             return  input_arg.lower()
#         else:
#             raise ValueError("input_arg must be a list, tuple, TensorShape, int, float or Tensor")
#     @wraps(func)
#     def wrappered(*args,**kwargs):
#         args = list(map(_arg_trans,args))
#         for key,value in kwargs.items():
#             kwargs[key] = _arg_trans(value)
#         _output = func(*args,**kwargs)
#         return _output
#     return wrappered

@typechecked
def get_conv_paddings(input_length:int,filter_size:int,stride:int,dilation_rate:int,padding:str):
    """
    Give out equivalent conv paddings from current padding to VALID padding.
    For example, there is a conv(X,padding='same'), find the equivalent conv paddings and
    make conv(X,padding='same')===conv(pad(X,equivalent_conv_paddings),padding='VALID')
   
    see the:
    https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
    If padding == 'SAME': output_shape = ceil(input_length/stride)
    If padding == 'VALID': output_shape = ceil((input_length-(filter_size-1)*dilation_rate)/stride)

    NOTE In conv op, FULL padding, CUSUAL padding and VALID padding have explicit padding prodedure. 
    We can easliy give out the equivalent conv paddings without know the input_length.
    However, it 'SAME' is very special and ambiguous, beacuse its padding prodedure is influenced by 
    input_length and stride to ensure "output_shape===ceil(input_length/stride)". So we should give out a equivalent
    padding prodedure to find the equivalent conv paddings from 'SAME' to 'VALID' finally.
    Consider the equation, where pad_length is should known first:
        ceil(input_length/stride) == ceil((input_length+pad_length-(filter_size-1)*dilation_rate)/stride)
        ceil function makes the pad_length not unique.
        but by testing, 
        we find tensorflow (C++ API) 'SAME' padding's feature:
        1. always choose the minimum pad_length.
        2. divide pad_length equally to pad_left and pad_right and make pad_left<=pad_right
    using function conv_output_length()'s 'another2' artificial_padding_behavior
    return  paddings(padding vectors) for conv's padding behaviour
    """
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation_rate - 1)
    padding = padding.lower()
    if padding in 'valid':
        pad_left =  0
        pad_right = 0
    elif padding == 'causal': 
        pad_left =  dilated_filter_size - 1
        pad_right = 0
    elif padding == 'same': 
        flag = input_length%stride
        if flag==0:
            pad_all= max(dilated_filter_size-stride,0)
        else:
            pad_all= max(dilated_filter_size-flag,0)
        pad_left =  pad_all // 2
        pad_right = pad_all - pad_left
    elif padding == 'full': # full padding has been deprecated in many conv or deconv layers
        pad_left =  dilated_filter_size - 1
        pad_right = dilated_filter_size - 1
    else:
        raise ValueError("Padding should in 'valid', 'causal', 'same' or 'full', not {}.",format)
    return [pad_left,pad_right]



@typechecked
def conv_output_length(input_length:int,filter_size:int,padding:str,stride:int,dilation:int=1):
    """Determines output length of a convolution given input length.
    ANCHOR 问题2 已经表明
    在4点基本假设的前提下
    存在至少M种artificial_padding_behavior 
    使得output_length(artificial_padding_conv)===output_length(padding_conv)
    因此
    可以规定一个 artificial_padding_behavior 继而得到一个 artificial_padding_conv 计算output_length
    因此 本函数mimic了keras中不对外开放的output_length计算方法 并给予了解释 
    Args:
        input_length: integer.
        filter_size: integer.
        padding: one of 'same', 'valid', 'full', 'causal'
        stride: integer.
        dilation: dilation rate, integer.
    Returns:
        The output length (integer).
    basic
        由 问题2 和 问题1 可知
        output_length = (ALL_L + stride - 1) // stride
        ALL_L == input_length+left_pad+right_pad-filter_size+1
        规定一个artificial_padding_behavior 
        为了尽可能统一
            当padding == valid或full或causal时 
                即使 存在其他合理的artificial_padding_behavior 也可使得output_length(artificial_padding_conv)===output_length(padding_conv) 始终成立
                规定 artificial_padding_behavior 与 padding_behavior相同 即同问题1
                即由valid或full或causal自身的定义规定artificial_padding_behavior
            当padding == same时  
                不强制要求artificial_padding_behavior 与 padding_behavior相同
                而是尽可能让最终的计算形式便于统一
        padding == valid 时
            pad_left = 0 
            pad_right = 0
            那么 ALL_L == input_length+0+0-filter_size+1==input_length-filter_size+1
            output_length = (ALL_L + stride - 1) // stride 
            必定与output_length(valid_conv)一致 即结果正确
        padding == full 时
            pad_left = filter_size-1 
            pad_right = filter_size-1 
            那么 ALL_L == input_length+filter_size-1+filter_size-1-filter_size+1 == input_length+filter_size-1
            output_length = (ALL_L + stride - 1) // stride
            必定与output_length(full_conv)一致 即结果正确
        padding == causal时
            pad_left = filter_size-1 
            pad_right = 0
            那么 ALL_L == input_length+filter_size-1-filter_size+1 == input_length
            output_length = (ALL_L + stride - 1) // stride
            必定与output_length(causal_conv)一致 即结果正确
        padding == same时
            pad_left = filter_size//2
            pad_right = filter_size-filter_size//2-1
            那么 ALL_L == input_length+filter_size//2+filter_size-filter_size//2-1-filter_size+1 == input_length
            output_length = (ALL_L + stride - 1) // stride = (input_length+stride-1)//stride
            考察 (input_length+stride-1)//stride === ceil(input_length/stride) 是否成立
            即考察整数等式 (X+S-1)//S===ceil(X/S)是否成立 
            记 X = K*S+T K>=0 0<=T<=S-1
            当 T==0时
                左边= (K*S+S-1)//S = K
                右边= ceil(K*S/S) = K
                左边==右边
            当 1<=T<=S-1 必然有S>=2
                左边= (K*S+T+S-1)//S = K+1+(T-1)//S=K+1
                右边= ceil((K*S+T)/S) = ceil(K+T/S)=K+ceil(T/S)=K+1
                左边==右边
            因此 整数等式 (X+S-1)//S===ceil(X/S)成立
            因此 (input_length+stride-1)//stride === ceil(input_length/stride)成立
            从而与output_length(same_conv)一致 即结果正确
    another1:
        若取一个新的 artificial_padding_behavior 满足 output_length(artificial_padding_conv)===output_length(padding_conv)
        相比于 basic 中的 artificial_padding_behavior 其余不变 
        只修改 padding == same 时策略为
            pad_left = filter_size//2
            pad_right = filter_size//2
            那么 ALL_L == input_length+filter_size//2+filter_size//2-filter_size+1
            由于 filter_size >= 1 
            所以 ALL_L == input_length+1 (filter_size为偶数)
                或 ALL_L == input_length (filter_size为奇数)
            当 filter_size为奇数时  basic中已经证明 output_length=(ALL_L + stride - 1)//stride===ceil(input_length/stride)成立
            当 filter_size为偶数时 
                output_length=(input_length+1 + stride - 1)//stride=(input_length+stride)//stride
                考察 (input_length+stride)//stride === ceil(input_length/stride) 是否成立
                即考察整数等式 (X+S)//S===ceil(X/S)是否成立 
                记 X = K*S+T K>=0 0<=T<=S-1
                当 T==0时
                    左边= (K*S+S)//S = K+1
                    右边= ceil(K*S/S) = K
                    左边!=右边
                当 1<=T<=S-1 必然有S>=2
                    左边= (K*S+T+S)//S = K+1+T//S=K+1
                    右边= ceil((K*S+T)/S) = ceil(K+T/S)=K+ceil(T/S)=K+1
                    左边==右边
                因此 整数等式 (X+S-1)//S!==ceil(X/S) 
                因此 (input_length+stride-1)//stride !==ceil(input_length/stride) 
            从而与output_length(same_conv)不一致
            该artificial_padding_behavior 是不正确的
    another2:
        若取一个新的 artificial_padding_behavior 满足 output_length(artificial_padding_conv)===output_length(padding_conv)
        相比于 basic 中的 artificial_padding_behavior 其余不变 
        只修改 padding == same 时策略为
            pad_left = pad_all//2
            pad_right = pad_all - pad_left
            pad_all 为满足 output_length(artificial_padding_conv)===output_length(padding_conv) 非负最小值
            自然也与 output_length(same_conv)一致 即结果正确 这可以尽可能减少pad计算 平等的看待数据的左右
            
            以下过程是具体的pad_all计算推导 并证明满足 output_length(artificial_padding_conv)===output_length(padding_conv) 非负最小值pad_all是可以取道的
                有
                output_length=(input_length+left_pad+right_pad-filter_size+1+stride-1)//stride=(input_length+left_pad+right_pad-filter_size+stride)//stride
                (input_length+pad_all-filter_size+stride)//stride=== ceil(input_length/stride)
                考察 Y//S==ceil(Z/S)
                    当 Z被S整除时 Z-0<=Y<=Z+S-1
                    当 Z不被S整除时 记 S*ceil(Z/S) - Z = M
                        Z+M-0<=Y<=Z+M+S-1
                设 input_length = K*stride+T K>=0 0<=T<=stride-1             
                T == 0 时 
                    input_length<=input_length+pad_all-filter_size+stride<=input_length+stride-1
                    pad_all>=filter_size-stride
                    pad_all<=filter_size-1
                    当 filter_size-stride>=0时
                        取 pad_all== filter_size-stride 为非负最小值
                    当 filter_size-stride<0 时
                        filter_size-1>=0
                        所以0包含于 pad_all范围中
                        取 pad_all== 0
                    所以 
                    pad_all = max(filter_size-stride,0)
                1<=T<=stride-1 必然有stride>=2
                    input_length+(stride-T)<=input_length+pad_all-filter_size+stride<=input_length+(stride-T)+stride-1
                    pad_all>=filter_size-T 
                    pad_all<=filter_size+stride-T-1
                    当 filter_size-T >=0时
                        取 pad_all == filter_size-T 为非负最小值
                    当 filter_size-T <0 时
                        又因为 stride-1>= T 
                        所以 stride-T-1>= 0
                        filter_size+stride-T-1 >= filter_size>=1>0
                        所以0包含于 pad_all范围中
                        取 pad_all== 0
                    所以 
                    pad_all = max(filter_size-T,0)
            存在 pad_all 为满足 output_length(artificial_padding_conv)===output_length(padding_conv) 的非负最小值
            继而存在 artificial_same_behavior
                pad_left = pad_all//2
                pad_right = pad_all - pad_left
                pad_all 为满足 output_length(artificial_padding_conv)===output_length(padding_conv) 的非负最小值
            因此计算过程如下
            0---compute T = input_length%strides
            1---if T==0 then compute pad_all= max(filter_size-stride,0)
                if T>0  then compute pad_all= max(filter_size-T,0)
            2---compute pad_letf = pad_all//2
                compute pad_right = pad_all-pad_left
    NOTE 问题2指出 
    在4点基本假设的前提下
    存在至少M种artificial_padding_behavior 
    使得output_length(artificial_padding_conv)===output_length(padding_conv)
    本函数采用 basic artificial_padding_behavior 并非一定与实践时框架底层的 padding_behavior完全一致 
        给出的 another1 artificial_padding_behavior 是不正确的
        给出的 another2 artificial_padding_behavior 是更贴近 padding_behavior的行为 但底层的设计变动是无法被用户控制的 无法断言artificial_padding_behavior就是padding_behavior
    """
    if input_length is None:
        return None
    assert padding in {'same', 'valid', 'full', 'causal'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding in ['same', 'causal']:
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride

@typechecked  
def conv_input_length(output_length:int,filter_size:int,padding:str,stride:int):
    """Determines input length of a convolution given output length.
    由问题3可知
    在4点基本假设的前提下
    存在至少M种 artificial_padding_behavior 使得 input_length_range(artificial_padding_conv)===input_length_range(padding_conv) 
        计算input_length_range 就是将artificial_padding_behavior下的 
        output_length = (input_length+left_pad+right_pad-filter_size+1+stride-1)//stride
        计算逆向进行
    由问题4可知
    在某个 artificial_padding_behavior 下
        如果规定 p 为 "使返回的input_length为input_length_range的最小(最大)"
        必然有 input_length(artificial_padding_conv,p)===input_length(padding_conv,p) 成立 
        要计算 input_length 即通过input_length(artificial_padding_conv,p)计算input_length
        就是将 artificial_padding_behavior下的 
        output_length = (input_length+left_pad+right_pad-filter_size+1+stride-1)//stride
        计算逆向进行
        找到符合 'p' 要求的input_length
    NOTE tf.keras源码中的 conv_input_length采用了conv_output_length()中规定的 another1 artificial_padding_behavior 是不正确的
    这里进行correction 改为 basic artificial_padding_behavior
    即
    padding == valid 时
        pad_left = 0 
        pad_right = 0
    padding == full 时
        pad_left = filter_size-1 
        pad_right = filter_size-1 
    padding == causal时
        pad_left = filter_size-1 
        pad_right = 0
    padding == same时
        pad_left = filter_size//2
        pad_right = filter_size-filter_size//2-1
    设 p 为 '$1'
    依据 
    output_length = (input_length+left_pad+right_pad-filter_size+1+stride-1)//stride
    对计算过程进行逆向 求出 最小的input_length
    因此可以直接改写为 
    output_length = (input_length+left_pad+right_pad-filter_size+stride)/stride # input_length是变量 其余为已知量 因此input_length最小是在整除与除等价时达到
    input_length = output_length*stride-left_pad-right_pad-stride+filter_size
    即 
    input_length = (output_length-1)*stride-left_pad-right_pad+filter_size
    Args:
        output_length: integer.
        filter_size: integer. means dilated_filter_size
        padding: one of 'same', 'valid', 'full'.
        stride: integer.
    Returns:
        The input length (integer).
    """
    padding = padding.lower()
    if padding == 'same':
        pad_left = filter_size // 2
        pad_right = filter_size-filter_size//2-1
    elif padding == 'valid':
        pad_left = pad_right = 0
    elif padding == 'full':
        pad_left = pad_right = filter_size - 1
    elif padding == 'causal':
        pad_left = filter_size - 1
        pad_right = 0
    else:
        raise ValueError(f"padding must be one of `same`, `valid`, `full`, `causal`, not `{padding}`.")
    return (output_length - 1) * stride - pad_left - pad_right + filter_size

@typechecked
def deconv_output_length(input_length:int,
                         filter_size:int,
                         padding:str,
                         output_padding:int|None,
                         stride:int,
                         dilation:int=1):
    """Determines output length of a transposed convolution given input length.
    
    由问题4可知
    在4点基本假设的前提下
    存在至少M种 artificial_padding_behavior 使得 input_length_range(artificial_padding_conv)===input_length_range(padding_conv) 
    在某个 artificial_padding_behavior 下
        如果规定 p 为 "使返回的input_length为input_length_range的最小(最大)"
            必然有 input_length(artificial_padding_conv,p)===input_length(padding_conv,p) 成立 
        如果规定 p 为 "padding==valid或者same或者causal时返回最接近output_length*stride的input_length padding==full时返回最小input_length"
            必然有 input_length(artificial_padding_conv,p)===input_length(padding_conv,p) 成立 
        要计算input_length 即通过input_length(artificial_padding_conv,p)计算input_length
        就是将artificial_padding_behavior下的 
        output_length = (input_length+left_pad+right_pad-filter_size+1+stride-1)//stride
        计算逆向进行
        找到符合 'p' 要求的input_length

    问题5指出 存在若干合理的 deconv(·)计算output_length的过程
    本函数mimic了和tf.keras源码一致的output_length计算过程
        用户有output_padding需求时
            先规定artificial_padding_behavior为 conv_output_length() 中的 basic artificial_padding_behavior
            规定 p 为 '$1'
            input_length(artificial_padding_conv,p)可以给出最小的input_length 同conv_input_length()函数 不再赘述
            即 
            _conv_input_length = conv_input_length(...)
            _conv_input_length += output_padding
        用户没有output_padding需求时 
            先规定artificial_padding_behavior为 conv_output_length() 中的 basic artificial_padding_behavior
            规定 p 为 "padding==valid或者same或者causal时返回最接近output_length*stride的input_length padding==full时返回最小input_length"   
            padding == 'full'时 
                pad_left = pad_right = filter_size - 1
                output_length = (input_length+left_pad+right_pad-filter_size+1+stride-1)//stride
                              = (input_length+filter_size+stride-2)//stride
                output_length*stride<=input_length+filter_size+stride-2<=output_length*stride+(stride-1)
                output_length*stride-filter_size-stride+2<=input_length<=output_length*stride-filter_size+1
                取最小 input_length = output_length*stride-filter_size-stride+2
                即
                _conv_input_length = _conv_output_length*stride-filter_size-stride+2
            padding == 'valid'时
                pad_left = pad_right = 0
                output_length = (input_length+left_pad+right_pad-filter_size+1+stride-1)//stride
                                    = (input_length-filter_size+stride)//stride
                output_length*stride<=input_length-filter_size+stride<=output_length*stride+(stride-1)
                output_length*stride+(filter_size-stride)<=input_length<=output_length*stride+(stride-1)+(filter_size-stride)
                当 filter_size>=stride时 最接近output_length*stride的值为
                    即 input_length = output_length*stride+(filter_size-stride)
                当 filter_size<stride时 
                    由于此时
                    output_length*stride+(filter_size-stride) < output_length*stride
                    output_length*stride+(stride-1)+(filter_size-stride) = output_length*stride+filter_size-1>=output_length*stride
                    output_length*stride 符合区间范围
                    input_length = output_length*stride
                所以
                input_length = output_length*stride+max(filter_size-stride,0) 是最终形式
                即
                _conv_input_length = _conv_output_length*stride+max(filter_size-stride,0)
            padding == 'causal'时  NOTE tf.keras 源码中无该情况 这里添加该情况
                pad_left = filter_size - 1
                pad_right = 0 
                output_length = (input_length+left_pad+right_pad-filter_size+1+stride-1)//stride
                                    = (input_length+stride-1)//stride
                output_length*stride<=input_length+stride-1<=output_length*stride+(stride-1) 
                output_length*stride-stride+1<=input_length<=output_length*stride
                取最接近output_length*stride的值为
                input_length = output_length*stride
                即
                _conv_input_length = _conv_output_length*stride
            padding == 'same'时
                pad_left = filter_size//2
                pad_right = filter_size-filter_size//2-1 
                output_length = (input_length+left_pad+right_pad-filter_size+1+stride-1)//stride
                                    = (input_length+stride-1)//stride
                output_length*stride<=input_length+stride-1<=output_length*stride+(stride-1) 
                output_length*stride-stride+1<=input_length<=output_length*stride
                取最接近output_length*stride的值为
                input_length = output_length*stride
                即
                _conv_input_length = _conv_output_length*stride  
    Args:
        input_length: Integer.
        filter_size: Integer.
        padding: one of `same`, `valid`, `full`, `causal`.
        output_padding: Integer, amount of padding along the output dimension. Can
            be set to `None` in which case the output length is inferred.
        stride: Integer.
        dilation: Integer.
    Returns:
        The output length (integer).
    """
    _conv_output_length = input_length
    # Get the dilated kernel size
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    # Infer length if output padding is None, else compute the exact length
    if output_padding is None:
        if padding == 'valid':
            _conv_input_length = _conv_output_length * stride + max(dilated_filter_size - stride, 0)
        elif padding == 'full':
            _conv_input_length = _conv_output_length * stride - (stride + dilated_filter_size - 2)
        elif padding == 'causal':
            _conv_input_length = _conv_output_length * stride
        elif padding == 'same':
            _conv_input_length = _conv_output_length * stride
        else:
            raise ValueError(f"padding must be one of `same`, `valid`, `full`, `causal`, not `{padding}`.")
    else:
        _conv_input_length = conv_input_length(output_length=_conv_output_length,filter_size=dilated_filter_size,padding=padding,stride=stride)
        _conv_input_length += output_padding
    return _conv_input_length

@typechecked
def get_padded_length_from_paddings(length:int|None,paddings:tuple[int,int]):
    if length is not None:
        pad_left,pad_right = paddings
        length = length + pad_left + pad_right
    return length
@typechecked
def normalize_paddings_by_data_format(data_format:str,paddings:Iterable):
    out_buf = []
    for data_format_per_dim in data_format:
        if data_format_per_dim.upper() in ['N','C']:
            out_buf.append(tuple([0,0]))
        elif data_format_per_dim.upper() in ['D','H','W']:
            out_buf.append(tuple(next(paddings)))
        else:
            raise ValueError(f"data_format should consist with `N`, `C`, `D`, `H` or `W` but not `{out_buf}`.")
    return out_buf
@typechecked
def grab_length_by_data_format(data_format:str,length:tuple|list):
    out_buf = []
    for data_format_per_dim,length_per_dim in zip(data_format,length):
        if data_format_per_dim.upper() in ['N','C']:
            pass
        elif data_format_per_dim.upper() in ['D','H','W']:
            out_buf.append(int(length_per_dim))
        else:
            raise ValueError(f"data_format should consist with `N`, `C`, `D`, `H` or `W` but not `{out_buf}`.")
    return out_buf
def normalize_padding(value):
    if isinstance(value, (list, tuple)):
        return value
    padding = value.lower()
    if padding not in {'valid', 'same', 'causal','full'}:
        raise ValueError(f"The `padding` argument must be a list/tuple or one of `valid`, `same`, `full` (or `causal`, only for `Conv1D`). Received: {padding}")
    return padding
def normalize_specific_padding_mode(value):
    if isinstance(value, (list, tuple)):
        return value
    padding = value.lower()
    if padding not in {'constant', 'reflect', 'symmetric'}:
        raise ValueError(f"The `padding` argument must be a list/tuple or one of `constant`, `reflect` or `symmetric`. Received: {padding}")
    return padding.upper()
def normalize_tuple(value, n, name, allow_zero=False):
    """Transforms non-negative/positive integer/integers into an integer tuple.
    Args:
        value: The value to validate and convert. Could an int, or any iterable of
        ints.
        n: The size of the tuple to be returned.
        name: The name of the argument being validated, e.g. 'strides' or
        'kernel_size'. This is only used to format error messages.
        allow_zero: Default to False. A ValueError will raised if zero is received
        and this param is False.
    Returns:
        A tuple of n integers.
    Raises:
        ValueError: If something else than an int/long or iterable thereof or a
        negative value is
        passed.
    """
    error_msg = (f"The `{name}` argument must be a tuple of {n} "
                f"integers. Received: {value}")

    if isinstance(value, int):
        value_tuple = (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError(error_msg)
        if len(value_tuple) != n:
            raise ValueError(error_msg)
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                error_msg += (f"including element {single_value} of "
                            f"type {type(single_value)}")
                raise ValueError(error_msg)
    if allow_zero:
        unqualified_values = {v for v in value_tuple if v < 0}
        req_msg = '>= 0'
    else:
        unqualified_values = {v for v in value_tuple if v <= 0}
        req_msg = '> 0'
    if unqualified_values:
        error_msg += (f" including {unqualified_values}"
                    f" that does not satisfy the requirement `{req_msg}`.")
        raise ValueError(error_msg)

    return value_tuple
