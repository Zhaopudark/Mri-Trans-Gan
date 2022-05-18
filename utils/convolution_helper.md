# 讨论卷积过程的各项参数与计算
# 0 基本假设
众所周知 
一个基本的卷积过程是是由一个输入经过一个卷积核的作用得到输出结果的过程
决定卷积核结果的各个参数定义如下

    输入: input 
    输入size: input_length
    卷积核: filter
    步长: stride
    膨胀率: dilation_rate 事实上 膨胀率对卷积参数的计算的影响是可以由改变卷积核和步长等效的
    补齐方式: padding
    输出：output 
    输出size: output_length
    卷积函数: conv(·)
    反卷积(转置卷积)函数：deconv(·) 
        
    为了区分, 我们始终认为在输入输出的size上,deconv是conv的逆过程，因此output_length既是conv的输出size又是deconv的输入size。input_length既是conv的输入size又是deconv的输出size

我们关注卷积过程前后的size的改变情况 改变的过程 因此将卷积过程中的size变化记为: 

    input_length->conv(filter_size,stride,dilation_rate,padding,..)->output_length 
    当conv各项参数确定时,input_length->output_length 是多对一的映射

而卷积的反向过程的size变化记为

    output_length->deconv(filter_size,stride,dilation_rate,padding,..)->input_length 
    当deconv各项参数确定时,output_length->input_length 是一个一对多的映射

# 1 探讨不同padding方式下output_length和input_length的计算过程 
## 1.1 约定如下
    我们用名称戳
    padding 有 valid same full causal 等不同的操作方式, 下文直接以
    以这些操作方式的名称表示,例如
    valid same full causal 
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