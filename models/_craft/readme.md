# About this file
The layers crafted from scratch. They involve layers that basic API or addons API have supported.
It is not recommended to use these layers unless your wanted layer is not supported in basic API of TF or pytorch. 


从0开始（从开放的低阶API开始）的layers构建。不仅包括若干基础API中的模型层，也包括若干addons等API中的模型层。
应当尽量避免使用本文件夹下的所有模型层，除非TF、pytorch框架的基础API不支持。
复现本就集成在框架基础API中的模型层，是一个失智行为，无论放在何处，其意义也不大，但这部分是作者在学习过程中积累的，算作一个学习过程的总结。另外，还有一个实际的目的是彰显各个模型层的主要数学逻辑，保证
该数学逻辑的实现与理解相一致。即，如果使用者复现了某个简单的tf.keras.layers.SeparableConv2D为
SeparableConv2D,且其运算结果、运算效率等在若干测试输入中一致，则可以认为，使用者理解了SeparableConv2D的运作原理，在其复杂的实验内容中，没有出现误解与误用的成分。

另外的，即便例如addons等附加的API支持一些模型层，此处也进行了复现。
原因是，附加的API缺乏相对广大的实践基础，且本身依赖一些相对较新，较难理解的论文内容，复现是为了牢牢掌握这部分内容的原理与实现，避免实验方法在不被深刻理解的情况下用于实验中，因此，与上面相反地，推荐使用此处复现的附加API中的模型层。

遵循上述内容，实验方法将具有一下2个特性：
+ 便捷性：不使用复现的基础API，不必关心基础API的底层实现，快速构建实验
+ 准确性：所使用的基础API由TF、pytorch保证准确性，除非他们出问题。附加API也处于受控状态。避免对实验方法的怀疑，转而有更多精力关注实验现象与思路。