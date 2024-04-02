open-gpt-LAL-CAL 
是指开源的个人生成式大语言模型，指在提供一个基底模型，输入个人生活记录语料或资料，创建一个私人的一生陪伴助手，即learn all life 和 accompany all life，目前拥有13万参数量，基于mingpt 
的基础生成模型。 
  训练语料采取text格式，后续将统合json格式，当然你也可以自己修改gpt_tokenizer.py部分。 
  运行文件可以在命令行直接输入：
  python3 gpt_run.py 
后续将完成下面几步工作： 
  1.进一步采用chatglm的模型架构，并使用其量化技术，争取实现chatglm.cpp或llama.cpp类似的独立加速包 
  2.模型多模态接口 
  3.用户友好界面 
    
