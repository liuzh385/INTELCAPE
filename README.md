

##### Capsule endoscopy segmentation文件夹中存放了第一步进行小肠分段的代码

* ``expconfigs_new/exp02_f0_pretrain_th_gau.yaml``是预训练编码器进行三分类，``CUDA_VISIBLE_DEVICES=0 python main.py --config expconfigs_new/exp02_f0_pretrain_th_gau.yaml``进行训练
* ``expconfigs_new/exp03_f0_resTFE_gau.yaml``是训练CNN+Transformer进行三分类，``CUDA_VISIBLE_DEVICES=0 python main.py --config expconfigs_new/exp03_f0_resTFE_gau.yaml``进行训练
* ``expconfigs_new/test_exp03_f0_resTFE_gau.yaml``是进行二分类搜索找到小肠起点和终点，``CUDA_VISIBLE_DEVICES=0 python main.py --config expconfigs_new/test_exp03_f0_resTFE_gau.yaml --test``进行测试

视频文件格式要个文件夹里放一个视频（格式为avi），然后命名格式为``名字的最后一个字对应的Unicode 20xx_xx_xx``.使用prepare/find_frame.py得到胃、小肠大肠起始时间在视频里对应的起始帧（因为视频传输会掉帧，所以只能通过视频帧右上角记录的时间来找对应帧）。csv文件需要与Data/data_example.csv格式一致。

##### Small intestine frame lesion classification文件夹中存放了第二部进行病变识别的代码

* ``CUDA_VISIBLE_DEVICES=0 python train.py``训练二分类
* ``CUDA_VISIBLE_DEVICES=0 python infer.py``测试二分类

图片文件格式要与data文件夹下的格式保存一致

``SingleObjectLocalization``文件夹存放了弱监督识别的代码，``Train.sh``用来训练，``Test.sh``用来测试生成热图和标注框并且获取测试指标

##### Lesion small intestine frame Crohn's diagnosis文件中中存放了第三步进行克罗恩病识别的代码

* ``expconfigs_new/test01_get_framsnpy.yaml``得到每个病人每张小肠帧的特征的npy，以及小肠帧对应的预测的病变概率。提取特征使用的上一步的Efficientnet的权重。``CUDA_VISIBLE_DEVICES=0 python main.py --config expconfigs_new/test01_get_framsnpy.yaml --test`` 进行提取特征
* ``expconfigs_new/test02_gettop2000.yaml``把小肠段分成4段，每段里面取预测病变概率最高的500帧。``CUDA_VISIBLE_DEVICES=0 python main.py --config expconfigs_new/test02_gettop2000.yaml --test``进行分段
* ``expconfigs_new/exp02_focalloss.yaml``训练TF2进行克罗恩病二分类。``CUDA_VISIBLE_DEVICES=0 python main.py --config expconfigs_new/exp02_focalloss.yaml``训练二分类
* ``expconfigs_new/test03_focalloss.yaml``测试克罗恩病分类效果。``CUDA_VISIBLE_DEVICES=0 python main.pt --config expconfigs_new/test03_focalloss.yaml --test``进行测试





