## iDNA-ABF-automl
### main
- train_ABF.py 训练调试
- train_model.py

### nni
` 一定要一个数据集单独一个config文件，指定对应的pl_nni_train_*.py`
- fusion: nnictl create -p 9990 -c config_idna.yml
- bert: nnictl create -p 9990 -c config_bert.yml
- searh_space_idna.json[fusion] *固定方便统一搜索空间*
- searh_space_bert.json[bert]

### module
- data_module: onehot:true自动tokenlize / false则直接序列输入
- lightning_module:加入模型地方
### frame

### model
- FusionDNAbert, FGM, DNAbert, ClassificationDNAbert
- TextCNN
- TextRNN
- 
### util
