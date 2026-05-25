本任务为基于Lstm的eVTOL轨迹预测，数据集来自https://huggingface.co/datasets/riotu-lab/Synthetic-UAV-Flight-Trajectories中的3维轨迹数据
在数据预处理部分，对整个数据集进行分割，得到5309段轨迹数据。采用分层聚类结合动态时间规整（DTW)将所有的轨迹分为circular和infinity like 两类
模型结构采用Sequence-to-Sequence，用两个LSTM分别作为编码器和解码器，对未来一段时间的轨迹进行预测
