python3 train_graph.py -gpu 0 -ep 200 -bs 16 -vSize 5120 -norm 0.17 -zdim 64 -hdim 64 -cdim 64 -ccnt 64 -odim 3 -knnk 16 -nsim -adam -lr 0.0003 -fp16 -log logs_shapeNet -name ShapeNet_deep_edgeMask_uniform_LN_std_0_17_z64_c64cd64_bs16 -load auto ../Dataset/MDSets/combined

