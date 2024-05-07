class dataset--->ava.py( classe Dataset Pytorch)
	           --->transform.py 

evaluator--->ava_eval_helper.py
      	 --->ava_evaluator.py (Étapes de validation)
      	 --->cal_frame_mAP.py (calcul de métrique mAP_frame)
      	 --->cal_video_mAP.py (calcul de métrique mAP_video) 
      	 --->utils.py

model--->Contient les architectures de modèle, loss
utils---> misc.py (Créer un dataloader, créer un evaluator , calculer  focal loss.) , distributed_utils.py (mode distributed)....)
