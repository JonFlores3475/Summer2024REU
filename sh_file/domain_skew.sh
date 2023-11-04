#python main.py --task domain_skew --dataset VLCS --method FedAVG --device_id 0 --csv_log  --csv_name mu_1 --save_checkpoint &
#python main.py --task domain_skew --dataset VLCS --method Scaffold --device_id 1 --csv_log --csv_name mu_1 --save_checkpoint &
#python main.py --task domain_skew --dataset VLCS --method MOON --device_id 2 --csv_name mu_1 --csv_log  --save_checkpoint &
#python main.py --task domain_skew --dataset VLCS --method FedProc --device_id 3 --csv_log --csv_name mu_1 --save_checkpoint &
#python main.py --task domain_skew --dataset VLCS --method FedDyn --device_id 5  --csv_log  --csv_name mu_1 --save_checkpoint &
#python main.py --task domain_skew --dataset VLCS --method FedProx --device_id 4 --csv_log --csv_name mu_1 --save_checkpoint &
#python main.py --task domain_skew --dataset VLCS --method FedOpt --device_id 6 --csv_log --csv_name mu_1 --save_checkpoint &
#python main.py --task domain_skew --dataset VLCS --method FedProto --device_id 7 --csv_log --csv_name mu_1 --save_checkpoint &
#python main.py --task domain_skew --dataset VLCS --method FedNTD --device_id 2 --csv_log --csv_name mu_1 --save_checkpoint &
#python main.py --task domain_skew --dataset VLCS --method FedNova --device_id 1 --csv_log  --save_checkpoint --csv_name mu_1 &
#python main.py --task domain_skew --dataset VLCS --method RHFL --device_id 3 --csv_log  --save_checkpoint --csv_name mu_1 &


#python main.py --task domain_skew --dataset PACS --method Scaffold --device_id 0 --csv_log  --save_checkpoint --csv_name mu_1 &
#python main.py --task domain_skew --dataset Digits --method Scaffold --device_id 1 --csv_log  --save_checkpoint --csv_name mu_1 &
#python main.py --task domain_skew --dataset OfficeCaltech --method Scaffold --device_id 2 --csv_log  --save_checkpoint --csv_name mu_1 &
#
#


#python main.py --task domain_skew --dataset PACS --method qffeAVG --device_id 0 --csv_log  --save_checkpoint --csv_name q_0 Local.qffeAVGLocal.q 0.0 &
#python main.py --task domain_skew --dataset PACS --method qffeAVG --device_id 0 --csv_log  --save_checkpoint --csv_name q_1 Local.qffeAVGLocal.q 1.0 &
#python main.py --task domain_skew --dataset PACS --method qffeAVG --device_id 1 --csv_log  --save_checkpoint --csv_name q_2 Local.qffeAVGLocal.q 2.0 &
#
#python main.py --task domain_skew --dataset OfficeCaltech --method qffeAVG --device_id 1 --csv_log  --save_checkpoint --csv_name q_0 Local.qffeAVGLocal.q 0.0 &
#python main.py --task domain_skew --dataset OfficeCaltech --method qffeAVG --device_id 2 --csv_log  --save_checkpoint --csv_name q_1 Local.qffeAVGLocal.q 1.0 &
#python main.py --task domain_skew --dataset OfficeCaltech --method qffeAVG --device_id 2 --csv_log  --save_checkpoint --csv_name q_2 Local.qffeAVGLocal.q 2.0 &
#
#python main.py --task domain_skew --dataset Digits --method qffeAVG --device_id 3 --csv_log  --save_checkpoint --csv_name q_0 Local.qffeAVGLocal.q 0.0 &
#python main.py --task domain_skew --dataset Digits --method qffeAVG --device_id 3 --csv_log  --save_checkpoint --csv_name q_1 Local.qffeAVGLocal.q 1.0 &
#python main.py --task domain_skew --dataset Digits --method qffeAVG --device_id 3 --csv_log  --save_checkpoint --csv_name q_2 Local.qffeAVGLocal.q 2.0 &
#
#python main.py --task domain_skew --dataset PACS --method AFL --device_id 4 --csv_log  --save_checkpoint &
#python main.py --task domain_skew --dataset OfficeCaltech --method AFL --device_id 4 --csv_log  --save_checkpoint &
python main.py --task domain_skew --dataset Digits --method AFL --device_id 5 --csv_log  --save_checkpoint &

#python main.py --task domain_skew --dataset PACS --method FedDf --device_id 6 --csv_log  --save_checkpoint --csv_name mu_1 &
#python main.py --task domain_skew --dataset Digits --method FedDf --device_id 7 --csv_log  --save_checkpoint --csv_name mu_1 &
#python main.py --task domain_skew --dataset OfficeCaltech --method FedDf --device_id 0 --csv_log  --save_checkpoint --csv_name mu_1 &
#
#
#python main.py --task domain_skew --dataset PACS --method FcclPlus --device_id 1 --csv_log  --save_checkpoint --csv_name mu_1 &
#python main.py --task domain_skew --dataset Digits --method FcclPlus --device_id 2 --csv_log  --save_checkpoint --csv_name mu_1 &
#python main.py --task domain_skew --dataset OfficeCaltech --method FcclPlus --device_id 3 --csv_log  --save_checkpoint --csv_name mu_1 &



#'''PACS'''
#python main.py --task domain_skew --dataset PACS \
#  --method FedAVG --device_id 0 --csv_log --csv_name 4_0.001 --save_checkpoint DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.001 &
#
#python main.py --task domain_skew --dataset PACS \
#  --method FedProx --device_id 1 --csv_log --csv_name 4_0.001_mu_0.01 --save_checkpoint DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.001 Local.FedProxLocal.mu 0.01&
#
#python main.py --task domain_skew --dataset PACS \
#  --method FedProx --device_id 2 --csv_log --csv_name 4_0.001_mu_0.001 --save_checkpoint DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.001 Local.FedProxLocal.mu 0.001&
#
#python main.py --task domain_skew --dataset PACS \
#  --method FedAVG --device_id 3 --csv_log --csv_name 4_0.005 --save_checkpoint DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.005 &
#
#python main.py --task domain_skew --dataset PACS \
#  --method FedProx --device_id 4 --csv_log --csv_name 4_0.005_mu_0.01 --save_checkpoint DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.005 Local.FedProxLocal.mu 0.01&
#
#python main.py --task domain_skew --dataset PACS \
#  --method FedProx --device_id 5 --csv_log --csv_name 4_0.005_mu_0.001 --save_checkpoint DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.005 Local.FedProxLocal.mu 0.001&

#'''PACS'''
#python main.py --task domain_skew --dataset PACS \
#  --method FedAVG --device_id 2 --csv_log --csv_name 4_0.001_0.5 --save_checkpoint DATASET.domain_ratio 0.5 DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.001 &

#python main.py --task domain_skew --dataset PACS \
#  --method FedProx --device_id 0 --csv_log --csv_name 4_0.001_mu_0.01_0.5 --save_checkpoint DATASET.domain_ratio 0.3 DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.001 Local.FedProxLocal.mu 0.01&
#
#python main.py --task domain_skew --dataset PACS \
#  --method FedProx --device_id 1 --csv_log --csv_name 4_0.005_mu_0.01_0.5 --save_checkpoint DATASET.domain_ratio 0.3 DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.005 Local.FedProxLocal.mu 0.01&

#python main.py --task domain_skew --dataset PACS \
#  --method FedProx --device_id 2 --csv_log --csv_name 4_0.001_mu_0.01_0.1 --save_checkpoint DATASET.domain_ratio 0.1 DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.001 Local.FedProxLocal.mu 0.01&
#wait
#python main.py --task domain_skew --dataset PACS \
#  --method FedProx --device_id 0 --csv_log --csv_name 4_0.005_mu_0.01_0.1 --save_checkpoint DATASET.domain_ratio 0.1 DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.005 Local.FedProxLocal.mu 0.01&

#'''pacs'''
#python main.py --task domain_skew --dataset PACS \
#  --method FedAVG --device_id 3 --csv_log --csv_name 4_0.001_0.5 --save_checkpoint DATASET.domain_ratio 0.5 DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.001 &

#wait
#python main.py --task domain_skew --dataset PACS \
#  --method FedProx --device_id 0 --csv_log --csv_name 4_0.001_mu_0.01_0.5 --save_checkpoint DATASET.domain_ratio 0.3 DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.001 Local.FedProxLocal.mu 0.01&

#python main.py --task domain_skew --dataset PACS \
#  --method FedProx --device_id 1 --csv_log --csv_name 4_0.005_mu_0.01_0.5 --save_checkpoint DATASET.domain_ratio 0.3 DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.005 Local.FedProxLocal.mu 0.01&

#python main.py --task domain_skew --dataset PACS \
#  --method FedProx --device_id 6 --csv_log --csv_name 4_0.001_mu_0.01_0.1 --save_checkpoint DATASET.domain_ratio 0.1 DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.001 Local.FedProxLocal.mu 0.01&
#
#python main.py --task domain_skew --dataset PACS \
#  --method FedProx --device_id 7 --csv_log --csv_name 4_0.005_mu_0.01_0.1 --save_checkpoint DATASET.domain_ratio 0.1 DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.005 Local.FedProxLocal.mu 0.01&

#python main.py --task domain_skew --dataset PACS \
#  --method FedProx --device_id 0 --csv_log --csv_name 4_0.001_mu_0.01_0.3 --save_checkpoint DATASET.domain_ratio 0.3 DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.001 Local.FedProxLocal.mu 0.01&
#
#python main.py --task domain_skew --dataset PACS \
#  --method FedProx --device_id 1 --csv_log --csv_name 4_0.001_mu_0.01_0.3 --save_checkpoint DATASET.domain_ratio 0.3 DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.001 Local.FedProxLocal.mu 0.01&
#wait
#python main.py --task domain_skew --dataset PACS \
#  --method FedProx --device_id 0 --csv_log --csv_name 4_0.005_mu_0.01_0.3 --save_checkpoint DATASET.domain_ratio 0.3 DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.005 Local.FedProxLocal.mu 0.01&
#
#python main.py --task domain_skew --dataset PACS \
#  --method FedProx --device_id 1 --csv_log --csv_name 4_0.005_mu_0.01_0.3 --save_checkpoint DATASET.domain_ratio 0.3 DATASET.parti_num 4 OPTIMIZER.local_train_lr 0.005 Local.FedProxLocal.mu 0.01&
