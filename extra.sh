#python main.py --dataset Digits --method FedAVG --OOD MNIST --device_id 0 --csv_log --save_checkpoint &
#python main.py --dataset Digits --method FedProx --OOD MNIST --device_id 0 --csv_log --save_checkpoint --csv_name mu0.01 FedProx.mu 0.01&
#python main.py --dataset Digits --method FedProxCOSNHNew --OOD MNIST --device_id 0 --csv_log --save_checkpoint --csv_name mu0.01lr0.01_1_0 FedProxCOSNHNew.mu 0.01 FedProxCOSNHNew.weight_lr 1e-2 FedProxCOSNHNew.alpha 1.0 FedProxCOSNHNew.beta 0.0 &

#python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD MNIST --device_id 1 --csv_log --save_checkpoint --csv_name W1.0T10.0mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 10.0 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD MNIST --device_id 1 --csv_log --save_checkpoint --csv_name W1.0T5.0mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 5.0 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD MNIST --device_id 1 --csv_log --save_checkpoint --csv_name W1.0T1.0mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 1.0 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD MNIST --device_id 2 --csv_log --save_checkpoint --csv_name W1.0T0.5mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 0.5 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD MNIST --device_id 2 --csv_log --save_checkpoint --csv_name W1.0T0.1mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 0.1 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD MNIST --device_id 2 --csv_log --save_checkpoint --csv_name W1.0T0.05mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 0.05 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD MNIST --device_id 2 --csv_log --save_checkpoint --csv_name W1.0T0.01mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 0.01 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &

#python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD MNIST --device_id 3 --csv_log --save_checkpoint --csv_name W1.0T10.0mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 10.0 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD MNIST --device_id 3 --csv_log --save_checkpoint --csv_name W1.0T5.0mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 5.0 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD MNIST --device_id 3 --csv_log --save_checkpoint --csv_name W1.0T1.0mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 1.0 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD MNIST --device_id 4 --csv_log --save_checkpoint --csv_name W1.0T0.5mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 0.5 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD MNIST --device_id 4 --csv_log --save_checkpoint --csv_name W1.0T0.1mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 0.1 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD MNIST --device_id 4 --csv_log --save_checkpoint --csv_name W1.0T0.05mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 0.05 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD MNIST --device_id 4 --csv_log --save_checkpoint --csv_name W1.0T0.01mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 0.01 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &

#python main.py --dataset Digits --method FedAVG --OOD SVHN --device_id 0 --csv_log --save_checkpoint &
#python main.py --dataset Digits --method FedProx --OOD SVHN --device_id 0 --csv_log --save_checkpoint --csv_name mu0.01 FedProx.mu 0.01&
#python main.py --dataset Digits --method FedProxCOSNHNew --OOD SVHN --device_id 0 --csv_log --save_checkpoint --csv_name mu0.01lr0.01_1_0 FedProxCOSNHNew.mu 0.01 FedProxCOSNHNew.weight_lr 1e-2 FedProxCOSNHNew.alpha 1.0 FedProxCOSNHNew.beta 0.0 &

#python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD SVHN --device_id 5 --csv_log --save_checkpoint --csv_name W1.0T10.0mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 10.0 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD SVHN --device_id 5 --csv_log --save_checkpoint --csv_name W1.0T5.0mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 5.0 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD SVHN --device_id 5 --csv_log --save_checkpoint --csv_name W1.0T1.0mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 1.0 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD SVHN --device_id 6 --csv_log --save_checkpoint --csv_name W1.0T0.5mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 0.5 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD SVHN --device_id 6 --csv_log --save_checkpoint --csv_name W1.0T0.1mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 0.1 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD SVHN --device_id 6 --csv_log --save_checkpoint --csv_name W1.0T0.05mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 0.05 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD SVHN --device_id 6 --csv_log --save_checkpoint --csv_name W1.0T0.01mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 0.01 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &

#python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD SVHN --device_id 7 --csv_log --save_checkpoint --csv_name W1.0T10.0mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 10.0 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD SVHN --device_id 7 --csv_log --save_checkpoint --csv_name W1.0T5.0mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 5.0 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD SVHN --device_id 7 --csv_log --save_checkpoint --csv_name W1.0T1.0mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 1.0 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD SVHN --device_id 5 --csv_log --save_checkpoint --csv_name W1.0T0.5mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 0.5 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD SVHN --device_id 7 --csv_log --save_checkpoint --csv_name W1.0T0.1mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 0.1 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD SVHN --device_id 7 --csv_log --save_checkpoint --csv_name W1.0T0.05mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 0.05 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
#python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD SVHN --device_id 7 --csv_log --save_checkpoint --csv_name W1.0T0.01mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 0.01 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &

#wait

python main.py --dataset Digits --method FedAVG --OOD USPS --device_id 0 --csv_log --save_checkpoint &
python main.py --dataset Digits --method FedProx --OOD USPS --device_id 0 --csv_log --save_checkpoint --csv_name mu0.01 FedProx.mu 0.01&
python main.py --dataset Digits --method FedProxCOSNHNew --OOD USPS --device_id 0 --csv_log --save_checkpoint --csv_name mu0.01lr0.01_1_0 FedProxCOSNHNew.mu 0.01 FedProxCOSNHNew.weight_lr 1e-2 FedProxCOSNHNew.alpha 1.0 FedProxCOSNHNew.beta 0.0 &

python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD USPS --device_id 1 --csv_log --save_checkpoint --csv_name W1.0T10.0mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 10.0 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &
python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD USPS --device_id 1 --csv_log --save_checkpoint --csv_name W1.0T5.0mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 5.0 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &
python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD USPS --device_id 1 --csv_log --save_checkpoint --csv_name W1.0T1.0mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 1.0 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &
python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD USPS --device_id 2 --csv_log --save_checkpoint --csv_name W1.0T0.5mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 0.5 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &
python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD USPS --device_id 2 --csv_log --save_checkpoint --csv_name W1.0T0.1mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 0.1 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &
python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD USPS --device_id 2 --csv_log --save_checkpoint --csv_name W1.0T0.01mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 0.01 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &

python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD USPS --device_id 3 --csv_log --save_checkpoint --csv_name W1.0T10.0mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 10.0 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD USPS --device_id 3 --csv_log --save_checkpoint --csv_name W1.0T5.0mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 5.0 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD USPS --device_id 3 --csv_log --save_checkpoint --csv_name W1.0T1.0mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 1.0 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD USPS --device_id 4 --csv_log --save_checkpoint --csv_name W1.0T0.5mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 0.5 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD USPS --device_id 4 --csv_log --save_checkpoint --csv_name W1.0T0.1mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 0.1 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD USPS --device_id 4 --csv_log --save_checkpoint --csv_name W1.0T0.01mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 0.01 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &

python main.py --dataset Digits --method FedAVG --OOD SYN --device_id 0 --csv_log --save_checkpoint &
python main.py --dataset Digits --method FedProx --OOD SYN --device_id 0 --csv_log --save_checkpoint --csv_name mu0.01 FedProx.mu 0.01&
python main.py --dataset Digits --method FedProxCOSNHNew --OOD SYN --device_id 0 --csv_log --save_checkpoint --csv_name mu0.01lr0.01_1_0 FedProxCOSNHNew.mu 0.01 FedProxCOSNHNew.weight_lr 1e-2 FedProxCOSNHNew.alpha 1.0 FedProxCOSNHNew.beta 0.0 &

python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD SYN --device_id 5 --csv_log --save_checkpoint --csv_name W1.0T10.0mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 10.0 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &
python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD SYN --device_id 5 --csv_log --save_checkpoint --csv_name W1.0T5.0mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 5.0 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &
python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD SYN --device_id 5 --csv_log --save_checkpoint --csv_name W1.0T1.0mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 1.0 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &
python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD SYN --device_id 6 --csv_log --save_checkpoint --csv_name W1.0T0.5mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 0.5 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &
python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD SYN --device_id 6 --csv_log --save_checkpoint --csv_name W1.0T0.1mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 0.1 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &
python main.py --dataset Digits --method FedProxCOSAddGlobal --OOD SYN --device_id 6 --csv_log --save_checkpoint --csv_name W1.0T0.01mu0.01lr0.01_1_0 FedProxCOSAddGlobal.temperature 0.01 FedProxCOSAddGlobal.mu 0.01 FedProxCOSAddGlobal.weight_lr 1e-2 FedProxCOSAddGlobal.alpha 1.0 FedProxCOSAddGlobal.beta 0.0 &

python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD SYN --device_id 7 --csv_log --save_checkpoint --csv_name W1.0T10.0mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 10.0 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD SYN --device_id 7 --csv_log --save_checkpoint --csv_name W1.0T5.0mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 5.0 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD SYN --device_id 7 --csv_log --save_checkpoint --csv_name W1.0T1.0mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 1.0 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD SYN --device_id 5 --csv_log --save_checkpoint --csv_name W1.0T0.5mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 0.5 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD SYN --device_id 6 --csv_log --save_checkpoint --csv_name W1.0T0.1mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 0.1 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
python main.py --dataset Digits --method FedProxCOSAddNHMad --OOD SYN --device_id 7 --csv_log --save_checkpoint --csv_name W1.0T0.01mu0.01lr0.01_1_0 FedProxCOSAddNHMad.temperature 0.01 FedProxCOSAddNHMad.mu 0.01 FedProxCOSAddNHMad.weight_lr 1e-2 FedProxCOSAddNHMad.alpha 1.0 FedProxCOSAddNHMad.beta 0.0 &
