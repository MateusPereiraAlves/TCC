python test_univad.py --dataset mvtec --data_path ./data/mvtec --round 0 --image_size 448 --k_shot 1;
python test_univad.py --dataset visa --data_path ./data/VisA_pytorch --round 0 --image_size 448 --k_shot 1;
python test_univad.py --dataset mvtec_loco --data_path ./data/mvtec_loco_caption --image_size 448 --k_shot 1;
python test_univad.py --dataset brainmri --data_path ./data/brainmri --round 0 --image_size 448 --k_shot 1;
python test_univad.py --dataset liverct --data_path ./data/liverct --round 0 --image_size 448 --k_shot 1; 
python test_univad.py --dataset resc --data_path ./data/resc --round 0 --image_size 448 --k_shot 1; 
python test_univad_no_pixel.py --dataset chestxray --data_path ./data/chestxray --round 0 --image_size 448 --k_shot 1; 
python test_univad_no_pixel.py --dataset his --data_path ./data/his --round 0 --image_size 448 --k_shot 1;
python test_univad_no_pixel.py --dataset oct17 --data_path ./data/oct17 --round 0 --image_size 448 --k_shot 1;
