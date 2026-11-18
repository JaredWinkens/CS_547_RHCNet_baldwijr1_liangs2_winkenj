conda create -n underwater python=3.7.11 -y
conda activate underwater

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y

conda install numpy==1.21.2 -y

python -m pip install --upgrade pip

pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html

pip install mmdet==2.22.0

conda install --override-channel -c defaults mkl=2021.4.0 intel-openmp=2021.4.0 mkl-service=2.4.0 -y

pip uninstall pillow -y
pip install pillow==8.4.0

pip uninstall yapf -y
pip install yapf==0.31.0

pip install opencv-python

pip install scikit-learn==1.0.2
<!--


# This needs special handling for RTX A4000.
# Don't blindly use the prebuilt wheel yet.

The following is to see wht things you have
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

python -c "import numpy, cv2, PIL, sklearn, mmcv, mmdet; print('NumPy:', numpy.__version__); print('OpenCV:', cv2.__version__); print('Pillow:', PIL.__version__); print('scikit-learn:', sklearn.__version__); print('MMCV:', mmcv.__version__); print('MMDetection:', mmdet.__version__)"

# Verify the custom RHCNet HFCP neck is registered
python -c "import mmdet.models; from mmdet.models.builder import NECKS; print('HFCP registered:', 'HFCP' in NECKS.module_dict); print(NECKS.module_dict.get('HFCP'))"

# Verify the RHCNet model can actually be constructed
python -c "from mmcv import Config; from mmdet.models import build_detector; cfg=Config.fromfile('configs/rhcnet/rhcnet_tood_r50_fpn_anchor_based_2x_duo.py'); model=build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')); print('MODEL BUILT:', type(model).__name__)"

  -->



Extract the zip from google drive in https://github.com/chongweiliu/DUO , go in duo folder, there should be macosx and duo folder, move that duo folder to data folder

