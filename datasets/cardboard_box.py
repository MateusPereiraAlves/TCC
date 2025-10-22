import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import os
import glob

class CardboardBoxDataset(data.Dataset):
    """
    Dataloader para o dataset de caixas de papelão.
    Lê a estrutura de pastas:
    - /cardboard_box/test/good/ (anomalia = 0)
    - /cardboard_box/test/bad/  (anomalia = 1)
    """
    def __init__(self, root, transform, target_transform, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        
        # Define o nome da classe permanentemente
        self.cls_names = ['cardboard_box']
        self.data_all = []
        self.anomaly_labels = []

        # Este Dataloader só é usado para carregar dados de 'test'
        if self.mode == 'test':
            for cls_name in self.cls_names:
                cls_dir = os.path.join(self.root, cls_name)
                test_dir = os.path.join(cls_dir, 'test')
                
                # Carregar imagens 'good' (anomalia = 0)
                good_dir = os.path.join(test_dir, 'good')
                good_paths = self._find_image_files(good_dir)
                self.data_all.extend(good_paths)
                self.anomaly_labels.extend([0] * len(good_paths))
                
                # Carregar imagens 'bad' (anomalia = 1)
                bad_dir = os.path.join(test_dir, 'bad')
                bad_paths = self._find_image_files(bad_dir)
                self.data_all.extend(bad_paths)
                self.anomaly_labels.extend([1] * len(bad_paths))
        
        self.length = len(self.data_all)

    def _find_image_files(self, dir_path):
        # Procura por extensões comuns de imagem
        patterns = [
            os.path.join(dir_path, '*.png'),
            os.path.join(dir_path, '*.jpg'),
            os.path.join(dir_path, '*.jpeg'),
            os.path.join(dir_path, '*.bmp'),
        ]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))
        return sorted(files)

    def __len__(self):
        return self.length

    def get_cls_names(self):
        # Retorna a lista de classes para o script principal
        return self.cls_names

    def __getitem__(self, index):
        img_path = self.data_all[index]
        anomaly = self.anomaly_labels[index]
        cls_name = 'cardboard box' # Nome formatado
        
        img_pil = Image.open(img_path).convert('RGB')
        
        # --- AQUI ESTÁ A MUDANÇA ---
        # Como não temos máscaras de ground truth, criamos uma máscara FALSA (vazia/preta)
        # O script principal precisa que 'items["img_mask"]' exista
        img_mask = Image.fromarray(np.zeros((img_pil.size[1], img_pil.size[0])), mode='L')
        
        # Aplicar transformações
        img = self.transform(img_pil) if self.transform is not None else img_pil
        img_mask = self.target_transform(img_mask) if self.target_transform is not None else img_mask

        # --- INÍCIO DA CORREÇÃO ---
        
        # O 'img' é um tensor (C, H, W) que já foi redimensionado 
        # corretamente pelo self.transform (para 224x224, no seu caso).
        # Vamos pegar o H e W dele:
        _, H, W = img.shape
        
        # Redimensionamos a imagem PIL original para esse mesmo H e W
        # antes de convertê-la para NumPy.
        img_pil_resized = img_pil.resize((W, H))

        return {
            'img_pil': np.array(img_pil_resized), # <--- CORRIGIDO
            'img': img, 
            'img_mask': img_mask, # A máscara falsa que criamos
            'cls_name': cls_name, 
            'anomaly': anomaly, # 0 (good) ou 1 (bad)
            'anomaly_class': 'bad' if anomaly == 1 else 'good',
            'img_path': img_path
        }
        # --- FIM DA CORREÇÃO ---