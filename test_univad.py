import argparse
import logging
import os
import numpy as np
import torch
import torchvision
import threading
import torchvision.transforms as transforms
from tabulate import tabulate
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import math
from PIL import Image
# from prefetch_generator import BackgroundGenerator # Removido
import cv2
import yaml
import glob
import shutil # Importado para o workaround

# --- Importações (sem alterações) ---
from UniVAD import UniVAD
from models.component_segmentaion import grounding_segmentation
from datasets.mvtec import MVTecDataset
from datasets.visa import VisaDataset
# ... (outras importações de dataset) ...
from datasets.cardboard_box import CardboardBoxDataset

class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        # return BackgroundGenerator(super().__iter__()) # Removido
        return super().__iter__()

# --- Funções auxiliares (resize_tokens, read_config - sem alterações) ---
def resize_tokens(x): B, N, C = x.shape; x = x.view(B, int(math.sqrt(N)), int(math.sqrt(N)), C); return x
def read_config(config_path):
    with open(config_path, "r") as f: config = yaml.load(f, Loader=yaml.SafeLoader); return config

# --- Função cal_score (sem alterações da versão anterior com robustez) ---
results = {}
table_ls = []; auroc_sp_ls = []; auroc_px_ls = []
# ... (código da função cal_score permanece o mesmo) ...
def cal_score(obj):
    global table_ls, auroc_sp_ls, auroc_px_ls
    table = []; gt_px_list = []; pr_px_list = []; gt_sp_list = []; pr_sp_list = []
    table.append(obj)
    found_obj = False
    for idxes in range(len(results.get("cls_names", []))):
        if results["cls_names"][idxes] == obj:
            found_obj = True; mask_data = results["imgs_masks"][idxes]
            if isinstance(mask_data, torch.Tensor): mask_np = mask_data.squeeze().cpu().numpy()
            elif isinstance(mask_data, np.ndarray): mask_np = np.squeeze(mask_data)
            else: print(f"WARN cal_score: Tipo inesperado gt_mask {type(mask_data)}"); mask_np = np.array([])
            if mask_np.ndim == 0: mask_np = mask_np[np.newaxis, np.newaxis]
            elif mask_np.ndim == 1: mask_np = mask_np[:, np.newaxis]
            gt_px_list.append(mask_np if mask_np.size > 0 else np.array([[]]))
            pr_px_data = results["anomaly_maps"][idxes]
            if isinstance(pr_px_data, np.ndarray): pr_px_list.append(np.squeeze(pr_px_data))
            else: print(f"WARN cal_score: Tipo inesperado pr_mask {type(pr_px_data)}"); pr_px_list.append(np.array([[]]))
            gt_sp_list.append(results["gt_sp"][idxes]); pr_sp_list.append(results["pr_sp"][idxes])
    if not found_obj: print(f"WARN cal_score: Objeto '{obj}' não encontrado."); return
    gt_sp_np = np.array(gt_sp_list); pr_sp_np = np.array(pr_sp_list)
    gt_px_valid = [arr for arr in gt_px_list if arr.ndim == 2 and arr.size > 0]
    pr_px_valid = [arr for arr in pr_px_list if arr.ndim == 2 and arr.size > 0]
    gt_px_flat = np.array([]); pr_px_flat = np.array([]); can_calc_px = False
    if gt_px_valid and pr_px_valid and len(gt_px_valid) == len(pr_px_valid):
        try:
            gt_px_stack = np.stack(gt_px_valid); pr_px_stack = np.stack(pr_px_valid)
            gt_px_flat = gt_px_stack.ravel(); pr_px_flat = pr_px_stack.ravel(); can_calc_px = True
        except ValueError:
            try:
                # print(f"WARN cal_score: Shapes diferentes {obj}, tentando concatenar.") # Log opcional
                gt_px_concat = np.concatenate([a.ravel() for a in gt_px_valid])
                pr_px_concat = np.concatenate([a.ravel() for a in pr_px_valid])
                min_len_flat = min(len(gt_px_concat), len(pr_px_concat))
                if min_len_flat > 0: gt_px_flat = gt_px_concat[:min_len_flat]; pr_px_flat = pr_px_concat[:min_len_flat]; can_calc_px = True
                # else: print(f"WARN cal_score: Arrays achatados vazios {obj}.") # Log opcional
            except Exception as e_concat: print(f"ERRO cal_score: Falha ao processar máscaras/mapas {obj}: {e_concat}")
    # elif gt_px_valid or pr_px_valid: print(f"WARN cal_score: Inconsistência no número de máscaras/mapas válidos para {obj}.") # Log opcional
    auroc_sp = np.nan
    try:
        if len(np.unique(gt_sp_np)) > 1: auroc_sp = roc_auc_score(gt_sp_np, pr_sp_np)
        # elif len(gt_sp_np) > 0: print(f"WARN cal_score: Apenas uma classe GT SP {obj}.") # Log opcional
    except Exception as e: print(f"ERRO calculando AUROC_SP para {obj}: {e}")
    auroc_px = np.nan; auroc_px_str = "N/A"
    if can_calc_px and gt_px_flat.size > 0:
        if np.max(gt_px_flat) > 0 and len(np.unique(gt_px_flat)) > 1:
            try:
                auroc_px = roc_auc_score(gt_px_flat, pr_px_flat)
                if not np.isnan(auroc_px): auroc_px_str = str(np.round(auroc_px * 100, decimals=1))
            except ValueError as e: print(f"ERRO calculando AUROC_PX para {obj}: {e}.")
            except Exception as e: print(f"ERRO inesperado calculando AUROC_PX para {obj}: {e}")
        else: auroc_px_str = "N/A (No GT Px Var)"
    # elif can_calc_px and gt_px_flat.size == 0: print(f"WARN cal_score: Arrays GT PX achatados vazios {obj}.") # Log opcional
    table.append(str(np.round(auroc_sp * 100, decimals=1)) if not np.isnan(auroc_sp) else "N/A")
    table.append(auroc_px_str)
    table_ls.append(table); auroc_sp_ls.append(auroc_sp); auroc_px_ls.append(auroc_px)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Test Sequencial UniVAD", add_help=True)
    # --- Argumentos (sem alterações) ---
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--k_shot", type=int, default=1, help="k-shot")
    parser.add_argument("--dataset", type=str, default="cardboard_box", help="train dataset name")
    parser.add_argument("--data_path",type=str, default="./data/CardboardBox", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default=f"./results/", help="path to save results")
    parser.add_argument("--round", type=int, default=0, help="round")
    parser.add_argument("--class_name", type=str, default="None", help="Filtra classe")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--config_path_base", type=str, default="./configs/class_histogram", help="Caminho base .yaml")
    parser.add_argument("--anomaly_threshold", type=float, default=0.5, help="Threshold para classificar scores > threshold como 'Bad'")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade do shuffle")
    args = parser.parse_args()

    # --- Configurações Iniciais (sem alterações) ---
    dataset_name = args.dataset; dataset_dir = args.data_path; device = args.device
    k_shot = args.k_shot; image_size = args.image_size; anomaly_threshold = args.anomaly_threshold
    save_path = os.path.join(args.save_path, f"{dataset_name}_sequential_classified")
    if not os.path.exists(save_path): os.makedirs(save_path)
    txt_path = os.path.join(save_path, "log_sequential_classified.txt")

    # --- Configuração do Logger (sem alterações) ---
    # ... (código do logger) ...
    root_logger = logging.getLogger(); [root_logger.removeHandler(h) for h in root_logger.handlers[:]]; root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger("test_classified"); formatter = logging.Formatter("%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s", datefmt="%y-%m-%d %H:%M:%S"); logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode="w"); file_handler.setFormatter(formatter); logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(); console_handler.setFormatter(formatter); logger.addHandler(console_handler)

    for arg in vars(args): logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info(f"Usando threshold de anomalia: {anomaly_threshold}")

    # --- Seed para Reprodutibilidade ---
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if device == 'cuda': torch.cuda.manual_seed_all(args.seed)
    logger.info(f"Seed definida: {args.seed}")

    # --- Carregamento de Modelos (sem alterações) ---
    logger.info("Carregando modelo UniVAD...")
    UniVAD_model = UniVAD(image_size=args.image_size).to(device)
    logger.info("Modelo UniVAD carregado.")

    # --- Configuração do Dataset (sem alterações) ---
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    target_transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    
    # --- Seleção de Dataset (sem alterações) ---
    # ... (código if/elif para carregar test_data) ...
    if dataset_name == "mvtec": test_data = MVTecDataset( root=dataset_dir, transform=transform, target_transform=target_transform, mode="test")
    elif dataset_name == "visa": test_data = VisaDataset( root=dataset_dir, transform=transform, target_transform=target_transform, mode="test")
    # ... (outros elifs) ...
    elif dataset_name == "cardboard_box": test_data = CardboardBoxDataset( root=dataset_dir, transform=transform, target_transform=target_transform, mode="test")
    else: raise NotImplementedError("Dataset not supported")

    # --- DataLoader (shuffle=True) ---
    use_shuffle = True
    logger.info(f"DataLoader shuffle: {use_shuffle}")
    test_dataloader = DataLoaderX(
        test_data, batch_size=1, shuffle=use_shuffle,
        num_workers=min(os.cpu_count(), 4), pin_memory=False, # <-- pin_memory=False pode ajudar RAM
        generator=torch.Generator().manual_seed(args.seed) if use_shuffle else None 
    )

    try: obj_list = test_data.get_cls_names()
    except Exception as e: logger.error(f"Erro ao obter nomes de classe: {e}"); obj_list = []

    # --- Variáveis de Resultados e Contadores ---
    results = {"cls_names": [], "imgs_masks": [], "anomaly_maps": [], "gt_sp": [], "pr_sp": []}
    cls_last = None
    grounding_config = None

    image_transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    total_good = 0; total_bad = 0; true_positives = 0; true_negatives = 0
    false_positives = 0; false_negatives = 0
    # --- FIM ADIÇÃO ---

    logger.info("Iniciando o loop de inferência sequencial...")
    
    # --- LOOP DE PROCESSAMENTO PRINCIPAL ---
    for i, items in enumerate(tqdm(test_dataloader, desc="Processando Amostras")):
        
        image_path = items["img_path"][0] 
        if items["anomaly"][0] == -1: logger.warning(f"Item inválido {image_path}. Pulando."); continue

        # --- Variáveis para guardar caminhos de máscara para o loop atual ---
        expected_mask_path_jpg_ref = None # Caminho .jpg esperado pelo setup
        actual_mask_path_png_ref = None # Caminho .png real da ref
        expected_mask_path_jpg_test = None # Caminho .jpg esperado pelo forward
        actual_mask_path_png_test = None # Caminho .png real do teste

        try: 
            # --- 1. Extração de Dados ---
            # ... (código igual anterior) ...
            image = items["img"].to(device)
            if image.nelement() == 0: logger.warning(f"Imagem vazia {image_path}. Pulando."); continue
            image_pil = items["img_pil"]; cls_name = items["cls_name"][0]
            if args.class_name != "None":
                 filter_name_norm = args.class_name.lower().replace("_", " ").strip()
                 current_cls_norm = cls_name.lower().strip()
                 if filter_name_norm != current_cls_norm: continue
            gt_mask_tensor = items["img_mask"]
            if not isinstance(gt_mask_tensor, torch.Tensor) or gt_mask_tensor.nelement() == 0: gt_mask_tensor = torch.zeros((1, image_size, image_size), dtype=torch.float32)
            else:
                 if gt_mask_tensor.dim() == 2: gt_mask_tensor = gt_mask_tensor.unsqueeze(0)
                 if gt_mask_tensor.shape[0] != 1: gt_mask_tensor = gt_mask_tensor[0:1, :, :]
                 gt_mask_tensor = gt_mask_tensor.float()
            gt_mask_processed = torch.where(gt_mask_tensor > 0.5, torch.tensor(1.0), torch.tensor(0.0))
            true_anomaly_label = items["anomaly"].item()

            results["cls_names"].append(cls_name)
            results["imgs_masks"].append(gt_mask_processed.cpu())
            results["gt_sp"].append(true_anomaly_label)

            # --- 2. Setup de Nova Classe (COM WORKAROUND PERSISTENTE) ---
            if cls_name != cls_last:
                logger.info(f"\nConfigurando para a nova classe: {cls_name}")
                
                # --- Lógica k-shot (igual anterior) ---
                normal_image_paths = [] 
                # ... (código if/elif para popular normal_image_paths para visa, cardboard_box, etc.) ...
                if dataset_name == "visa":
                    cls_name_fs = cls_name.replace(" ", "_")
                    if cls_name_fs in ["capsules", "cashew", "chewinggum", "fryum", "pipe_fryum"]: zfill_num, ext = 3, "JPG"
                    else: zfill_num, ext = 4, "JPG"
                    ref_dir = os.path.join(dataset_dir, cls_name_fs, "train/good")
                    try:
                         all_files = sorted([f for f in os.listdir(ref_dir) if f.lower().endswith(ext.lower())])
                         selected_files = all_files[args.round : args.round + k_shot]
                         normal_image_paths = [os.path.join(ref_dir, f) for f in selected_files]
                         if len(normal_image_paths) < k_shot: logger.warning(f"Menos que {k_shot} refs encontradas para {cls_name}")
                    except FileNotFoundError: logger.error(f"Dir de ref não encontrado: {ref_dir}"); normal_image_paths = []
                elif dataset_name == "cardboard_box":
                    dir_path = os.path.join(dataset_dir, cls_name.replace(" ", "_"), "train/good")
                    patterns = [os.path.join(dir_path, f'*.{ext}') for ext in ['png', 'jpg', 'jpeg', 'bmp', 'JPG', 'JPEG']]
                    all_files = []; [all_files.extend(glob.glob(p)) for p in patterns]
                    files_sorted = sorted(all_files)
                    normal_image_paths = files_sorted[args.round : args.round + k_shot]
                    if not normal_image_paths: logger.warning(f"Nenhuma ref encontrada em {dir_path}")
                else: # Fallback genérico
                     ref_dir = os.path.join(dataset_dir, cls_name.replace(" ", "_"), "train/good")
                     logger.warning(f"Usando k-shot genérico para {cls_name} em {ref_dir}")
                     patterns = [ os.path.join(ref_dir, f'*.{ext}') for ext in ['png', 'jpg', 'jpeg', 'bmp', 'JPG', 'JPEG'] ]
                     all_files = []; [all_files.extend(glob.glob(p)) for p in patterns]
                     files_sorted = sorted(all_files)
                     normal_image_paths = files_sorted[args.round : args.round + k_shot]
                     if not normal_image_paths: raise FileNotFoundError(f"Refs k-shot não encontradas em {ref_dir}")

                # --- Carregar Config ---
                config_path = os.path.join(args.config_path_base, f"{cls_name.replace(' ', '_')}.yaml")
                # ... (lógica para carregar grounding_config) ...
                if not os.path.exists(config_path): logger.warning(f"Config não encontrado: {config_path}"); grounding_config = None
                else:
                    config = read_config(config_path); grounding_config = config.get("grounding_config")
                    if grounding_config: logger.info(f"Config '{cls_name}' carregada.")
                    else: logger.warning(f"'grounding_config' não encontrado em {config_path}")

                # --- Gerar e Renomear Máscaras de Referência (WORKAROUND) ---
                if grounding_config and normal_image_paths:
                    dataset_base_path = os.path.relpath(dataset_dir, "./data")
                    reference_mask_path_base = os.path.join("./masks", dataset_base_path, cls_name.replace(' ', '_'))
                    logger.info(f"Gerando máscaras de ref e renomeando para .jpg em base: {reference_mask_path_base}")
                    
                    generated_mask_info_current_cls = [] # Reset para a classe atual
                    try:
                        # Roda segmentação (salva .png)
                        grounding_segmentation(normal_image_paths, reference_mask_path_base, grounding_config)

                        # Calcula caminhos e renomeia
                        for ref_img_path in normal_image_paths:
                            img_name_parts_gs = ('/'.join((ref_img_path.split(".")[-2]).split("/")[-3:]))
                            actual_mask_path_png_ref = os.path.join(reference_mask_path_base, img_name_parts_gs, "grounding_mask.png")
                            
                            # Calcula o path .jpg esperado pelo UniVAD
                            try:
                                 relative_img_path_for_setup = os.path.relpath(ref_img_path, './data') 
                                 expected_mask_path_jpg_ref = os.path.join("./masks", relative_img_path_for_setup)
                            except ValueError: 
                                 split_part_setup = ref_img_path.split('/data/')[-1]
                                 expected_mask_path_jpg_ref = os.path.join("./masks", split_part_setup)

                            # Renomeia .png para .jpg
                            if os.path.exists(actual_mask_path_png_ref):
                                try:
                                    os.makedirs(os.path.dirname(expected_mask_path_jpg_ref), exist_ok=True)
                                    # Apaga o .jpg antigo se existir (para evitar erro no move)
                                    if os.path.exists(expected_mask_path_jpg_ref):
                                        os.remove(expected_mask_path_jpg_ref)
                                    shutil.move(actual_mask_path_png_ref, expected_mask_path_jpg_ref)
                                    # logger.info(f"  Movido ref: {os.path.basename(actual_mask_path_png_ref)} -> {expected_mask_path_jpg_ref}")
                                except Exception as e_rename:
                                    logger.error(f"Falha ao renomear ref {actual_mask_path_png_ref} -> {expected_mask_path_jpg_ref}: {e_rename}")
                                    raise RuntimeError("Falha ao renomear máscaras ref.") from e_rename
                            else:
                                # Se a máscara .png não foi criada, verifica se o .jpg já existe (de run anterior)
                                if not os.path.exists(expected_mask_path_jpg_ref):
                                     logger.warning(f"Máscara ref {actual_mask_path_png_ref} não encontrada E {expected_mask_path_jpg_ref} não existe. Setup falhará.")
                                     raise FileNotFoundError(f"Máscara de referência essencial não encontrada: {actual_mask_path_png_ref}")
                                # else: logger.info(f"Usando máscara ref .jpg pré-existente: {expected_mask_path_jpg_ref}") # Log opcional

                    except Exception as e_gs: 
                        logger.error(f"Erro durante geração/renomeação de máscaras ref: {e_gs}")
                        raise e_gs
                else:
                    logger.warning("Pulando geração/renomeação de máscara de referência (sem config ou sem refs)")
                
                # --- Setup UniVAD ---
                if not normal_image_paths: raise ValueError(f"Imagens de ref não carregadas para {cls_name}")
                try:
                    normal_images = torch.cat([ image_transform(Image.open(x).convert("RGB")).unsqueeze(0) for x in normal_image_paths ], dim=0).to(device)
                except Exception as e: logger.error(f"Erro ao carregar/concatenar imgs ref: {e}"); raise e
                setup_data = {"few_shot_samples": normal_images, "dataset_category": cls_name.replace(" ", "_"), "image_path": normal_image_paths}
                
                UniVAD_model.setup(setup_data) # AGORA deve encontrar os .jpg
                logger.info(f"Setup UniVAD para {cls_name} concluído.")
                
                # NÃO renomeia de volta para .png aqui
                cls_last = cls_name 
                del normal_images 
                if 'cuda' in device: torch.cuda.empty_cache() 

            # --- 3. Processamento Sequencial por Amostra ---
            
            # --- PASSO A: Grounding Segmentation (Teste - COM RENAME) ---
            if grounding_config:
                dataset_base_path = os.path.relpath(dataset_dir, "./data")
                mask_output_dir_base = os.path.join("./masks", dataset_base_path, cls_name.replace(' ', '_'))
                
                # Roda segmentação (salva .png)
                grounding_segmentation([image_path], mask_output_dir_base, grounding_config)

                # Calcula os caminhos .png e .jpg para a imagem de TESTE
                img_name_parts_gs_test = ('/'.join((image_path.split(".")[-2]).split("/")[-3:]))
                actual_mask_path_png_test = os.path.join(mask_output_dir_base, img_name_parts_gs_test, "grounding_mask.png")
                try:
                     relative_img_path_test = os.path.relpath(image_path, './data')
                     expected_mask_path_jpg_test = os.path.join("./masks", relative_img_path_test)
                except ValueError:
                     split_part_test = image_path.split('/data/')[-1]
                     expected_mask_path_jpg_test = os.path.join("./masks", split_part_test)

                # Renomeia .png para .jpg
                if os.path.exists(actual_mask_path_png_test):
                    try:
                        os.makedirs(os.path.dirname(expected_mask_path_jpg_test), exist_ok=True)
                        if os.path.exists(expected_mask_path_jpg_test):
                            os.remove(expected_mask_path_jpg_test)
                        shutil.move(actual_mask_path_png_test, expected_mask_path_jpg_test)
                        # logger.info(f"  Movido teste: {os.path.basename(actual_mask_path_png_test)} -> {expected_mask_path_jpg_test}") # Log opcional
                    except Exception as e_rename_test:
                        logger.error(f"Falha ao renomear teste {actual_mask_path_png_test} -> {expected_mask_path_jpg_test}: {e_rename_test}")
                        # Decide se continua ou para
                        continue # Pula esta imagem se não conseguiu renomear a máscara
                else:
                    # Se .png não existe, verifica se .jpg já existe
                    if not os.path.exists(expected_mask_path_jpg_test):
                         logger.warning(f"Máscara teste {actual_mask_path_png_test} não encontrada E {expected_mask_path_jpg_test} não existe. Inferência falhará.")
                         # Decide se continua ou para
                         continue # Pula esta imagem
                    # else: logger.info(f"Usando máscara teste .jpg pré-existente: {expected_mask_path_jpg_test}") # Log opcional
            
            # --- PASSO B: Inferência UniVAD ---
            anomaly_map = None 
            with torch.no_grad():
                # A inferência AGORA deve encontrar a máscara .jpg
                pred_value = UniVAD_model(image, image_path, image_pil) 
                anomaly_score, anomaly_map_gpu = (pred_value["pred_score"], pred_value["pred_mask"])
                anomaly_map = anomaly_map_gpu.detach().cpu().numpy()
                results["anomaly_maps"].append(anomaly_map)
                overall_anomaly_score = anomaly_score.item()
                results["pr_sp"].append(overall_anomaly_score)

            # --- PASSO C: Salvar Heatmap (sem alterações) ---
            # ... (código para salvar overlay) ...
            output_dir_heatmap = os.path.join(save_path, "overlaid_results", cls_name.replace(" ", "_"))
            os.makedirs(output_dir_heatmap, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"{base_name}_overlay.png"
            output_filepath = os.path.join(output_dir_heatmap, output_filename)
            try:
                # --- INÍCIO DA CORREÇÃO ---
                # image_pil é um Tensor [1, H, W, C] vindo do dataloader.
                # Pegamos o primeiro item [0], movemos para a CPU e convertemos para NumPy.
                # O resize é desnecessário, pois seu Dataloader já redimensionou a 'img_pil'.
                original_img_rgb_np = image_pil[0].cpu().numpy()
                # --- FIM DA CORREÇÃO ---

                if original_img_rgb_np.size > 0 and original_img_rgb_np.ndim == 3:
                     original_img_bgr = cv2.cvtColor(original_img_rgb_np, cv2.COLOR_RGB2BGR)
                     heatmap_np = np.squeeze(anomaly_map) 
                     if heatmap_np.size > 0:
                          heatmap_normalized = (heatmap_np - np.min(heatmap_np)) / (np.max(heatmap_np) - np.min(heatmap_np) + 1e-8)
                          heatmap_uint8 = (heatmap_normalized * 255).astype(np.uint8)
                          heatmap_resized = cv2.resize(heatmap_uint8, (original_img_bgr.shape[1], original_img_bgr.shape[0]))
                          heatmap_colored_bgr = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                          overlayed_img_bgr = cv2.addWeighted(original_img_bgr, 0.6, heatmap_colored_bgr, 0.4, 0)
                          overlayed_img_rgb = cv2.cvtColor(overlayed_img_bgr, cv2.COLOR_BGR2RGB)
                          Image.fromarray(overlayed_img_rgb).save(output_filepath)
            except Exception as e_overlay: logger.error(f"Erro ao salvar overlay para {image_path}: {e_overlay}")

            # --- Classificação e Status Detalhado (sem alterações) ---
            # ... (código para calcular acurácia e logar) ...
            if true_anomaly_label == 0: total_good += 1
            else: total_bad += 1
            predicted_label_int = 1 if overall_anomaly_score > anomaly_threshold else 0
            predicted_label_str = "BAD" if predicted_label_int == 1 else "GOOD"
            true_label_str = "BAD" if true_anomaly_label == 1 else "GOOD"
            is_correct = False
            if predicted_label_int == 1 and true_anomaly_label == 1: true_positives += 1; is_correct = True
            elif predicted_label_int == 0 and true_anomaly_label == 0: true_negatives += 1; is_correct = True
            elif predicted_label_int == 1 and true_anomaly_label == 0: false_positives += 1
            elif predicted_label_int == 0 and true_anomaly_label == 1: false_negatives += 1
            total_processed = total_good + total_bad
            total_correct = true_positives + true_negatives
            overall_accuracy = (total_correct / total_processed) * 100 if total_processed > 0 else 0
            accuracy_good = (true_negatives / total_good) * 100 if total_good > 0 else 0
            accuracy_bad = (true_positives / total_bad) * 100 if total_bad > 0 else 0
            status_log = (f"Img [{i+1}/{len(test_dataloader)}]: {os.path.basename(image_path)} | Score: {overall_anomaly_score:.4f} | Prev: {predicted_label_str} | Real: {true_label_str} | {'CORRETO' if is_correct else 'INCORRETO'} | Acc Total: {overall_accuracy:.2f}% ({total_correct}/{total_processed}) | Acc Good: {accuracy_good:.2f}% | Acc Bad: {accuracy_bad:.2f}%")
            logger.info(status_log)

            # --- Liberação de Memória do Loop ---
            del image, image_pil, gt_mask_tensor, gt_mask_processed, pred_value, anomaly_score, anomaly_map_gpu, anomaly_map
            if 'cuda' in device: torch.cuda.empty_cache()

        # --- Tratamento de Erros (sem alterações) ---
        except FileNotFoundError as e: logger.error(f"\n--- ERRO ARQUIVO ---"); logger.error(f"Arq: {image_path}"); logger.error(f"Erro: {e}"); logger.error(f"Verifique paths/máscaras."); logger.error(f"------------------\n"); continue
        except Exception as e:
            logger.error(f"\n--- ERRO AMOSTRA ---"); logger.error(f"Arq: {image_path}"); logger.error(f"Erro: {e}")
            import traceback; logger.error(traceback.format_exc()); logger.error(f"Pulando."); logger.error(f"------------------\n")
            continue 

    # --- 4. Relatório Final de Métricas (AUROC - sem alterações) ---
    # ... (código para calcular AUROC com cal_score e imprimir tabela) ...
    logger.info("Cálculo final de métricas AUROC...")
    table_ls = []; auroc_sp_ls = []; auroc_px_ls = [] 
    if not obj_list: obj_list = sorted(list(set(results.get("cls_names", []))))
    if not obj_list: logger.warning("Nenhuma classe processada para AUROC.")
    for obj in tqdm(obj_list, desc="Calculando Métricas AUROC"):
        try: cal_score(obj)
        except Exception as e: logger.error(f"Erro métricas AUROC para {obj}: {e}")
    if table_ls: 
        mean_auroc_sp = np.nanmean([s for s in auroc_sp_ls if not np.isnan(s)])
        mean_auroc_px = np.nanmean([s for s in auroc_px_ls if not np.isnan(s)])
        mean_auroc_sp_str = str(np.round(mean_auroc_sp * 100, decimals=1)) if not np.isnan(mean_auroc_sp) else "N/A"
        mean_auroc_px_str = str(np.round(mean_auroc_px * 100, decimals=1)) if not np.isnan(mean_auroc_px) else "N/A"
        if mean_auroc_sp_str != "N/A" or mean_auroc_px_str != "N/A":
             table_ls.append( ["mean", mean_auroc_sp_str, mean_auroc_px_str] )
        results_table = tabulate(table_ls, headers=["objects", "auroc_sp", "auroc_px"], tablefmt="pipe")
        logger.info("\n%s", results_table)
    else: logger.warning("Nenhuma métrica AUROC calculada.")
    
    # --- Relatório Final Matriz de Confusão (sem alterações) ---
    # ... (código para imprimir matriz e métricas de acurácia) ...
    final_accuracy = ( (true_positives + true_negatives) / (total_good + total_bad) ) * 100 if (total_good + total_bad) > 0 else 0
    accuracy_good = (true_negatives / total_good) * 100 if total_good > 0 else 0
    accuracy_bad = (true_positives / total_bad) * 100 if total_bad > 0 else 0 
    precision_bad = (true_positives / (true_positives + false_positives)) * 100 if (true_positives + false_positives) > 0 else 0
    logger.info("\n--- Resumo Final da Classificação (baseado no threshold) ---")
    logger.info(f"Threshold de Anomalia: {anomaly_threshold}")
    logger.info(f"Total Imagens: {total_good + total_bad} (Good: {total_good}, Bad: {total_bad})")
    logger.info("--- Matriz de Confusão ---")
    conf_matrix = [["Real \\ Previsto", "Prev: GOOD", "Prev: BAD"], ["Real: GOOD", true_negatives, false_positives], ["Real: BAD", false_negatives, true_positives]]
    logger.info("\n" + tabulate(conf_matrix, headers="firstrow", tablefmt="grid"))
    logger.info("--- Métricas ---")
    metrics_summary = [["Acurácia Geral", f"{final_accuracy:.2f}%"], ["Acurácia Classe GOOD (Especificidade)", f"{accuracy_good:.2f}%"], ["Acurácia Classe BAD (Recall/Sensibilidade)", f"{accuracy_bad:.2f}%"], ["Precisão Classe BAD", f"{precision_bad:.2f}%"]]
    logger.info("\n" + tabulate(metrics_summary))


    logger.info(f"Processamento sequencial concluído. Resultados salvos em: {save_path}")