# Salve este arquivo como: run_single_inference.py

import argparse
import logging
import os
import time
import numpy as np
import torch
import torchvision.transforms as transforms
import math
from PIL import Image
import cv2
import yaml
import glob
import shutil
import random

# --- Importações do Projeto ---
# (Assumindo que estão no sys.path quando executado)
from UniVAD import UniVAD
from models.component_segmentaion import SegmentationHandler

# --- Funções auxiliares (copiadas do test_univad.py) ---
def resize_tokens(x): B, N, C = x.shape; x = x.view(B, int(math.sqrt(N)), int(math.sqrt(N)), C); return x
def read_config(config_path):
    with open(config_path, "r") as f: config = yaml.load(f, Loader=yaml.SafeLoader); return config

def calculate_top_k_score(anomaly_map, top_k_percent, mask=None):
    if mask is not None and np.any(mask):
        pixels = anomaly_map[mask]
    else:
        pixels = anomaly_map.flatten()
    if pixels.size == 0: return 0.0 
    if top_k_percent <= 0.0: return np.max(pixels)
    k = max(1, int(len(pixels) * top_k_percent))
    top_k_values = np.sort(pixels)[-k:]
    return np.mean(top_k_values)

# --- 1. FUNÇÃO DE SETUP DE MODELOS ---
def setup_models(args, logger):
    """
    Carrega e inicializa os modelos UniVAD e de segmentação.
    (Lógica de 'test_univad.py', linhas 97-104)
    """
    logger.info("Carregando modelo UniVAD...")
    UniVAD_model = UniVAD(image_size=args.image_size).to(args.device)
    logger.info("Modelo UniVAD carregado.")
    
    logger.info("Carregando modelos de segmentação (GroundingDINO + SAM)...")
    segmentation_handler = SegmentationHandler(device=args.device)
    segmentation_handler.load_models()
    logger.info("Modelos de segmentação carregados.")
    
    return UniVAD_model, segmentation_handler

# --- 2. FUNÇÃO DE PROCESSAMENTO DE IMAGEM ÚNICA ---
def process_single_image(
    UniVAD_model, 
    segmentation_handler, 
    image_path_to_process, 
    reference_dir_path,  # Ex: .../train/good
    class_name, 
    dataset_base_path, # Ex: ./data/CardboardBox
    args, 
    logger
):
    """
    Processa uma única imagem para detecção de anomalias.
    (Lógica de 'test_univad.py', linhas 118-271, adaptada)
    """
    
    seg_time = 0.0
    pred_time = 0.0
    
    try:
        logger.info(f"Processando imagem: {image_path_to_process}")
        
        # --- Carregamento da imagem ---
        image_transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()])
        image_pil = Image.open(image_path_to_process).convert("RGB")
        image_tensor = image_transform(image_pil).unsqueeze(0).to(args.device)
        cls_name_on_disk = class_name.replace(" ", "_")
        class_config_path = os.path.join(args.config_path_base, f"{cls_name_on_disk}.yaml")

        # --- 1. Configuração do K-Shot (Setup da Classe) ---
        logger.info(f"\nConfigurando para a classe: {class_name}")
        
        normal_image_paths = sorted(glob.glob(os.path.join(reference_dir_path, '*.*')))
        if not normal_image_paths: 
            raise FileNotFoundError(f"Nenhuma imagem de referência (k-shot) encontrada em: {reference_dir_path}")

        grounding_config = None
        if os.path.exists(class_config_path):
            config = read_config(class_config_path); grounding_config = config.get("grounding_config")
            if grounding_config: logger.info(f"Config de segmentação para {class_name} carregada.")
        
        normal_image_paths_linux = []
        if grounding_config:
            rel_dataset_path = os.path.relpath(dataset_base_path, "./data")
            reference_mask_path = os.path.join("./masks", rel_dataset_path, cls_name_on_disk)
            
            normal_image_paths_linux = [p.replace(os.path.sep, '/') for p in normal_image_paths]
            
            logger.info(f"Gerando máscaras de referência (k-shot) em: {reference_mask_path}")
            segmentation_handler.segment(normal_image_paths_linux, reference_mask_path, grounding_config, find_only_one_object=args.find_only_one_object)
            
            logger.info("Aplicando workaround de renomeação nas máscaras de referência...")
            for ref_path in normal_image_paths:
                relative_path_part = os.path.relpath(ref_path, os.path.join(dataset_base_path, cls_name_on_disk))
                img_name_slug_dir = os.path.dirname(relative_path_part)
                img_name_slug_base = os.path.splitext(os.path.basename(relative_path_part))[0]
                
                actual_mask_dir = os.path.join(reference_mask_path, img_name_slug_dir, img_name_slug_base)
                actual_mask_file = os.path.join(actual_mask_dir, 'grounding_mask.png')
                
                relative_dest_path = os.path.relpath(ref_path, './data')
                expected_mask_path = os.path.join("./masks", relative_dest_path)
                
                if os.path.exists(actual_mask_file):
                    os.makedirs(os.path.dirname(expected_mask_path), exist_ok=True)
                    shutil.move(actual_mask_file, expected_mask_path)
                    if not os.listdir(actual_mask_dir): os.rmdir(actual_mask_dir)
                elif not os.path.exists(expected_mask_path):
                    raise FileNotFoundError(f"Máscara {actual_mask_file} não gerada E destino {expected_mask_path} não existe.")

        normal_images = torch.cat([image_transform(Image.open(x).convert("RGB")).unsqueeze(0) for x in normal_image_paths], dim=0).to(args.device)
        if not normal_image_paths_linux: 
             normal_image_paths_linux = [p.replace(os.path.sep, '/') for p in normal_image_paths]
             
        setup_data = {"few_shot_samples": normal_images, "dataset_category": cls_name_on_disk, "image_path": normal_image_paths_linux}
        UniVAD_model.setup(setup_data)
        logger.info(f"Setup do UniVAD para {class_name} concluído.")

        # --- 2. Segmentação da Imagem de Teste ---
        expected_mask_path_test = None
        if grounding_config:
            rel_dataset_path = os.path.relpath(dataset_base_path, "./data")
            mask_output_dir = os.path.join("./masks", rel_dataset_path, cls_name_on_disk)
            image_path_linux = image_path_to_process.replace(os.path.sep, '/')
            
            seg_start_time = time.time()
            segmentation_handler.segment([image_path_linux], mask_output_dir, grounding_config, find_only_one_object=args.find_only_one_object)
            seg_time = time.time() - seg_start_time
            
            relative_path_part_test = os.path.relpath(image_path_to_process, os.path.join(dataset_base_path, cls_name_on_disk))
            img_name_slug_dir_test = os.path.dirname(relative_path_part_test)
            img_name_slug_base_test = os.path.splitext(os.path.basename(relative_path_part_test))[0]
            
            actual_mask_dir_test = os.path.join(mask_output_dir, img_name_slug_dir_test, img_name_slug_base_test)
            actual_mask_file_test = os.path.join(actual_mask_dir_test, 'grounding_mask.png')
            
            relative_dest_path_test = os.path.relpath(image_path_to_process, './data')
            expected_mask_path_test = os.path.join("./masks", relative_dest_path_test)
            
            if os.path.exists(actual_mask_file_test):
                os.makedirs(os.path.dirname(expected_mask_path_test), exist_ok=True)
                shutil.move(actual_mask_file_test, expected_mask_path_test)
                if not os.listdir(actual_mask_dir_test): os.rmdir(actual_mask_dir)
            elif not os.path.exists(expected_mask_path_test):
                logger.warning(f"Máscara de teste {actual_mask_file_test} não gerada. A predição continuará sem máscara.")
                expected_mask_path_test = None

        # --- 3. Predição UniVAD ---
        with torch.no_grad():
            image_path_linux = image_path_to_process.replace(os.path.sep, '/')
            pred_start_time = time.time()
            pred_value = UniVAD_model(image_tensor, image_path_linux, image_pil, debug=args.debug)
            pred_time = time.time() - pred_start_time
            _, anomaly_map_tensor = (pred_value["pred_score"], pred_value["pred_mask"])
            anomaly_map = anomaly_map_tensor.squeeze().detach().cpu().numpy()
        
        # --- 4. Cálculo de Score ---
        seg_mask_np = None; score_log_info = ""
        if args.filter_with_mask and grounding_config and expected_mask_path_test and os.path.exists(expected_mask_path_test):
            try:
                seg_mask = Image.open(expected_mask_path_test).convert('L').resize((args.image_size, args.image_size), Image.NEAREST)
                seg_mask_np = np.array(seg_mask) > 0
                if not np.any(seg_mask_np): score_log_info = "(Máscara vazia) "
            except Exception as e:
                logger.warning(f"Não foi possível aplicar filtro de máscara: {e}")
        overall_anomaly_score = calculate_top_k_score(anomaly_map, args.top_k_percent, mask=seg_mask_np)

        # --- 5. Classificação ---
        predicted_label = 1 if overall_anomaly_score > args.anomaly_threshold else 0
        predicted_label_str = "BAD" if predicted_label == 1 else "GOOD"
        
        logger.info(f"Score: {overall_anomaly_score:.4f} {score_log_info} | Predição: {predicted_label_str}")
        logger.info(f"T_Seg: {seg_time:.2f}s | T_Pred: {pred_time:.2f}s")

        # --- 6. Salvar Heatmap ---
        output_dir = os.path.join(args.save_path, "single_inference_results", cls_name_on_disk)
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path_to_process))[0]
        output_filepath = os.path.join(output_dir, f"{base_name}_overlay_{predicted_label_str}_{overall_anomaly_score:.3f}.png")
        
        original_img_np = (image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        original_img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)
        heatmap_normalized = (anomaly_map - np.min(anomaly_map)) / (np.max(anomaly_map) - np.min(anomaly_map) + 1e-8)
        heatmap_uint8 = (heatmap_normalized * 255).astype(np.uint8)
        heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        overlayed_img_bgr = cv2.addWeighted(original_img_bgr, 0.6, heatmap_colored_bgr, 0.4, 0)
        
        if args.filter_with_mask and seg_mask_np is not None and np.any(seg_mask_np):
            final_display_img = np.where(seg_mask_np[..., None], overlayed_img_bgr, original_img_bgr)
        else: 
            final_display_img = overlayed_img_bgr
            
        Image.fromarray(cv2.cvtColor(final_display_img, cv2.COLOR_BGR2RGB)).save(output_filepath)
        logger.info(f"Heatmap salvo em: {output_filepath}")

        # Retorna os valores para o bloco __main__
        return overall_anomaly_score, predicted_label_str, output_filepath, seg_time, pred_time

    except Exception as e:
        logger.error(f"\n--- ERRO AO PROCESSAR IMAGEM: {image_path_to_process} ---")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("Retornando None.\n")
        return None, None, None, seg_time, pred_time

# --- 3. BLOCO DE EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    
    # --- Parser de argumentos ---
    parser = argparse.ArgumentParser("Executar Inferência Única UniVAD", add_help=True)
    
    # --- Argumentos da Função ---
    parser.add_argument("--image_path", type=str, required=True, help="Caminho para a imagem a ser processada.")
    parser.add_argument("--reference_dir", type=str, required=True, help="Caminho para o diretório de referência (ex: .../train/good).")
    parser.add_argument("--class_name", type=str, required=True, help="Nome da classe (ex: 'cardboard box').")
    # Path base do dataset, ex: ./data/CardboardBox (usado para estrutura de pastas)
    parser.add_argument("--data_path", type=str, required=True, help="Caminho base para o dataset (ex: ./data/CardboardBox).")
    
    # --- Argumentos de Configuração (copiados de test_univad.py) ---
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--save_path", type=str, default=f"./results_single_inference/", help="path to save results")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--config_path_base", type=str, default="./configs/class_histogram", help="Caminho base .yaml")
    parser.add_argument("--anomaly_threshold", type=float, default=0.5, help="Score threshold para classificar como 'Bad'")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--filter_with_mask", action='store_true', help="Filtra o score pela máscara")
    parser.add_argument("--debug", action='store_true', help="Ativa o modo debug")
    parser.add_argument("--top_k_percent", type=float, default=0.0, help="Top K% pixels para score (0.0 usa o max())")
    parser.add_argument("--find_only_one_object", action='store_true', help="Limita a segmentação a um objeto")
    
    args = parser.parse_args()

    # --- Configurações Iniciais ---
    os.makedirs(args.save_path, exist_ok=True)
    txt_path = os.path.join(args.save_path, "log_single_inference.txt")

    # --- Logger ---
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S", handlers=[logging.FileHandler(txt_path, mode='w'), logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    for arg in vars(args): logger.info(f"{arg}: {getattr(args, arg)}")
    
    # --- Seed ---
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    
    # --- Carregamento de Modelos ---
    UniVAD_model, segmentation_handler = setup_models(args, logger)
    
    # --- Verificação de Paths ---
    if not os.path.exists(args.image_path):
        logger.error(f"Imagem de entrada não encontrada: {args.image_path}"); exit()
    if not os.path.exists(args.reference_dir):
        logger.error(f"Diretório de referência não encontrado: {args.reference_dir}"); exit()
    if not os.path.exists(args.data_path):
        logger.error(f"Path base do dataset não encontrado: {args.data_path}"); exit()

    # --- Executar a Função de Inferência ---
    logger.info("\n" + "="*80 + "\nIniciando Inferência\n" + "="*80)
    
    start_total_time = time.time()
    
    score, label, heatmap_path, seg_time, pred_time = process_single_image(
        UniVAD_model=UniVAD_model,
        segmentation_handler=segmentation_handler,
        image_path_to_process=args.image_path,
        reference_dir_path=args.reference_dir,
        class_name=args.class_name,
        dataset_base_path=args.data_path, # Passa o novo argumento
        args=args,
        logger=logger
    )
    
    total_time = time.time() - start_total_time

    # --- Relatório Final (imprime no console) ---
    logger.info("\n" + "="*80 + "\nResultado da Inferência\n" + "="*80)
    if score is not None:
        print(f"  Imagem Processada: {args.image_path}")
        print(f"  Score de Anomalia: {score:.4f}")
        print(f"  Predição Final:    {label}")
        print(f"  Heatmap Salvo em:  {heatmap_path}")
        print(f"  Tempo de Segmentação: {seg_time:.2f}s")
        print(f"  Tempo de Predição:    {pred_time:.2f}s")
        print(f"  Tempo Total (Setup + Seg + Pred): {total_time:.2f}s")
    else:
        print("A inferência falhou. Verifique os logs de erro.")
        
    logger.info(f"\nProcessamento concluído. Resultados salvos em: {args.save_path}")