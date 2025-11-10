import argparse
import logging
import os
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from tabulate import tabulate
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from tqdm import tqdm
import math
from PIL import Image
from prefetch_generator import BackgroundGenerator
import cv2
import yaml
import glob
import shutil
import random
from UniVAD import UniVAD
from models.component_segmentaion import SegmentationHandler
from datasets.cardboard_box import CardboardBoxDataset

class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# --- Funções auxiliares ---
def resize_tokens(x): B, N, C = x.shape; x = x.view(B, int(math.sqrt(N)), int(math.sqrt(N)), C); return x
def read_config(config_path):
    with open(config_path, "r") as f: config = yaml.load(f, Loader=yaml.SafeLoader); return config

# --- Função para cálculo do score Top- ---
def calculate_top_k_score(anomaly_map, top_k_percent, mask=None):
    if mask is not None and np.any(mask):
        pixels = anomaly_map[mask]
    else:
        pixels = anomaly_map.flatten()

    if pixels.size == 0:
        return 0.0 

    if top_k_percent <= 0.0:
        return np.max(pixels)
    
    k = max(1, int(len(pixels) * top_k_percent))
    top_k_values = np.sort(pixels)[-k:]
    return np.mean(top_k_values)

def cal_score(obj):
    table = []; gt_px, pr_px, gt_sp, pr_sp = [], [], [], []
    table.append(obj)
    for idxes in range(len(results["cls_names"])):
        if results["cls_names"][idxes] == obj:
            gt_px.append(results["imgs_masks"][idxes].squeeze(1).numpy())
            pr_px.append(results["anomaly_maps"][idxes])
            gt_sp.append(results["gt_sp"][idxes])
            pr_sp.append(results["pr_sp"][idxes])
    gt_px, gt_sp, pr_px, pr_sp = np.array(gt_px), np.array(gt_sp), np.array(pr_px), np.array(pr_sp)
    if len(np.unique(gt_sp)) > 1: auroc_sp = roc_auc_score(gt_sp, pr_sp)
    else: auroc_sp = np.nan
    if gt_px.max() > 0 and len(np.unique(gt_px)) > 1:
        auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
        auroc_px_str = str(np.round(auroc_px * 100, decimals=1))
    else: auroc_px, auroc_px_str = np.nan, "N/A"
    table.append(str(np.round(auroc_sp * 100, decimals=1)) if not np.isnan(auroc_sp) else "N/A")
    table.append(auroc_px_str)
    table_ls.append(table); auroc_sp_ls.append(auroc_sp); auroc_px_ls.append(auroc_px)

if __name__ == "__main__":
    # --- Parser de argumento ---
    parser = argparse.ArgumentParser("Test Sequencial UniVAD com Classificação", add_help=True)
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--k_shot", type=int, default=1, help="k-shot")
    parser.add_argument("--dataset", type=str, default="cardboard_box", help="train dataset name")
    parser.add_argument("--data_path", type=str, default="./data/CardboardBox", help="path to the specific dataset directory")
    parser.add_argument("--save_path", type=str, default=f"./results/", help="path to save results")
    parser.add_argument("--round", type=int, default=0, help="round")
    parser.add_argument("--class_name", type=str, default="None", help="class to filter")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--config_path_base", type=str, default="./configs/class_histogram", help="Caminho base .yaml")
    parser.add_argument("--anomaly_threshold", type=float, default=0.5, help="Score threshold para classificar como 'Bad'")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade do shuffle")
    parser.add_argument("--filter_with_mask", action='store_true', help="Filtra o score e o heatmap final pela máscara de segmentação")
    parser.add_argument("--debug", action='store_true', help="Ativa o modo debug para salvar mapas de calor e scores intermediários")
    parser.add_argument("--top_k_percent", type=float, default=0.0, help="Porcentagem (0.0 a 1.0) de pixels de maior anomalia para usar no score. 0.0 usa o max().")
    parser.add_argument("--find_only_one_object", action='store_true', help="Limita a segmentação a apenas um objeto (o de maior confiança).")
    args = parser.parse_args()

    # --- Configurações Iniciai ---
    dataset_name = args.dataset; dataset_dir = args.data_path; device = args.device
    k_shot = args.k_shot; image_size = args.image_size; anomaly_threshold = args.anomaly_threshold
    save_path = os.path.join(args.save_path, f"{dataset_name}_sequential_classified") 
    os.makedirs(save_path, exist_ok=True)
    txt_path = os.path.join(save_path, "log_sequential_classified.txt")

    # --- Logger ---
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S", handlers=[logging.FileHandler(txt_path, mode='w'), logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    for arg in vars(args): logger.info(f"{arg}: {getattr(args, arg)}")
    
    # --- Seed e Carregamento de Modelo ---
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    logger.info(f"Seed para reprodutibilidade definida: {args.seed}")
    logger.info("Carregando modelo UniVAD...")
    UniVAD_model = UniVAD(image_size=args.image_size).to(device)
    logger.info("Modelo UniVAD carregado.")
    logger.info("Carregando modelos de segmentação (GroundingDINO + SAM)...")
    segmentation_handler = SegmentationHandler(device=args.device)
    segmentation_handler.load_models()

    # --- Configuração do Dataset ---
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    if dataset_name == "cardboard_box":
        test_data = CardboardBoxDataset(root=dataset_dir, transform=transform, target_transform=transform, mode="test")
    else: raise NotImplementedError(f"Dataset '{dataset_name}' não suportado")
    use_shuffle = True
    logger.info(f"DataLoader shuffle está {'ATIVADO' if use_shuffle else 'DESATIVADO'}")
    test_dataloader = DataLoaderX(test_data, batch_size=1, shuffle=use_shuffle, num_workers=2, pin_memory=True, generator=torch.Generator().manual_seed(args.seed) if use_shuffle else None)
    
    # --- Variáveis ---
    results = {}; results["cls_names"] = []; results["imgs_masks"] = []; results["anomaly_maps"] = []; results["gt_sp"] = []; results["pr_sp"] = []
    cls_last = None; grounding_config = None
    image_transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    all_true_labels = []; all_pred_labels = []

    logger.info("Iniciando o loop de inferência sequencial...")
    
    for items in tqdm(test_dataloader, desc="Processando Amostras"):
        image_path = items["img_path"][0]
        try: 
            image = items["img"].to(device); image_pil = items["img_pil"]; cls_name = items["cls_name"][0]
            true_anomaly_label = items["anomaly"].item()
            
            seg_time = 0.0
            pred_time = 0.0
            
            if args.class_name != "None" and args.class_name.replace("_", " ") != cls_name: continue
            
            if cls_name != cls_last:
                logger.info(f"\nConfigurando para a nova classe: {cls_name}")
                cls_name_on_disk = cls_name.replace(" ", "_")
                if dataset_name == "cardboard_box":
                    train_good_dir = os.path.join(dataset_dir, cls_name_on_disk, 'train', 'good')
                    normal_image_paths = sorted(glob.glob(os.path.join(train_good_dir, '*.*')))[:k_shot]
                else: normal_image_paths = []
                if not normal_image_paths: raise FileNotFoundError(f"Nenhuma imagem de referência (k-shot) encontrada para {cls_name}")
                config_path = os.path.join(args.config_path_base, f"{cls_name_on_disk}.yaml")
                if os.path.exists(config_path):
                    config = read_config(config_path); grounding_config = config.get("grounding_config")
                    if grounding_config: logger.info(f"Config de segmentação para {cls_name} carregada.")
                else: grounding_config = None
                if grounding_config:
                    dataset_base_path = os.path.relpath(dataset_dir, "./data")
                    reference_mask_path = os.path.join("./masks", dataset_base_path, cls_name_on_disk)
                    
                    normal_image_paths_linux = [p.replace(os.path.sep, '/') for p in normal_image_paths]
                    
                    logger.info(f"Gerando máscaras de referência (k-shot) em: {reference_mask_path}")
                    segmentation_handler.segment(normal_image_paths_linux, reference_mask_path, grounding_config, find_only_one_object=args.find_only_one_object)
                    
                    logger.info("Aplicando workaround de renomeação nas máscaras de referência...")
                    for ref_path in normal_image_paths:
                        relative_path_part = os.path.relpath(ref_path, os.path.join(dataset_dir, cls_name_on_disk))
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
                
                normal_images = torch.cat([image_transform(Image.open(x).convert("RGB")).unsqueeze(0) for x in normal_image_paths], dim=0).to(device)
                setup_data = {"few_shot_samples": normal_images, "dataset_category": cls_name_on_disk, "image_path": normal_image_paths_linux}
                UniVAD_model.setup(setup_data)
                logger.info(f"Setup do UniVAD para {cls_name} concluído.")
                cls_last = cls_name

            expected_mask_path_test = None
            if grounding_config:
                cls_name_on_disk = cls_name.replace(" ", "_")
                dataset_base_path = os.path.relpath(dataset_dir, "./data")
                mask_output_dir = os.path.join("./masks", dataset_base_path, cls_name_on_disk)
                image_path_linux = image_path.replace(os.path.sep, '/')
                
                seg_start_time = time.time()
                segmentation_handler.segment([image_path_linux], mask_output_dir, grounding_config)
                seg_time = time.time() - seg_start_time
                
                relative_path_part_test = os.path.relpath(image_path, os.path.join(dataset_dir, cls_name_on_disk))
                img_name_slug_dir_test = os.path.dirname(relative_path_part_test)
                img_name_slug_base_test = os.path.splitext(os.path.basename(relative_path_part_test))[0]
                
                actual_mask_dir_test = os.path.join(mask_output_dir, img_name_slug_dir_test, img_name_slug_base_test)
                actual_mask_file_test = os.path.join(actual_mask_dir_test, 'grounding_mask.png')
                
                relative_dest_path_test = os.path.relpath(image_path, './data')
                expected_mask_path_test = os.path.join("./masks", relative_dest_path_test)
                
                if os.path.exists(actual_mask_file_test):
                    os.makedirs(os.path.dirname(expected_mask_path_test), exist_ok=True)
                    shutil.move(actual_mask_file_test, expected_mask_path_test)
                    if not os.listdir(actual_mask_dir_test): os.rmdir(actual_mask_dir_test)
                elif not os.path.exists(expected_mask_path_test):
                    logger.warning(f"Máscara de teste {actual_mask_file_test} não gerada. Pulando imagem.")
                    continue

            with torch.no_grad():
                image_path_linux = image_path.replace(os.path.sep, '/')
                pred_start_time = time.time()
                pred_value = UniVAD_model(image, image_path_linux, image_pil, debug=args.debug)
                pred_time = time.time() - pred_start_time
                _, anomaly_map_tensor = (pred_value["pred_score"], pred_value["pred_mask"])
                anomaly_map = anomaly_map_tensor.squeeze().detach().cpu().numpy()
            
            # --- Lógica de cálculo de score ---
            seg_mask_np = None; score_log_info = ""
            if args.filter_with_mask and grounding_config and expected_mask_path_test and os.path.exists(expected_mask_path_test):
                try:
                    seg_mask = Image.open(expected_mask_path_test).convert('L').resize((image_size, image_size), Image.NEAREST)
                    seg_mask_np = np.array(seg_mask) > 0
                    if not np.any(seg_mask_np): score_log_info = "(Máscara vazia) "
                except Exception as e:
                    logger.warning(f"Não foi possível aplicar filtro de máscara: {e}")
            overall_anomaly_score = calculate_top_k_score(anomaly_map, args.top_k_percent, mask=seg_mask_np)
            k_percent_str = f"{args.top_k_percent*100:.0f}%" if args.top_k_percent > 0 else "Max"
            results["cls_names"].append(cls_name); results["imgs_masks"].append(items["img_mask"])
            results["anomaly_maps"].append(anomaly_map); results["pr_sp"].append(overall_anomaly_score); results["gt_sp"].append(true_anomaly_label)

            # --- Lógica de classificação e log ---
            predicted_label = 1 if overall_anomaly_score > anomaly_threshold else 0
            all_true_labels.append(true_anomaly_label); all_pred_labels.append(predicted_label)
            is_correct = (predicted_label == true_anomaly_label)
            status_symbol = "✅" if is_correct else "❌"
            true_label_str = "BAD" if true_anomaly_label == 1 else "GOOD"
            pred_label_str = "BAD" if predicted_label == 1 else "GOOD"
            log_message = (f"{status_symbol} Imagem: {os.path.basename(image_path):<20} | Score: {overall_anomaly_score:.4f} {score_log_info} | "
                           f"Prev: {pred_label_str:<4} | Real: {true_label_str:<4} | "
                           f"T_Seg: {seg_time:.2f}s | T_Pred: {pred_time:.2f}s")
            logger.info(log_message)

            # --- Lógica para Salvar Heatmap ---
            output_dir = os.path.join(save_path, "overlaid_results", cls_name.replace(" ", "_")); os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filepath = os.path.join(output_dir, f"{base_name}_overlay.png")
            original_img_np = (image.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            original_img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)
            heatmap_normalized = (anomaly_map - np.min(anomaly_map)) / (np.max(anomaly_map) - np.min(anomaly_map) + 1e-8)
            heatmap_uint8 = (heatmap_normalized * 255).astype(np.uint8)
            heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            overlayed_img_bgr = cv2.addWeighted(original_img_bgr, 0.6, heatmap_colored_bgr, 0.4, 0)
            if args.filter_with_mask and seg_mask_np is not None and np.any(seg_mask_np):
                final_display_img = np.where(seg_mask_np[..., None], overlayed_img_bgr, original_img_bgr)
            else: final_display_img = overlayed_img_bgr
            Image.fromarray(cv2.cvtColor(final_display_img, cv2.COLOR_BGR2RGB)).save(output_filepath)
            
        except Exception as e:
            logger.error(f"\n--- ERRO AO PROCESSAR AMOSTRA: {image_path} ---"); import traceback; logger.error(traceback.format_exc()); logger.error("Pulando esta amostra.\n")
            if 'true_anomaly_label' in locals():
                all_true_labels.append(true_anomaly_label); all_pred_labels.append(1 - true_anomaly_label)
            if results["cls_names"] and len(results["cls_names"]) > len(results["pr_sp"]):
                results["cls_names"].pop(); results["imgs_masks"].pop(); results["gt_sp"].pop()
            continue

    # --- Relatórios Finail ---
    logger.info("\n" + "="*80 + "\nRelatório Final de Métricas (Threshold-Independente)\n" + "="*80)
    table_ls, auroc_sp_ls, auroc_px_ls = [], [], []
    processed_obj_list = sorted(list(set(results.get("cls_names", []))))
    for obj in tqdm(processed_obj_list, desc="Calculando Métricas AUROC"): cal_score(obj)
    if auroc_sp_ls:
        mean_auroc_sp = np.round(np.nanmean(auroc_sp_ls) * 100, decimals=1); mean_auroc_px = np.round(np.nanmean(auroc_px_ls) * 100, decimals=1)
        mean_auroc_px_str = str(mean_auroc_px) if not np.isnan(mean_auroc_px) else "N/A"
        table_ls.append(["mean", str(mean_auroc_sp), mean_auroc_px_str])
        results_table = tabulate(table_ls, headers=["Objects", "AUROC Imagem (%)", "AUROC Pixel (%)"], tablefmt="pipe"); logger.info("\n%s\n", results_table)
    else: logger.warning("Nenhuma métrica AUROC pôde ser calculada.")
    if all_true_labels:
        logger.info("\n" + "="*80 + f"\nRelatório de Classificação (Threshold = {anomaly_threshold})\n" + "="*80)
        tn, fp, fn, tp = confusion_matrix(all_true_labels, all_pred_labels, labels=[0, 1]).ravel()
        conf_matrix_data = [["Real \\ Previsto", "Prev: GOOD (0)", "Prev: BAD (1)"], ["Real: GOOD (0)", tn, fp], ["Real: BAD (1)", fn, tp]]
        logger.info("--- Matriz de Confusão ---\n" + tabulate(conf_matrix_data, headers="firstrow", tablefmt="heavy_grid"))
        acc = accuracy_score(all_true_labels, all_pred_labels)
        specificity_good = tn / (tn + fp) if (tn + fp) > 0 else 0; recall_bad = tp / (tp + fn) if (tp + fn) > 0 else 0; precision_bad = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics_summary = [["Acurácia Geral", f"{acc * 100:.2f}%"], ["Acurácia Classe GOOD (Especificidade)", f"{specificity_good * 100:.2f}%"], ["Acurácia Classe BAD (Recall)", f"{recall_bad * 100:.2f}%"], ["Precisão Classe BAD", f"{precision_bad * 100:.2f}%"]]
        logger.info("\n--- Métricas de Classificação ---\n" + tabulate(metrics_summary, tablefmt="heavy_grid"))
    else: logger.warning("Nenhuma amostra foi processada, relatório de classificação não pode ser gerado.")
    logger.info(f"\nProcessamento concluído. Resultados salvos em: {save_path}")