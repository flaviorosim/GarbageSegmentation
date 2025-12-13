import json
import os
import cv2
import numpy as np
from tqdm import tqdm

# --- CONFIGURAÇÃO ---
# Lista com os dois arquivos de anotação
ANNOTATION_FILES = ['annotations.json', 'annotations_unofficial.json']
IMAGE_DIR = 'taco_dataset/data'
MASK_OUTPUT_DIR = 'taco_dataset/masks'

# Mapeamento de Categorias (TACO -> Super Categorias 1, 2, 3, 4)
taco_to_super_category = {
    # 1: PLÁSTICO
    4: 1, 5: 1, 7: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 
    36: 1, 37: 1, 38: 1, 39: 1, 40: 1, 41: 1, 42: 1, 43: 1, 44: 1, 45: 1, 
    46: 1, 47: 1, 48: 1, 49: 1, 55: 1, 57: 1,
    # 2: PAPEL
    13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 18: 2, 19: 2, 20: 2, 
    30: 2, 31: 2, 32: 2, 33: 2, 34: 2, 35: 2,
    # 3: METAL
    0: 3, 1: 3, 2: 3, 3: 3, 8: 3, 10: 3, 11: 3, 12: 3, 28: 3, 29: 3, 50: 3, 51: 3, 52: 3,
    # 4: OUTROS
    6: 4, 9: 4, 53: 4, 54: 4, 56: 4, 58: 4, 59: 4
}

def generate_standardized_masks():
    os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)
    
    total_masks = 0
    non_empty_masks = 0

    for ann_file in ANNOTATION_FILES:
        if not os.path.exists(ann_file):
            print(f"AVISO: {ann_file} não encontrado. Pulando.")
            continue
            
        print(f"\n--- Lendo {ann_file} ---")
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)

        # Mapeamento rápido
        images_info = {img['id']: img for img in coco_data['images']}
        annotations_map = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_map:
                annotations_map[img_id] = []
            annotations_map[img_id].append(ann)

        for image_id, image_info in tqdm(images_info.items()):
            # 1. Preparar a máscara preta
            mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
            
            # 2. Desenhar as anotações (se houver)
            has_pixels = False
            if image_id in annotations_map:
                # Ordenar por área para pintar os pequenos por cima
                anns = sorted(annotations_map[image_id], key=lambda x: x['area'], reverse=True)
                
                for ann in anns:
                    cat_id = ann['category_id']
                    # Converte para 1, 2, 3 ou 4. Se não achar, usa 4.
                    super_id = taco_to_super_category.get(cat_id, 4)
                    
                    if 'segmentation' in ann:
                        for polygon in ann['segmentation']:
                            if polygon:
                                points = np.array(polygon, dtype=np.int32).reshape((-1, 2))
                                cv2.fillPoly(mask, [points], color=super_id)
                                has_pixels = True

            # 3. FORÇAR NOME .PNG
            # Pega o nome original (ex: batch_1/000006.jpg)
            original_filename = image_info['file_name']
            
            # Remove a extensão antiga (.jpg, .jpeg, etc) e adiciona .png
            filename_base = os.path.splitext(original_filename)[0]
            final_filename = filename_base + ".png"
            
            # Caminho completo de saída
            output_path = os.path.join(MASK_OUTPUT_DIR, final_filename)
            
            # Criar subpastas (ex: masks/batch_1)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Salvar como PNG (Sem perdas!)
            cv2.imwrite(output_path, mask)
            
            total_masks += 1
            if has_pixels:
                non_empty_masks += 1

    print(f"\n=== RELATÓRIO FINAL ===")
    print(f"Total de máscaras geradas: {total_masks}")
    print(f"Máscaras com lixo (pixels > 0): {non_empty_masks}")
    print(f"Máscaras vazias (apenas fundo): {total_masks - non_empty_masks}")
    print(f"Todas salvas em .png na pasta '{MASK_OUTPUT_DIR}'")
    print("Nota: As imagens parecerão pretas no visualizador, mas contêm os valores 1, 2, 3, 4.")

if __name__ == '__main__':
    generate_standardized_masks()