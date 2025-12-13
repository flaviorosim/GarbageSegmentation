import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURAÇÃO ---
# ESCOLHA UM PAR DE IMAGEM E MÁSCARA QUE VOCÊ GEROU PARA TESTAR
# Exemplo: Se você tem data/batch_1/000006.jpg, a máscara será masks/batch_1/000006.png

# Substitua pelos caminhos reais de um par no seu computador:
IMAGE_PATH = 'taco_dataset/data/batch_3/IMG_4994.jpg'  # Caminho da imagem original
MASK_PATH = 'taco_dataset/masks/batch_3/IMG_4994.png'    # Caminho da máscara gerada (.png)

# Cores para os contornos (em BGR para OpenCV)
# 1: Plástico (Vermelho), 2: Papel (Verde), 3: Metal (Azul), 4: Outros (Amarelo)
COLORS = {
    1: (0, 0, 255),   # Vermelho
    2: (0, 255, 0),   # Verde
    3: (255, 0, 0),   # Azul
    4: (0, 255, 255)  # Amarelo (Ciano no BGR, na vdd)
}
LABELS = {1: "Plastico", 2: "Papel", 3: "Metal", 4: "Outros"}

def visualize_contours():
    if not os.path.exists(IMAGE_PATH) or not os.path.exists(MASK_PATH):
        print("Erro: Arquivos não encontrados. Verifique os caminhos no script.")
        print(f"Img: {IMAGE_PATH}")
        print(f"Msk: {MASK_PATH}")
        return

    # 1. Carregar imagens
    # Imagem original em cores
    img_orig = cv2.imread(IMAGE_PATH)
    img_display = img_orig.copy() # Cópia para desenhar em cima
    
    # Máscara em escala de cinza (IMPORTANTE: IMREAD_GRAYSCALE)
    mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

    print(f"Analisando máscara: {MASK_PATH}")
    # Verifica quais valores (IDs) existem nesta máscara
    unique_values = np.unique(mask)
    print(f"Valores encontrados na máscara: {unique_values}")

    if len(unique_values) == 1 and unique_values[0] == 0:
        print("Aviso: Esta máscara é totalmente preta (só fundo). Escolha outra imagem.")

    # 2. Loop para desenhar contornos para cada ID encontrado
    found_any = False
    for class_id in [1, 2, 3, 4]:
        if class_id in unique_values:
            found_any = True
            # Cria uma máscara temporária só para este ID
            # Onde for igual ao ID vira branco (255), o resto preto (0)
            binary_mask_for_id = np.where(mask == class_id, 255, 0).astype(np.uint8)
            
            # Encontra contornos nesta máscara binária
            contours, _ = cv2.findContours(binary_mask_for_id, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Desenha os contornos na imagem de visualização
            # thickness=2 (espessura da linha)
            cv2.drawContours(img_display, contours, -1, COLORS[class_id], thickness=3)
            print(f"-> Desenhados contornos para ID {class_id} ({LABELS[class_id]}) - Cor: {COLORS[class_id]}")

    # 3. Plotar usando Matplotlib para ver o resultado
    # Converter BGR (OpenCV) para RGB (Matplotlib) para as cores ficarem certas na tela
    img_orig_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img_display_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 10))

    # Imagem 1: Original
    plt.subplot(1, 2, 1)
    plt.title("Imagem Original")
    plt.imshow(img_orig_rgb)
    plt.axis('off')

    # Imagem 2: Com contornos
    plt.subplot(1, 2, 2)
    plt.title("Visualização dos Contornos da Máscara")
    plt.imshow(img_display_rgb)
    plt.axis('off')
    
    # Legenda improvisada no título
    legend_text = "Legenda: " + ", ".join([f"{LABELS[k]}={COLORS[k]}" for k in COLORS if k in unique_values])
    plt.xlabel(legend_text)

    plt.tight_layout()
    print("Mostrando imagem...")
    plt.show()

if __name__ == '__main__':
    visualize_contours()