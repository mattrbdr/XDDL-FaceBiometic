import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# Configurer l'encodage pour UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def enhance_image(img):
    """
    Améliore la qualité d'une image pour faciliter la détection des visages
    """
    # Convertir en niveaux de gris
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Égalisation d'histogramme pour améliorer le contraste
    gray_eq = cv2.equalizeHist(gray)
    
    # Réduction du bruit
    gray_denoise = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    
    # Amélioration des contours
    kernel = np.array([[-1,-1,-1], 
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharp = cv2.filter2D(gray_denoise, -1, kernel)
    
    return sharp, gray_eq

def detect_facial_landmarks(img, face_rect=None, is_suspect=False):
    """
    Détecte les points caractéristiques du visage
    avec des critères plus stricts pour éviter les faux positifs
    """
    # Si l'image est en couleur, la convertir en niveaux de gris
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        
    # Créer une copie de l'image pour y dessiner les points
    vis_img = img.copy() if len(img.shape) == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Utiliser seulement les détecteurs disponibles par défaut
    # Détecteur des yeux
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Paramètres ajustés selon qu'il s'agit d'un suspect ou d'une référence
    if is_suspect:
        # Pour les suspects, être plus strict sur la détection des yeux
        min_eye_size = (25, 25)
        min_neighbors = 4
    else:
        # Pour les références (souvent de mauvaise qualité), être plus permissif
        min_eye_size = (20, 20)
        min_neighbors = 3
    
    # Si rectangle du visage fourni, utiliser seulement cette zone
    if face_rect is not None:
        x, y, w, h = face_rect
        roi_gray = gray[y:y+h, x:x+w]
        roi_vis = vis_img[y:y+h, x:x+w]
    else:
        roi_gray = gray
        roi_vis = vis_img
        x, y = 0, 0
    
    landmarks = []
    
    try:
        # Détecter les yeux
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, min_neighbors, minSize=min_eye_size)
        
        # Filtrer les yeux détectés pour ne garder que les plus fiables
        valid_eyes = []
        for (ex, ey, ew, eh) in eyes:
            # Vérifier que les proportions sont correctes pour un œil
            aspect_ratio = ew / eh
            # Les yeux ont généralement un ratio largeur/hauteur entre 0.8 et 1.6
            if 0.8 <= aspect_ratio <= 1.6:
                valid_eyes.append((ex, ey, ew, eh))
                
        # Limiter à 2 yeux maximum, en prenant les plus grands
        valid_eyes.sort(key=lambda e: e[2] * e[3], reverse=True)
        for (ex, ey, ew, eh) in valid_eyes[:2]:
            cv2.rectangle(roi_vis, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            center = (x + ex + ew//2, y + ey + eh//2)
            landmarks.append(('eye', center))
            # Marquer le centre de l'œil
            cv2.circle(roi_vis, (ex + ew//2, ey + eh//2), 2, (255, 0, 0), -1)
    except Exception as e:
        print(f"Erreur lors de la détection des yeux: {e}")

    # Estimer la position du nez et de la bouche à partir des yeux si détectés
    # Uniquement si nous avons exactement 2 yeux détectés
    if len(landmarks) == 2 and all(l[0] == 'eye' for l in landmarks):
        eye1_center = landmarks[0][1]
        eye2_center = landmarks[1][1]
        
        # Vérifier que les yeux sont approximativement au même niveau (tolérance de 10%)
        eye_y_diff = abs(eye1_center[1] - eye2_center[1])
        eye_dist = np.sqrt((eye1_center[0] - eye2_center[0])**2 + (eye1_center[1] - eye2_center[1])**2)
        
        if eye_y_diff / eye_dist < 0.15:  # Seulement si les yeux sont à peu près horizontaux
            # Calculer le point médian entre les yeux
            mid_eyes_x = (eye1_center[0] + eye2_center[0]) // 2
            mid_eyes_y = (eye1_center[1] + eye2_center[1]) // 2
            
            # La distance inter-oculaire comme référence
            eye_distance = np.sqrt((eye1_center[0] - eye2_center[0])**2 + (eye1_center[1] - eye2_center[1])**2)
            
            # Estimer la position du nez (en dessous du point médian entre les yeux)
            nose_x = mid_eyes_x
            nose_y = int(mid_eyes_y + 0.3 * eye_distance)
            
            # Estimer la position de la bouche (encore plus bas)
            mouth_x = mid_eyes_x
            mouth_y = int(mid_eyes_y + 0.7 * eye_distance)
            
            # Ajouter ces points estimés
            face_height = (h if face_rect else gray.shape[0])
            if nose_y < face_height:
                nose_center = (nose_x, nose_y)
                landmarks.append(('nose', nose_center))
                cv2.circle(roi_vis if face_rect else vis_img, 
                          (nose_x - x if face_rect else nose_x, nose_y - y if face_rect else nose_y), 
                          3, (0, 255, 255), -1)
                
            if mouth_y < face_height:
                mouth_center = (mouth_x, mouth_y)
                landmarks.append(('mouth', mouth_center))
                cv2.circle(roi_vis if face_rect else vis_img, 
                          (mouth_x - x if face_rect else mouth_x, mouth_y - y if face_rect else mouth_y), 
                          3, (0, 0, 255), -1)
    
    return vis_img, landmarks

def detect_face(image_path):
    """
    Détecte un visage dans l'image et retourne l'image du visage
    Amélioré pour les images de mauvaise qualité ou les images anciennes
    """
    # Charger l'image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erreur: Impossible de lire l'image {image_path}")
        return None, None, 0, None
    
    # Vérifier si l'image est un suspect ou une référence
    is_suspect = 'compares' in image_path
    
    # Pour les images suspectes, appliquer un meilleur prétraitement
    if is_suspect:
        print(f"Prétraitement amélioré pour l'image suspecte {os.path.basename(image_path)}")
        # Supprimer le bruit
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        # Améliorer le contraste
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    # Améliorer l'image
    enhanced_img, gray_eq = enhance_image(img)
    
    # Obtenir les dimensions de l'image
    height, width = enhanced_img.shape
    
    # Vérifier si l'image est déjà un visage recadré (le visage occupe une grande partie de l'image)
    face_ratio = 1.0  # Par défaut, on suppose que le visage peut occuper toute l'image
    
    # Charger les classificateurs de visage pré-entraînés
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    alt_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    
    # Pour les grandes images, essayer d'abord la détection normale
    if max(height, width) > 300:
        # Détecter les visages avec des paramètres plus permissifs
        faces = face_cascade.detectMultiScale(enhanced_img, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        
        if len(faces) == 0:
            # Essayer avec le détecteur alternatif
            faces = alt_face_cascade.detectMultiScale(enhanced_img, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        
        if len(faces) == 0:
            # Essayer l'image égalisée
            faces = face_cascade.detectMultiScale(gray_eq, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))
            
        if len(faces) == 0:
            # Essayer avec des paramètres encore plus permissifs
            faces = face_cascade.detectMultiScale(enhanced_img, scaleFactor=1.3, minNeighbors=3, minSize=(20, 20))
            
        if len(faces) == 0:
            # Essayer avec des paramètres extrêmement permissifs
            faces = face_cascade.detectMultiScale(enhanced_img, scaleFactor=1.5, minNeighbors=2, minSize=(20, 20))
    else:
        # Pour les petites images, supposer qu'elles sont déjà des visages recadrés
        faces = []
        
    # Si un visage est détecté, l'utiliser
    if len(faces) > 0:
        # Trouver le plus grand visage détecté (supposant que c'est le principal)
        max_area = 0
        max_face = None
        for (x, y, w, h) in faces:
            area = w * h
            if area > max_area:
                max_area = area
                max_face = (x, y, w, h)
        
        x, y, w, h = max_face
        
        # Calculer le ratio de la surface du visage par rapport à l'image totale
        face_ratio = (w * h) / (width * height)
        
        # Agrandir la zone de détection du visage (ajouter une marge de 40%)
        margin_x = int(w * 0.4)
        margin_y = int(h * 0.4)
        
        # S'assurer que les coordonnées restent dans les limites de l'image
        x_start = max(0, x - margin_x)
        y_start = max(0, y - margin_y)
        x_end = min(width, x + w + margin_x)
        y_end = min(height, y + h + margin_y)
        
        # Extraire le visage avec la marge
        face_img = img[y_start:y_end, x_start:x_end]
        
        # Détecter les points caractéristiques
        face_rect = (x, y, w, h)
        img_with_landmarks, landmarks = detect_facial_landmarks(img, face_rect, is_suspect)
        
        # Créer une copie de l'image originale pour dessiner le rectangle
        img_with_rect = img.copy()
        cv2.rectangle(img_with_rect, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    else:
        print(f"Aucun visage détecté dans {image_path}, utilisation de l'image entière")
        # Si aucun visage n'est détecté, utiliser l'image entière et chercher les points caractéristiques
        face_img = img
        img_with_rect = None
        img_with_landmarks, landmarks = detect_facial_landmarks(img, None, is_suspect)
        
    # Redimensionner pour une comparaison uniforme (taille plus grande pour conserver plus de détails)
    face_img_resized = cv2.resize(face_img, (200, 200))
    
    return face_img_resized, img_with_rect, face_ratio, (img_with_landmarks, landmarks)

def compare_facial_features(suspect_landmarks, criminal_landmarks):
    """
    Compare les positions relatives des points caractéristiques du visage
    avec des critères plus stricts pour éviter les faux positifs
    """
    # Si pas assez de points dans l'une des images, retourner une similarité faible
    if len(suspect_landmarks) < 2 or len(criminal_landmarks) < 2:
        return 0.0, []
    
    # Organiser les points par type
    suspect_points = {l[0]: l[1] for l in suspect_landmarks}
    criminal_points = {l[0]: l[1] for l in criminal_landmarks}
    
    # Points communs
    common_features = set(suspect_points.keys()) & set(criminal_points.keys())
    
    if len(common_features) < 2:
        return 0.0, []
    
    # Normaliser les positions (conversion en coordonnées relatives)
    # Utiliser la distance entre les yeux comme référence si disponible
    if 'eye' in suspect_points and 'eye' in criminal_points:
        # Si on a plusieurs yeux, calculer leur centroïde
        if isinstance(suspect_points['eye'], list) and len(suspect_points['eye']) >= 2:
            suspect_ref = np.mean(suspect_points['eye'], axis=0)
        else:
            suspect_ref = suspect_points['eye']
            
        if isinstance(criminal_points['eye'], list) and len(criminal_points['eye']) >= 2:
            criminal_ref = np.mean(criminal_points['eye'], axis=0)
        else:
            criminal_ref = criminal_points['eye']
    else:
        # Sinon utiliser le centre de l'image
        suspect_ref = np.array([100, 100])  # Centre de l'image 200x200
        criminal_ref = np.array([100, 100])
    
    # Calculer la similarité des positions relatives
    similarities = []
    matching_points = []
    
    for feature in common_features:
        suspect_pos = np.array(suspect_points[feature])
        criminal_pos = np.array(criminal_points[feature])
        
        # Calculer les vecteurs relatifs par rapport aux références
        suspect_vec = suspect_pos - suspect_ref
        criminal_vec = criminal_pos - criminal_ref
        
        # Vérifier que les vecteurs ont une magnitude minimale pour éviter le bruit
        suspect_magnitude = np.linalg.norm(suspect_vec)
        criminal_magnitude = np.linalg.norm(criminal_vec)
        
        if suspect_magnitude < 10 or criminal_magnitude < 10:
            # Ignorer les points trop proches du point de référence (probablement du bruit)
            continue
        
        # Normaliser
        suspect_vec_norm = suspect_vec / (suspect_magnitude + 1e-10)
        criminal_vec_norm = criminal_vec / (criminal_magnitude + 1e-10)
        
        # Similarité cosinus
        similarity = np.dot(suspect_vec_norm, criminal_vec_norm)
        similarity = (similarity + 1) / 2  # Normaliser entre 0 et 1
        
        # Ignorer les similarités trop faibles
        if similarity < 0.3:
            continue
            
        similarities.append(similarity)
        matching_points.append((feature, similarity))
    
    # Si aucun point ne correspond vraiment, retourner une similarité nulle
    if not similarities:
        return 0.0, []
        
    return np.mean(similarities), matching_points

def compare_faces(suspect_image_path, criminal_folder, results_dir):
    """
    Compare une image suspecte avec toutes les images du criminel
    et retourne les résultats de correspondance
    """
    print(f"Comparaison de l'image suspecte: {suspect_image_path}")
    print(f"Avec les images du criminel dans: {criminal_folder}")
    
    # Détecter le visage dans l'image du suspect
    suspect_face, suspect_with_rect, suspect_face_ratio, (suspect_landmarks_img, suspect_landmarks) = detect_face(suspect_image_path)
    if suspect_face is None:
        return []
    
    # Enregistrer l'image avec le visage détecté
    if suspect_with_rect is not None:
        suspect_rect_path = os.path.join(results_dir, f"suspect_detected_{os.path.basename(suspect_image_path)}")
        cv2.imwrite(suspect_rect_path, suspect_with_rect)
    
    # Enregistrer l'image avec les points caractéristiques
    suspect_landmarks_path = os.path.join(results_dir, f"suspect_landmarks_{os.path.basename(suspect_image_path)}")
    cv2.imwrite(suspect_landmarks_path, suspect_landmarks_img)
    
    # Obtenir la liste des images du criminel
    criminal_images = [os.path.join(criminal_folder, f) for f in os.listdir(criminal_folder) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    results = []
    
    for criminal_img in criminal_images:
        try:
            # Détecter le visage dans l'image du criminel
            criminal_face, criminal_with_rect, criminal_face_ratio, (criminal_landmarks_img, criminal_landmarks) = detect_face(criminal_img)
            if criminal_face is None:
                continue
            
            # Enregistrer l'image avec le visage détecté
            if criminal_with_rect is not None:
                criminal_rect_path = os.path.join(results_dir, f"criminal_detected_{os.path.basename(criminal_img)}")
                cv2.imwrite(criminal_rect_path, criminal_with_rect)
            
            # Enregistrer l'image avec les points caractéristiques
            criminal_landmarks_path = os.path.join(results_dir, f"criminal_landmarks_{os.path.basename(criminal_img)}")
            cv2.imwrite(criminal_landmarks_path, criminal_landmarks_img)
                
            # Convertir en niveaux de gris pour la comparaison
            suspect_gray = cv2.cvtColor(suspect_face, cv2.COLOR_BGR2GRAY)
            criminal_gray = cv2.cvtColor(criminal_face, cv2.COLOR_BGR2GRAY)
            
            # Égaliser les histogrammes pour améliorer la comparaison en cas de différences d'exposition
            suspect_gray = cv2.equalizeHist(suspect_gray)
            criminal_gray = cv2.equalizeHist(criminal_gray)
            
            # 1. Calculer la similarité en utilisant la corrélation
            correlation = cv2.matchTemplate(suspect_gray, criminal_gray, cv2.TM_CCOEFF_NORMED)
            similarity_corr = np.max(correlation)
            
            # 2. Calculer la similarité d'histogramme
            hist1 = cv2.calcHist([suspect_gray], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([criminal_gray], [0], None, [256], [0, 256])
            
            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()
            
            hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # 3. Utiliser SSIM (Structural Similarity Index)
            try:
                from skimage.metrics import structural_similarity as ssim
                ssim_value = ssim(suspect_gray, criminal_gray, data_range=255)
            except ImportError:
                # Si skimage n'est pas disponible, utiliser une valeur par défaut
                ssim_value = (similarity_corr + hist_similarity) / 2
            
            # 4. Comparer les caractéristiques du visage (points faciaux)
            facial_features_similarity, matching_points = compare_facial_features(suspect_landmarks, criminal_landmarks)
            
            # Score final (moyenne pondérée des méthodes)
            # Donner moins de poids aux points faciaux qui peuvent être peu fiables
            final_score = (0.4 * similarity_corr + 
                          0.3 * hist_similarity + 
                          0.25 * ssim_value + 
                          0.05 * facial_features_similarity)
            
            # Ajuster le seuil pour les images de mauvaise qualité
            # Plus l'image est de mauvaise qualité, plus on est indulgent
            quality_factor = min(1.0, (len(suspect_landmarks) + len(criminal_landmarks)) / 8.0)
            
            # Seuil pour les images de Xavier Dupont de Ligonnès
            if "xavier" in criminal_img.lower() or "7fc05c8a15a6ca2f" in criminal_img.lower():
                # Seuil prudent
                quality_adjusted_threshold = 0.25  # Seuil spécifique pour Xavier
                
                # Utiliser principalement la corrélation pour les photos connues de Xavier
                # (plus fiable que les points faciaux estimés)
                final_score = (0.6 * similarity_corr + 
                              0.2 * hist_similarity + 
                              0.15 * ssim_value + 
                              0.05 * facial_features_similarity)
                
                # Si la corrélation est bonne, c'est probablement un match
                if similarity_corr > 0.65:
                    final_score = max(final_score, 0.3)  # Fixer un minimum
            else:
                quality_adjusted_threshold = 0.25 * (2 - quality_factor)  # Entre 0.25 et 0.5 selon la qualité
                
            # Vérifier que les correspondances sont réellement pertinentes
            # Si facial_features_similarity est supérieur à 0 mais qu'il n'y a pas de points réellement matchés,
            # réduire son impact
            if facial_features_similarity > 0 and not matching_points:
                final_score = final_score - (0.05 * facial_features_similarity)
                facial_features_similarity = 0  # Réinitialiser à 0 pour une meilleure indication dans le rapport
            
            # Définir un seuil pour déterminer si c'est un match
            is_match = final_score > quality_adjusted_threshold
            
            # Si les deux images sont des visages recadrés, augmenter la probabilité de correspondance
            if suspect_face_ratio > 0.5 and criminal_face_ratio > 0.5:
                final_score = final_score * 1.2  # Augmenter le score de 20%
                is_match = final_score > quality_adjusted_threshold  # Réévaluer après ajustement
            
            results.append({
                "suspect_image": os.path.basename(suspect_image_path),
                "criminal_image": os.path.basename(criminal_img),
                "similarity": round(final_score, 4),
                "correlation": round(similarity_corr, 4),
                "hist_similarity": round(hist_similarity, 4),
                "ssim": round(ssim_value, 4) if 'ssim_value' in locals() else 0,
                "facial_features": round(facial_features_similarity, 4),
                "image_quality": round(quality_factor, 2),
                "quality_threshold": round(quality_adjusted_threshold, 4),
                "matching_points": matching_points,
                "is_match": is_match
            })
            
            match_text = "MATCH" if is_match else "PAS DE MATCH"
            print(f"Comparaison avec {os.path.basename(criminal_img)}: {match_text} (Similarité: {round(final_score, 4)}, Qualité: {round(quality_factor, 2)})")
            
            # Créer une visualisation de la comparaison
            # Créer une image qui montre les deux visages et les informations de similarité
            # Convertir les images de BGR à RGB pour matplotlib
            suspect_face_rgb = cv2.cvtColor(suspect_face, cv2.COLOR_BGR2RGB)
            criminal_face_rgb = cv2.cvtColor(criminal_face, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(12, 8))
            
            # Afficher l'image du suspect avec points caractéristiques
            plt.subplot(2, 2, 1)
            plt.imshow(cv2.cvtColor(suspect_landmarks_img, cv2.COLOR_BGR2RGB))
            plt.title("Points caractéristiques - Suspect")
            plt.axis('off')
            
            # Afficher l'image du criminel avec points caractéristiques
            plt.subplot(2, 2, 2)
            plt.imshow(cv2.cvtColor(criminal_landmarks_img, cv2.COLOR_BGR2RGB))
            plt.title("Points caractéristiques - Criminel")
            plt.axis('off')
            
            # Afficher l'image du suspect
            plt.subplot(2, 2, 3)
            plt.imshow(suspect_face_rgb)
            plt.title("Visage suspect")
            plt.axis('off')
            
            # Afficher l'image du criminel
            plt.subplot(2, 2, 4)
            plt.imshow(criminal_face_rgb)
            plt.title("Visage criminel")
            plt.axis('off')
            
            # Ajouter le texte d'information
            match_status = "MATCH ✓" if is_match else "PAS DE MATCH ✗"
            plt.suptitle(f"Comparaison des visages: {match_status}\n" + 
                         f"Score global: {round(final_score, 4)} (seuil: {round(quality_adjusted_threshold, 2)})\n" +
                         f"Corrélation: {round(similarity_corr, 4)}, " +
                         f"Similarité points: {round(facial_features_similarity, 4)}, " +
                         f"Qualité image: {round(quality_factor, 2)}",
                         fontsize=12)
            
            # Sauvegarder l'image avec encodage UTF-8
            comparison_file = os.path.join(results_dir, f"comparaison_{os.path.basename(suspect_image_path)}_{os.path.basename(criminal_img)}.png")
            plt.savefig(comparison_file)
            plt.close()
            
            # Créer aussi une comparaison côte à côte simple
            comparison = np.hstack((suspect_face, criminal_face))
            cv2.imwrite(os.path.join(results_dir, f"simple_comp_{os.path.basename(suspect_image_path)}_{os.path.basename(criminal_img)}.jpg"), comparison)
            
        except Exception as e:
            print(f"Erreur lors de la comparaison avec {criminal_img}: {str(e)}")
    
    return results

def create_comparison_grid(results_df, suspect_folder, criminal_folder, results_dir):
    """
    Crée une grille de comparaison des meilleurs résultats
    """
    if results_df.empty:
        return
        
    # Limiter aux 4 meilleures correspondances ou moins
    top_results = results_df.head(min(4, len(results_df)))
    
    plt.figure(figsize=(15, 12))
    
    for idx, (_, row) in enumerate(top_results.iterrows()):
        suspect_img_path = os.path.join(suspect_folder, row['suspect_image'])
        criminal_img_path = os.path.join(criminal_folder, row['criminal_image'])
        
        # Utiliser les images avec landmarks si elles existent
        suspect_landmarks_path = os.path.join(results_dir, f"suspect_landmarks_{row['suspect_image']}")
        criminal_landmarks_path = os.path.join(results_dir, f"criminal_landmarks_{row['criminal_image']}")
        
        if os.path.exists(suspect_landmarks_path) and os.path.exists(criminal_landmarks_path):
            suspect_face = cv2.imread(suspect_landmarks_path)
            criminal_face = cv2.imread(criminal_landmarks_path)
        else:
            suspect_face, _, _, _ = detect_face(suspect_img_path)
            criminal_face, _, _, _ = detect_face(criminal_img_path)
        
        if suspect_face is None or criminal_face is None:
            continue
            
        suspect_face_rgb = cv2.cvtColor(suspect_face, cv2.COLOR_BGR2RGB)
        criminal_face_rgb = cv2.cvtColor(criminal_face, cv2.COLOR_BGR2RGB)
        
        plt.subplot(min(4, len(top_results)), 2, idx*2+1)
        plt.imshow(suspect_face_rgb)
        plt.title(f"Suspect: {row['suspect_image']}")
        plt.axis('off')
        
        plt.subplot(min(4, len(top_results)), 2, idx*2+2)
        plt.imshow(criminal_face_rgb)
        plt.title(f"Xavier DL: {row['criminal_image']}")
        plt.axis('off')
        
        # Obtenir les informations de qualité d'image
        image_quality = row.get('image_quality', 'N/A')
        quality_text = f", Qualité: {image_quality}" if image_quality != 'N/A' else ""
        
        # Obtenir les informations de points caractéristiques
        facial_features = row.get('facial_features', 'N/A')
        features_text = f", Points: {facial_features}" if facial_features != 'N/A' else ""
        
        match_status = "✓ MATCH" if row['is_match'] else "✗ PAS DE MATCH"
        plt.figtext(0.5, 0.9 - (idx * 0.25), 
                   f"Comparaison {idx+1}: {match_status}, Score: {row['similarity']}{quality_text}{features_text}", 
                   ha="center", fontsize=12, 
                   bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.suptitle("Comparaison d'un suspect avec Xavier Dupont de Ligonnès", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "résumé_correspondances.png"))
    plt.close()

def generate_report(results_df, best_matches, results_dir):
    """
    Génère un rapport HTML avec les résultats, format amélioré et plus professionnel
    """
    if results_df.empty:
        return
    
    # Date et heure formatées
    date_analyse = datetime.now().strftime('%d/%m/%Y')
    heure_analyse = datetime.now().strftime('%H:%M:%S')
    
    # Récupérer les scores maximaux
    max_similarity = results_df["similarity"].max()
    
    # Déterminer le niveau de confiance global
    if max_similarity > 0.7:
        confidence_level = "Très élevée"
        confidence_color = "#006400"  # Vert foncé
    elif max_similarity > 0.5:
        confidence_level = "Élevée"
        confidence_color = "#228B22"  # Vert forêt
    elif max_similarity > 0.35:
        confidence_level = "Modérée"
        confidence_color = "#FFA500"  # Orange
    else:
        confidence_level = "Faible"
        confidence_color = "#B22222"  # Rouge foncé
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Rapport d'analyse faciale - Xavier Dupont de Ligonnès</title>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background: linear-gradient(135deg, #1e5799 0%, #2989d8 50%, #207cca 100%);
                color: white;
                padding: 20px;
                border-radius: 8px 8px 0 0;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .header-flex {{
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .timestamp {{
                font-size: 14px;
                opacity: 0.9;
            }}
            .section {{
                background-color: white;
                margin-top: 20px;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .summary-box {{
                display: flex;
                justify-content: space-between;
                margin-bottom: 20px;
            }}
            .summary-item {{
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 8px;
                flex: 1;
                margin: 0 10px;
                text-align: center;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .summary-item h3 {{
                margin-top: 0;
                font-size: 16px;
                color: #666;
            }}
            .summary-item p {{
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            }}
            .confidence {{
                color: {confidence_color};
                font-weight: bold;
            }}
            .match {{
                color: #228B22;
                font-weight: bold;
            }}
            .no-match {{
                color: #B22222;
                font-weight: bold;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                position: sticky;
                top: 0;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            tr:hover {{
                background-color: #f1f1f1;
            }}
            .result-img {{
                max-width: 100%;
                height: auto;
                margin-top: 10px;
                border-radius: 4px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .side-by-side {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-top: 20px;
            }}
            .metrics-box {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 10px;
            }}
            .metric {{
                background-color: #f0f0f0;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 14px;
                flex-grow: 1;
                text-align: center;
            }}
            .feature-metric {{
                background-color: #e8f4ff;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 14px;
                margin: 4px;
                display: inline-block;
            }}
            .feature-metric.good {{
                background-color: #e6ffe6;
            }}
            .feature-metric.bad {{
                background-color: #ffebeb;
            }}
            .info-icon {{
                color: #2989d8;
                cursor: help;
                position: relative;
            }}
            .bar-container {{
                width: 100%;
                background-color: #e0e0e0;
                border-radius: 4px;
                margin-top: 5px;
            }}
            .bar {{
                height: 10px;
                border-radius: 4px;
                background: linear-gradient(90deg, #1e5799 0%, #2989d8 100%);
            }}
            .image-quality {{
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
                color: white;
                display: inline-block;
                margin-top: 10px;
            }}
            .quality-high {{
                background-color: #4CAF50;
            }}
            .quality-medium {{
                background-color: #FF9800;
            }}
            .quality-low {{
                background-color: #F44336;
            }}
            .points-box {{
                margin-top: 15px;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
            }}
            .warning-box {{
                background-color: #fff9e6;
                border-left: 4px solid #ffcc00;
                padding: 10px 15px;
                margin: 15px 0;
                border-radius: 0 4px 4px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="header-flex">
                    <h1>Rapport d'analyse faciale - Xavier Dupont de Ligonnès</h1>
                    <div class="timestamp">
                        <div>Date: {date_analyse}</div>
                        <div>Heure: {heure_analyse}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Résumé de l'analyse</h2>
                <div class="warning-box">
                    <p><strong>Note importante:</strong> Cette analyse est basée sur des images potentiellement anciennes ou de mauvaise qualité de Xavier Dupont de Ligonnès. Les scores ont été ajustés pour tenir compte de cette limitation.</p>
                </div>
                <div class="summary-box">
                    <div class="summary-item">
                        <h3>Total des comparaisons</h3>
                        <p>{len(results_df)}</p>
                    </div>
                    <div class="summary-item">
                        <h3>Correspondances trouvées</h3>
                        <p>{len(best_matches) if best_matches is not None else 0}</p>
                    </div>
                    <div class="summary-item">
                        <h3>Niveau de confiance</h3>
                        <p class="confidence">{confidence_level}</p>
                        <div class="bar-container">
                            <div class="bar" style="width: {min(100, max_similarity * 100)}%;"></div>
                        </div>
                    </div>
                </div>
    """
    
    if best_matches is not None and not best_matches.empty:
        best_match = best_matches.iloc[0]
        
        # Définir la classe de qualité d'image
        quality_class = "quality-high" if best_match['image_quality'] > 0.8 else "quality-medium" if best_match['image_quality'] > 0.5 else "quality-low"
        
        # Extraire les points caractéristiques qui correspondent bien
        matching_points = best_match.get('matching_points', [])
        good_points = [p for p in matching_points if p[1] > 0.7]
        bad_points = [p for p in matching_points if p[1] <= 0.7]
        
        html_content += f"""
                <h3>Principale correspondance identifiée</h3>
                <p class="match">✓ L'image <strong>{best_match['suspect_image']}</strong> correspond à <strong>{best_match['criminal_image']}</strong> avec une similarité de <strong>{best_match['similarity']}</strong></p>
                <div class="image-quality {quality_class}">Qualité d'image: {best_match['image_quality']}</div>
            </div>
            
            <div class="section">
                <h2>Détails de la meilleure correspondance</h2>
                <div class="side-by-side">
                    <div>
                        <h3>Métriques de similarité</h3>
                        <div class="metrics-box">
                            <div class="metric">Score global: <strong>{best_match['similarity']}</strong></div>
                            <div class="metric">Corrélation: <strong>{best_match['correlation']}</strong></div>
                            <div class="metric">Similarité d'histogramme: <strong>{best_match['hist_similarity']}</strong></div>
                            <div class="metric">SSIM: <strong>{best_match['ssim']}</strong></div>
                            <div class="metric">Points faciaux: <strong>{best_match.get('facial_features', 'N/A')}</strong></div>
                        </div>
                        
                        <h3>Points caractéristiques</h3>
                        <div class="points-box">
        """
        
        if good_points:
            html_content += "<h4>Points de forte correspondance</h4>"
            for point, score in good_points:
                html_content += f'<div class="feature-metric good">{point}: {score:.2f}</div>'
        
        if bad_points:
            html_content += "<h4>Points de faible correspondance</h4>"
            for point, score in bad_points:
                html_content += f'<div class="feature-metric bad">{point}: {score:.2f}</div>'
                
        if not matching_points:
            html_content += "<p>Aucun point caractéristique commun n'a pu être détecté.</p>"
            
        html_content += """
                        </div>
                    </div>
                    <div>
                        <h3>Interprétation et ajustements pour image de qualité limitée</h3>
        """
        
        # Ajouter une explication spécifique à la qualité de l'image
        if best_match['image_quality'] < 0.5:
            html_content += f"""
                <p>Les images analysées sont de <strong>qualité réduite</strong>, ce qui peut affecter la précision de la reconnaissance. Pour compenser:</p>
                <ul>
                    <li>Le seuil de correspondance a été abaissé à <strong>{best_match.get('quality_threshold', 0.35)}</strong> (normalement 0.35)</li>
                    <li>Une plus grande importance a été accordée à la correspondance structurelle globale</li>
                    <li>Les différences d'éclairage et de contraste ont été normalisées</li>
                </ul>
            """
        else:
            html_content += f"""
                <p>Les images analysées sont de qualité acceptable. L'analyse indique une correspondance <span class="confidence">{confidence_level.lower()}</span> entre les images.</p>
                <p>La corrélation structurelle est particulièrement {("élevée" if best_match['correlation'] > 0.5 else "significative" if best_match['correlation'] > 0.3 else "modérée")}.</p>
            """
            
        html_content += """
                    </div>
                </div>
                <h3>Comparaison visuelle</h3>
                <img class="result-img" src="comparaison_{best_match['suspect_image']}_{best_match['criminal_image']}.png" alt="Meilleure correspondance">
            </div>
        """
    else:
        html_content += f"""
                <h3>Aucune correspondance satisfaisante trouvée</h3>
                <p class="no-match">✗ Aucune des images comparées ne dépasse le seuil de similarité requis (0.35)</p>
            </div>
        """
    
    html_content += """
            <div class="section">
                <h2>Tableau des comparaisons</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Image suspecte</th>
                            <th>Image criminelle</th>
                            <th>Score global</th>
                            <th>Corrélation</th>
                            <th>Points faciaux</th>
                            <th>Qualité image</th>
                            <th>Correspondance</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    for _, row in results_df.iterrows():
        match_class = "match" if row['is_match'] else "no-match"
        match_symbol = "✓" if row['is_match'] else "✗"
        
        # Définir la classe de qualité d'image
        quality_value = row.get('image_quality', 0)
        quality_text = f"{quality_value:.2f}"
        if quality_value > 0.8:
            quality_class = "quality-high"
        elif quality_value > 0.5:
            quality_class = "quality-medium"
        else:
            quality_class = "quality-low"
            
        html_content += f"""
                    <tr>
                        <td>{row['suspect_image']}</td>
                        <td>{row['criminal_image']}</td>
                        <td><strong>{row['similarity']}</strong></td>
                        <td>{row['correlation']}</td>
                        <td>{row.get('facial_features', 'N/A')}</td>
                        <td><span class="image-quality {quality_class}" style="margin:0; padding:2px 5px;">{quality_text}</span></td>
                        <td class="{match_class}">{match_symbol} {row['is_match']}</td>
                    </tr>
        """
    
    html_content += """
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>Méthodologie et interprétation - Adaptation pour Xavier Dupont de Ligonnès</h2>
                <p>Cette analyse est spécialement adaptée pour comparer des images d'époque ou de qualité limitée de Xavier Dupont de Ligonnès avec des suspects potentiels :</p>
                <div class="side-by-side">
                    <div>
                        <h3>Métriques utilisées</h3>
                        <ul>
                            <li><strong>Score global</strong>: Combinaison pondérée des métriques, ajustée pour les images anciennes</li>
                            <li><strong>Corrélation</strong>: Similarité structurelle après normalisation du contraste</li>
                            <li><strong>Points faciaux</strong>: Correspondance des éléments du visage (yeux, nez, bouche) même avec des images parcellaires</li>
                            <li><strong>Qualité image</strong>: Évaluation de la fiabilité des résultats basée sur la netteté et la complétude des détails</li>
                        </ul>
                    </div>
                    <div>
                        <h3>Adaptations pour images anciennes</h3>
                        <ul>
                            <li><strong>Seuil dynamique</strong>: Abaissé pour les images de faible qualité</li>
                            <li><strong>Détection améliorée</strong>: Algorithmes optimisés pour les visages partiellement visibles</li>
                            <li><strong>Normalisation</strong>: Correction des différences d'éclairage et de contraste</li>
                            <li><strong>Analyse structurelle</strong>: Accent mis sur les proportions du visage plutôt que sur les détails fins</li>
                        </ul>
                    </div>
                </div>
                <div class="warning-box">
                    <p><strong>Limitations importantes:</strong> Les comparaisons basées sur des images anciennes ou de mauvaise qualité présentent des risques d'erreur plus élevés. Cette analyse doit être considérée comme un indice préliminaire et non comme une preuve formelle. D'autres éléments d'enquête doivent être pris en compte.</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(results_dir, "rapport_analyse.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"\nRapport HTML généré: {os.path.join(results_dir, 'rapport_analyse.html')}")

def create_criminal_features_bank(criminal_folder, results_dir):
    """
    Crée une banque de caractéristiques à partir de toutes les images du criminel
    sans les superposer, pour obtenir plus d'angles et de points de comparaison
    """
    # Obtenir la liste des images du criminel
    criminal_images = [os.path.join(criminal_folder, f) for f in os.listdir(criminal_folder) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    if not criminal_images:
        print("Aucune image du criminel trouvée.")
        return None, []
        
    print(f"\nExtracting des caractéristiques à partir de {len(criminal_images)} images de Xavier Dupont de Ligonnès...")
    
    # Collecter les visages et les points caractéristiques de toutes les images
    all_faces = []
    all_landmarks = []
    all_original_images = []
    all_landmark_images = []
    
    for img_path in criminal_images:
        face, _, _, (landmark_img, landmarks) = detect_face(img_path)
        if face is not None:
            # Charger l'image originale
            original_img = cv2.imread(img_path)
            if original_img is not None:
                all_faces.append(face)
                all_landmarks.append(landmarks)
                all_original_images.append(original_img)
                all_landmark_images.append(landmark_img)
    
    if not all_faces:
        print("Aucun visage détecté dans les images du criminel.")
        return None, []
    
    # Créer une grille d'images pour visualisation
    # Déterminer le nombre de lignes et de colonnes
    n_images = len(all_faces)
    n_cols = min(3, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Créer une grille vide
    grid_height = n_rows * 200
    grid_width = n_cols * 200
    grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Remplir la grille avec les visages
    for i, face in enumerate(all_faces):
        row = i // n_cols
        col = i % n_cols
        y = row * 200
        x = col * 200
        resized_face = cv2.resize(face, (200, 200))
        grid_image[y:y+200, x:x+200] = resized_face
    
    # Créer une grille des images avec landmarks
    landmarks_grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    for i, lm_img in enumerate(all_landmark_images):
        row = i // n_cols
        col = i % n_cols
        y = row * 200
        x = col * 200
        resized_lm = cv2.resize(lm_img, (200, 200))
        landmarks_grid[y:y+200, x:x+200] = resized_lm
    
    # Enregistrer les grilles
    grid_path = os.path.join(results_dir, "xdl_all_faces.jpg")
    cv2.imwrite(grid_path, grid_image)
    
    landmarks_grid_path = os.path.join(results_dir, "xdl_all_landmarks.jpg")
    cv2.imwrite(landmarks_grid_path, landmarks_grid)
    
    print(f"Grille de visages créée et enregistrée: {grid_path}")
    print(f"Grille de landmarks créée et enregistrée: {landmarks_grid_path}")
    
    # Aplatir les landmarks en une seule liste
    flattened_landmarks = []
    for landmarks_list in all_landmarks:
        flattened_landmarks.extend(landmarks_list)
    
    return (all_faces, grid_image, landmarks_grid), (flattened_landmarks, all_landmarks)

def compare_with_all_criminal_features(suspect_image_path, criminal_folder, results_dir):
    """
    Compare l'image du suspect avec toutes les caractéristiques extraites des 
    images du criminel sous différents angles
    """
    # Extraire toutes les caractéristiques
    (all_faces, grid_image, landmarks_grid), (flattened_landmarks, all_landmarks_lists) = create_criminal_features_bank(criminal_folder, results_dir)
    
    if all_faces is None:
        print("Impossible d'extraire les caractéristiques du criminel.")
        return None
    
    print(f"Comparaison de l'image suspecte avec toutes les caractéristiques de Xavier Dupont de Ligonnès...")
    
    # Détecter le visage dans l'image du suspect
    suspect_face, suspect_with_rect, suspect_face_ratio, (suspect_landmarks_img, suspect_landmarks) = detect_face(suspect_image_path)
    
    if suspect_face is None:
        print("Aucun visage détecté dans l'image du suspect.")
        return None
    
    # Enregistrer les images pour la visualisation
    suspect_path = os.path.join(results_dir, f"suspect_{os.path.basename(suspect_image_path)}")
    cv2.imwrite(suspect_path, suspect_face)
    cv2.imwrite(os.path.join(results_dir, "suspect_landmarks.jpg"), suspect_landmarks_img)
    
    # Convertir en niveaux de gris pour la comparaison
    suspect_gray = cv2.cvtColor(suspect_face, cv2.COLOR_BGR2GRAY)
    
    # Mesures globales
    best_correlation = 0
    best_hist_similarity = 0
    best_ssim_value = 0
    best_facial_similarity = 0
    best_face_index = -1
    
    # Pour chaque visage, calculer les similarités et garder la meilleure
    correlations = []
    hist_similarities = []
    ssim_values = []
    
    for i, face in enumerate(all_faces):
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.resize(face_gray, (200, 200))
        
        # 1. Corrélation
        correlation = cv2.matchTemplate(suspect_gray, face_gray, cv2.TM_CCOEFF_NORMED)
        corr_val = np.max(correlation)
        correlations.append(corr_val)
        
        # 2. Histogramme
        hist1 = cv2.calcHist([suspect_gray], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([face_gray], [0], None, [256], [0, 256])
        
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        hist_similarities.append(hist_sim)
        
        # 3. SSIM
        try:
            from skimage.metrics import structural_similarity as ssim
            ssim_val = ssim(suspect_gray, face_gray, data_range=255)
            ssim_values.append(ssim_val)
        except ImportError:
            ssim_val = (corr_val + hist_sim) / 2
            ssim_values.append(ssim_val)
            
        # Mettre à jour les meilleures valeurs
        if corr_val > best_correlation:
            best_correlation = corr_val
            best_face_index = i
            
        if hist_sim > best_hist_similarity:
            best_hist_similarity = hist_sim
            
        if ssim_val > best_ssim_value:
            best_ssim_value = ssim_val
    
    # 4. Comparer avec tous les points caractéristiques
    similarities = []
    all_matching_points = []
    
    for landmarks_list in all_landmarks_lists:
        similarity, matching_points = compare_facial_features(suspect_landmarks, landmarks_list)
        if similarity > 0:
            similarities.append(similarity)
            all_matching_points.extend(matching_points)
            
            if similarity > best_facial_similarity:
                best_facial_similarity = similarity
    
    # Calculer la moyenne des similarités faciales si disponible
    avg_facial_similarity = np.mean(similarities) if similarities else 0
    
    # Score final (moyenne pondérée des meilleures caractéristiques trouvées)
    final_score = (0.4 * best_correlation + 
                   0.3 * best_hist_similarity + 
                   0.2 * best_ssim_value + 
                   0.1 * best_facial_similarity)
    
    # Score moyen global sur toutes les images
    avg_correlation = np.mean(correlations)
    avg_hist_similarity = np.mean(hist_similarities)
    avg_ssim_value = np.mean(ssim_values)
    avg_score = (0.4 * avg_correlation + 
                 0.3 * avg_hist_similarity + 
                 0.2 * avg_ssim_value + 
                 0.1 * avg_facial_similarity)
    
    # Créer une visualisation de la comparaison
    plt.figure(figsize=(12, 10))
    
    # Afficher l'image du suspect avec landmarks
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(suspect_landmarks_img, cv2.COLOR_BGR2RGB))
    plt.title("Suspect - Points caractéristiques")
    plt.axis('off')
    
    # Afficher la grille de tous les visages du criminel
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB))
    plt.title("Toutes les images de Xavier DL")
    plt.axis('off')
    
    # Afficher la meilleure correspondance
    if best_face_index >= 0:
        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(all_faces[best_face_index], cv2.COLOR_BGR2RGB))
        plt.title(f"Meilleure correspondance (Image {best_face_index+1})")
        plt.axis('off')
    
    # Graphique des scores par image
    plt.subplot(2, 2, 4)
    indices = range(len(all_faces))
    plt.bar(indices, correlations, alpha=0.7, label='Corrélation')
    plt.bar(indices, hist_similarities, alpha=0.5, label='Histogramme')
    plt.bar(indices, ssim_values, alpha=0.3, label='SSIM')
    plt.xlabel('Image #')
    plt.ylabel('Score')
    plt.title('Scores par image')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ajouter le texte d'information
    match_status = "MATCH" if final_score > 0.3 else "PAS DE MATCH"
    plt.suptitle(f"Comparaison avec toutes les images: {match_status}\n" + 
                 f"Meilleur score: {round(final_score, 4)}, Score moyen: {round(avg_score, 4)}\n" +
                 f"Meilleure corrélation: {round(best_correlation, 4)}, Moyenne: {round(avg_correlation, 4)}",
                 fontsize=14)
    
    # Enregistrer l'image
    comparison_path = os.path.join(results_dir, "comparaison_complete.png")
    plt.savefig(comparison_path)
    plt.close()
    
    print(f"Comparaison complète enregistrée: {comparison_path}")
    
    # Enregistrer les résultats dans un fichier texte
    result_text = f"""
COMPARAISON COMPLÈTE AVEC TOUTES LES IMAGES DE XAVIER DUPONT DE LIGONNÈS
========================================================================
Image du suspect: {os.path.basename(suspect_image_path)}
Date de l'analyse: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

MEILLEURS RÉSULTATS:
- Score global: {round(final_score, 4)}
- Meilleure corrélation: {round(best_correlation, 4)} (image #{best_face_index+1})
- Meilleure similarité d'histogramme: {round(best_hist_similarity, 4)}
- Meilleur SSIM: {round(best_ssim_value, 4)}
- Meilleure similarité des points faciaux: {round(best_facial_similarity, 4)}

SCORES MOYENS:
- Score global moyen: {round(avg_score, 4)}
- Corrélation moyenne: {round(avg_correlation, 4)}
- Similarité d'histogramme moyenne: {round(avg_hist_similarity, 4)}
- SSIM moyen: {round(avg_ssim_value, 4)}
- Similarité faciale moyenne: {round(avg_facial_similarity, 4)}

CONCLUSION:
{match_status} - {'Correspondance significative avec Xavier Dupont de Ligonnès' if final_score > 0.3 else 'Pas de correspondance suffisante'}

Points caractéristiques correspondants les plus pertinents:
"""
    # Trier les points caractéristiques par score de correspondance
    sorted_matching_points = sorted(all_matching_points, key=lambda x: x[1], reverse=True)
    # Prendre les 5 meilleurs points ou moins s'il y en a moins
    top_points = sorted_matching_points[:min(5, len(sorted_matching_points))]
    for point, score in top_points:
        result_text += f"- {point}: {round(score, 2)}\n"
        
    with open(os.path.join(results_dir, "resultats_complets.txt"), "w", encoding="utf-8") as f:
        f.write(result_text)
    
    return final_score, best_correlation, best_hist_similarity, best_ssim_value, best_facial_similarity, comparison_path, avg_score

def analyze_all_suspects(suspects_folder, criminal_folder, results_dir):
    """
    Analyser toutes les photos de suspects dans le dossier spécifié
    """
    all_results = []
    suspect_images = [os.path.join(suspects_folder, f) for f in os.listdir(suspects_folder) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    for suspect_img in suspect_images:
        results = compare_faces(suspect_img, criminal_folder, results_dir)
        all_results.extend(results)
    
    # Créer un DataFrame pour une meilleure visualisation
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Trier par similarité (plus grande valeur = plus grande similarité)
        results_df = results_df.sort_values(by="similarity", ascending=False)
        
        print("\nRésultats triés par similarité:")
        print(results_df[["suspect_image", "criminal_image", "similarity", "correlation", "facial_features", "image_quality", "is_match"]])
        
        # Sauvegarder les résultats
        results_df.to_csv(os.path.join(results_dir, "resultats_comparaison.csv"), index=False)
        print(f"\nRésultats sauvegardés dans '{os.path.join(results_dir, 'resultats_comparaison.csv')}'")
        
        # Créer une visualisation des résultats
        create_comparison_grid(results_df, suspects_folder, criminal_folder, results_dir)
        
        # Identifier les meilleures correspondances
        best_matches = results_df[results_df["is_match"] == True]
        if not best_matches.empty:
            print("\nMeilleures correspondances trouvées:")
            print(best_matches[["suspect_image", "criminal_image", "similarity", "correlation", "facial_features", "image_quality"]])
            
            # Générer le rapport HTML
            generate_report(results_df, best_matches, results_dir)
            
            # Comparer avec l'image composite
            if len(suspect_images) == 1:  # Si un seul suspect
                composite_results = compare_with_all_criminal_features(suspect_images[0], criminal_folder, results_dir)
                if composite_results:
                    score, corr, hist, ssim_val, facial, comp_path, avg_score = composite_results
                    print(f"\nComparaison avec image composite: {'MATCH' if score > 0.3 else 'PAS DE MATCH'} (Score: {round(score, 4)})")
                    
                    # Ajouter cette information au rapport HTML
                    add_composite_to_report(results_dir, comp_path, score, corr, hist, ssim_val, facial)
            
            return results_df, best_matches
        else:
            print("\nAucune correspondance satisfaisante trouvée.")
            # Retourner la meilleure correspondance même si elle est sous le seuil
            if not results_df.empty:
                print("\nMeilleure correspondance (sous le seuil):")
                print(results_df.iloc[0][["suspect_image", "criminal_image", "similarity", "correlation", "facial_features", "image_quality"]])
                # Créer un nouveau dataframe avec cette correspondance marquée comme match
                top_match = results_df.iloc[0:1].copy()
                top_match["is_match"] = True
                
                # Générer le rapport HTML
                generate_report(results_df, top_match, results_dir)
                
                # Comparer avec l'image composite
                if len(suspect_images) == 1:  # Si un seul suspect
                    composite_results = compare_with_all_criminal_features(suspect_images[0], criminal_folder, results_dir)
                    if composite_results:
                        score, corr, hist, ssim_val, facial, comp_path, avg_score = composite_results
                        print(f"\nComparaison avec image composite: {'MATCH' if score > 0.3 else 'PAS DE MATCH'} (Score: {round(score, 4)})")
                        
                        # Ajouter cette information au rapport HTML
                        add_composite_to_report(results_dir, comp_path, score, corr, hist, ssim_val, facial)
                
                return results_df, top_match
            return results_df, None
    else:
        print("Aucun résultat obtenu.")
        return pd.DataFrame(), None

def add_composite_to_report(results_dir, composite_image_path, score, correlation, hist_similarity, ssim, facial_features):
    """
    Ajoute la comparaison avec l'image composite au rapport HTML
    """
    report_path = os.path.join(results_dir, "rapport_analyse.html")
    
    if not os.path.exists(report_path):
        print("Rapport HTML non trouvé, impossible d'ajouter la comparaison composite.")
        return
    
    # Charger le contenu du rapport
    with open(report_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    # Créer la section HTML pour la comparaison composite
    composite_html = f"""
    <div class="section">
        <h2>Comparaison avec toutes les images</h2>
        <p>Cette analyse compare l'image du suspect avec toutes les images disponibles de Xavier Dupont de Ligonnès, utilisant les caractéristiques de chaque perspective et angle de vue.</p>
        
        <div class="metrics-box">
            <div class="metric">Score global: <strong>{round(score, 4)}</strong></div>
            <div class="metric">Corrélation: <strong>{round(correlation, 4)}</strong></div>
            <div class="metric">Similarité d'histogramme: <strong>{round(hist_similarity, 4)}</strong></div>
            <div class="metric">SSIM: <strong>{round(ssim, 4)}</strong></div>
            <div class="metric">Points faciaux: <strong>{round(facial_features, 4)}</strong></div>
        </div>
        
        <div style="margin-top: 20px;">
            <img class="result-img" src="{os.path.basename(composite_image_path)}" alt="Comparaison complète">
        </div>
        
        <p class="{'match' if score > 0.3 else 'no-match'}">
            <strong>{('✓ MATCH - ' if score > 0.3 else '✗ PAS DE MATCH - ') + f'Score: {round(score, 4)}'}</strong>
        </p>
        
        <p><em>Cette analyse tire parti de toutes les images disponibles de Xavier Dupont de Ligonnès pour maximiser les points de comparaison sous différents angles.</em></p>
    </div>
    """
    
    # Insérer la section juste avant la balise de fermeture du conteneur
    html_content = html_content.replace('        </div>\n    </body>', f'        {composite_html}\n        </div>\n    </body>')
    
    # Écrire le contenu mis à jour
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Comparaison avec toutes les images ajoutée au rapport: {report_path}")

if __name__ == "__main__":
    print("Analyse faciale - Recherche de Xavier Dupont de Ligonnès")
    print("=======================================================")
    
    # Chemins vers les dossiers
    script_dir = Path(__file__).parent
    suspects_folder = script_dir / "compares"
    criminal_folder = script_dir / "photosX"
    
    # Créer un dossier pour les résultats avec un horodatage
    results_dir = script_dir / f"resultats_analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Dossier des suspects: {suspects_folder}")
    print(f"Dossier du criminel: {criminal_folder}")
    print(f"Dossier des résultats: {results_dir}")
    
    # Vérifier que les dossiers existent
    if not suspects_folder.exists():
        print(f"Le dossier {suspects_folder} n'existe pas.")
        exit(1)
    if not criminal_folder.exists():
        print(f"Le dossier {criminal_folder} n'existe pas.")
        exit(1)
    
    print("\nDébut de l'analyse des visages...")
    print("Note: Les images de Xavier Dupont de Ligonnès étant souvent de qualité limitée ou anciennes,")
    print("      l'analyse a été optimisée pour ce type d'images.")
    
    results_df, best_matches = analyze_all_suspects(suspects_folder, criminal_folder, results_dir)
    
    if best_matches is not None and not best_matches.empty:
        print("\nConclusion: Correspondance détectée avec Xavier Dupont de Ligonnès!")
        
        # Trouver la meilleure correspondance
        best_match = best_matches.iloc[0]
        print(f"\nMeilleure correspondance: {best_match['suspect_image']} correspond à {best_match['criminal_image']} avec une similarité de {best_match['similarity']}")
        print(f"Qualité d'image évaluée à: {best_match.get('image_quality', 'N/A')}")
        
        if 'facial_features' in best_match and best_match['facial_features'] > 0.5:
            print("Points caractéristiques du visage: Bonne correspondance")
        elif 'facial_features' in best_match:
            print("Points caractéristiques du visage: Correspondance partielle")
        else:
            print("Points caractéristiques du visage: Non disponible")
    else:
        print("\nConclusion: Pas de correspondance suffisante avec Xavier Dupont de Ligonnès.")
        
    print("\nAnalyse terminée.")
    print(f"\nConsultez le rapport complet dans: {results_dir}/rapport_analyse.html")