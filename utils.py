import torch
import torchvision.transforms as transforms
import PIL as pl


def has_intersection(carr1, carr2):
    """
    Regarde si carr1 et carr2 ont une partie commune
    carr1 et carr2 sont au format ((top,left),taille) 
    """
    if (carr1[0][0]<carr2[0][0] and carr1[0][0]+carr1[1]>carr2[0][0]) and (carr1[0][1]<carr2[0][1] and carr1[0][1]+carr1[1]>carr2[0][1]):
        return True
    return False

def repear_faces(model, img, min_size, max_size, step_square, step_size):
    """
    Repère les visages  d'une image en utilisant un modèle pytorch 
    et l'algorithme cascade
    min_size et max_size sont les tailles min et maximales des visages que l'on veut
    repérer
    step_size est le pas pour la fenêtre de l'algorithme
    step_square est le pas pour la detecion
    """
    faces_coords = []
    size = min_size
    def add_face(coords, step_size):
        for face in faces_coords:
            if has_intersection(face, (coords, size)) or has_intersection((coords, size), face):
                return
        faces_coords.append((coords,step_size))
        
    while size < max_size:
        for i in range(0,img.shape[0], step_square):
            for j in range(0,img.shape[0], step_square):
                tile = img[i:i+size,j:j+size]
                im = pl.Image.fromarray(tile).convert("L")
                inp = transforms.Resize(224,224,3)(im).unsqueeze(0)
                out = model(inp).argmax(1)
                if out == torch.tensor(1):
                    add_face((i,j),size)
        size += step_size
    return faces_coords