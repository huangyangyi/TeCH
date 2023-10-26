import replicate
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import numpy as np
from PIL import Image
import argparse
import os

mylabel2ids = {
    'hat': [1],
    'sunglasses': [3],
    'upper-clothes': [4],
    #'hair': [2],
    'skirt': [5],
    'pants': [6],
    'dress': [7],
    'belt': [8],
    'shoes': [9, 10],
    'bag': [16],
    'scarf': [17]
}



def is_necessary(g, garments):
    if 'dress' not in garments:
        if g == 'upper-clothes':
            return True
        if (g == 'pants') and ('skirt' not in g):
            return True
        if (g == 'skirt') and ('pants' not in g):
            return True
    return False

def get_prompt_segments(img_path, feature_extractor, model):
    image = open(img_path, 'rb')
    def ask(q):
        print('Question: {}'.format(q))
        answer = replicate.run(
            "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746",
            input={"image": image, 
            "task": "visual_question_answering",
            "question":q}
        ).replace('Answer: ', '')
        print('Answer:', answer)
        return answer
    def clean_prompt(prompt):
        while ('  ' in prompt):
            prompt = prompt.replace('  ', ' ')
        while (' ,' in prompt):
            prompt = prompt.replace(' ,', ',')
        return prompt
    gender = ask('Is this person a man or a woman')
    if 'woman' in gender:
        gender = 'woman'
    elif 'man' in gender:
        gender = 'man'
    else:
        gender = 'person'
    prompt = 'a sks {}'.format(gender)
    garments = get_garments(image, feature_extractor, model)
    haircolor = ask('What is the hair color of this person?')
    hairstyle = ask('What is the hair style of this person?')
    face = ask('Describe the facial appearance of this person.')
    prompt = prompt + ', {} {} hair, {}'.format(haircolor, hairstyle, face)
    for g in garments:
        has_g = is_necessary(g, garments) or ('yes' in ask('Is this person wearing {}?'.format(g)))
        if has_g:
            kind = ask('What {} is the person wearing?'.format(g))
            if (g in kind) or (g == 'upper-clothes'):
                g = ''
            color = ask('What is the color of the {} {}?'.format(kind, g))
            style = ask('What is the style of the {} {}?'.format(kind, g))
            if style in kind or style in g:
                style = ''
            if color in kind or color in g:
                color = ''
            prompt = prompt + ', sks {} {} {} {}'.format(color, style, kind, g)
    has_beard = ask('Do this person has facial hair?')
    if 'yes' in has_beard:
        beard = ask('How is the facial hair of this person?')
        if beard != 'none':
            prompt = prompt + ', {} beard'.format(beard)
    pose = ask('Describe the pose of this person.')
    prompt = prompt + ', {}'.format(pose)
    prompt = clean_prompt(prompt)
    return prompt, gender


def get_garments(img_path, feature_extractor, model):
    image = np.array(Image.open(img_path))
    alpha = image[..., 3:] > 250
    image = (image[..., :3] * alpha).astype(np.uint8)
    inputs = feature_extractor(images=image, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    image = np.array(Image.open(img_path).resize((128, 128)))
    alpha = image[..., 3:] > 250
    image = image[..., :3] * alpha
    seg = outputs.logits[0].argmax(dim=0)
    result = dict()
    for label in mylabel2ids:
        label_mask = np.zeros_like(alpha[..., 0])
        for id in mylabel2ids[label]:
            label_mask |= (seg == id).cpu().numpy()
        label_mask &= alpha[..., 0]
        #print(label, label_mask.sum())
        if label_mask.sum() == 0:
            continue
        result[label] = label_mask
    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str, required=True, help="input image")
    parser.add_argument('--out-path', type=str, required=True, help="output path")
    opt = parser.parse_args()
    print(f'[INFO] Generating text prompt for {opt.img_path}...')
    model = SegformerForSemanticSegmentation.from_pretrained("matei-dorian/segformer-b5-finetuned-human-parsing").cuda()
    feature_extractor = SegformerFeatureExtractor.from_pretrained("matei-dorian/segformer-b5-finetuned-human-parsing")
    prompt, gender = get_prompt_segments(opt.img_path, feature_extractor, model)
    print(f'[INFO] generated prompt: {prompt}, estimated category: {gender}')
    with open(opt.out_path, 'w') as f:
        f.write(f'{prompt}|{gender}')