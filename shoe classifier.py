#hide
! [ -e /content ] && pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

#hide
from fastbook import *
from fastai.vision.widgets import *

key = os.environ.get('AZURE_SEARCH_KEY', '82ce3bfa143c4892b6e1d4f27189fa70')

search_images_bing

results = search_images_bing(key, 'boots')
ims = results.attrgot('contentUrl')
len(ims)

#hide
ims = ['https://m.media-amazon.com/images/I/81aw6vvX5EL._AC_SL1500_.jpg']

dest = 'images/shoes.jpg'
download_url(ims[0], dest)

im = Image.open(dest)
im.to_thumb(128,128)

shoe_types = 'boots','crocs','ballerina flats'
path = Path('shoes')

if not path.exists():
    path.mkdir()
    for o in shoe_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(key, f'{o}')
        download_images(dest, urls=results.attrgot('contentUrl'))

fns = get_image_files(path)
fns

failed = verify_images(fns)
failed

failed.map(Path.unlink);

shoes = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))

dls = shoes.dataloaders(path)

dls.valid.show_batch(max_n=4, nrows=1)

shoes = shoes.new(item_tfms=Resize(128, ResizeMethod.Squish))
dls = shoes.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)

shoes = shoes.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))
dls = shoes.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)

shoes = shoes.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
dls = shoes.dataloaders(path)
dls.train.show_batch(max_n=4, nrows=1, unique=True)

shoes = shoes.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = shoes.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)

shoes = shoes.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = shoes.dataloaders(path)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

interp.plot_top_losses(5, nrows=1)

cleaner = ImageClassifierCleaner(learn)
cleaner

#hide
for idx in cleaner.delete(): cleaner.fns[idx].unlink()
for idx,shoe in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/shoe)

learn.export()

path = Path()
path.ls(file_exts='.pkl')

learn_inf = load_learner(path/'export.pkl')

learn_inf.predict('images/shoes.jpg')

learn_inf.dls.vocab

img = PILImage.create(upload.data[-1])

import ipywidgets as widgets

from ipywidgets import FileUpload
upload = FileUpload(accept='.jpg', multiple=False)
upload

out_pl = widgets.Output()
out_pl.clear_output()
with out_pl: display(img.to_thumb(128,128))
out_pl

pred,pred_idx,probs = learn_inf.predict(img)

lbl_pred = widgets.Label()
lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
lbl_pred

classify = widgets.Button(
    description='Classify',
    disabled=False,
    button_style='',
    tooltip='Classify',
    icon='check'
    )

def on_click_classify(change):
    img = PILImage.create(upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
    
classify
classify.on_click(on_click_classify)


VBox([widgets.Label('Select your shoe!'), 
      upload, classify, out_pl, lbl_pred])

!pip install ipywidgets
!jupyter nbextension enable --py widgetsnbextension

#hide
!pip install voila
!jupyter serverextension enable --sys-prefix voila 
