# Training Images

Place your Batman reference images here before running `lora_train.py`.

## How many images
15-20 is the sweet spot. Fewer than 10 and the model won't generalise well; more than 40 and you risk overfitting.

## What images to use
- **Pick one consistent style** — e.g. all from the same comic run, all from the same film, or all from the same animated series. Mixing styles confuses the model.
- **Variety of poses and angles** — front, side, action shots, close-ups. Don't use 20 near-identical images.
- **Clean subjects** — avoid images where Batman is tiny in the frame, heavily occluded, or blurry.
- **Consistent costume** — if you want the yellow-belt classic look, use images with that. If you want the dark knight armoured look, stick to that.

## Naming
Name them anything — `01.jpg`, `batman_01.png`, etc. The script finds all images in this folder automatically.

## Quick sources
- Google Images (search "batman comic panel" or "batman dark knight" — filter by image size > 512px)
- [DC Comics covers](https://www.dccomics.com/comics) 
- Movie stills from a single film

## Then run training
```bash
cd instance-segmentation/yolov8
source venv/bin/activate
python3 lora_train.py --instance_data_dir training_data/ --output_dir lora_weights/
```
