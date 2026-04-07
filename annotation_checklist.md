## Per-Image Checklist

- [ ] All visible threat objects have bounding boxes
- [ ] Boxes are tight around objects (not too loose)
- [ ] Class labels are correct
- [ ] No overlapping boxes that should be separate
- [ ] Boxes follow object boundaries (not square when object is long)

## Per-Dataset Checklist

- [ ] All images in train/ have corresponding .txt files
- [ ] All images in val/ have corresponding .txt files
- [ ] No empty .txt files (unless truly no objects)
- [ ] All class IDs match classes.txt
- [ ] Coordinates are normalized (between 0 and 1)

## Quick QA Commands

Run full QA after each annotation batch:

```powershell
.\annotate_check.ps1 -Split train
.\annotate_check.ps1 -Split val
```

Run without visualization (faster):

```powershell
.\annotate_check.ps1 -Split train -SkipVisual
```
