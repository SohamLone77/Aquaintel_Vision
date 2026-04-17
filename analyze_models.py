import csv, pathlib, json

log_dir = pathlib.Path("logs/csv")
reg_path = pathlib.Path("results/model_registry.json")

registry = {}
if reg_path.exists():
    registry = json.loads(reg_path.read_text("utf-8"))

results = []

for f in sorted(log_dir.glob("*_training.csv")):
    run = f.stem.replace("_training", "")
    rows = list(csv.DictReader(f.read_text("utf-8").splitlines()))
    if not rows:
        continue

    def g(row, k):
        try:
            return float(row[k])
        except Exception:
            return None

    val_losses = [g(r, "val_loss") for r in rows if g(r, "val_loss") is not None]
    val_maes   = [g(r, "val_mae")  for r in rows if g(r, "val_mae")  is not None]

    if not val_losses:
        continue

    best_vl  = min(val_losses)
    best_vm  = min(val_maes) if val_maes else None
    final_vl = g(rows[-1], "val_loss")
    final_vm = g(rows[-1], "val_mae")
    epochs   = len(rows)

    reg_data = registry.get(run, {})
    cfg      = reg_data.get("config", {})
    aug      = cfg.get("augmentation_profile", "—")
    lr       = cfg.get("learning_rate", "—")
    bs       = cfg.get("batch_size", "—")

    results.append({
        "run": run,
        "epochs": epochs,
        "best_val_loss": best_vl,
        "best_val_mae": best_vm,
        "final_val_loss": final_vl,
        "final_val_mae": final_vm,
        "aug": aug,
        "lr": lr,
        "batch": bs,
    })

results.sort(key=lambda r: r["best_val_loss"])

print()
hdr = "{:<44} {:>6}  {:>13}  {:>12}  {:>14}  {:>12}"
row_fmt = "{run:<44} {epochs:>6}  {best_val_loss:>13.6f}  {best_val_mae_str:>12}  {final_val_loss_str:>14}  {final_val_mae_str:>12}"
print(hdr.format("Run", "Epochs", "Best Val Loss", "Best ValMAE", "Final Val Loss", "Final ValMAE"))
print("-" * 110)

for i, r in enumerate(results):
    bvm_s  = f"{r['best_val_mae']:.6f}"  if r["best_val_mae"]  is not None else "—"
    fvl_s  = f"{r['final_val_loss']:.6f}" if r["final_val_loss"] is not None else "—"
    fvm_s  = f"{r['final_val_mae']:.6f}"  if r["final_val_mae"]  is not None else "—"
    prefix = ">>> #1 BEST  " if i == 0 else f"    #{i+1:>2}       "
    print(prefix + row_fmt.format(
        run=r["run"], epochs=r["epochs"],
        best_val_loss=r["best_val_loss"],
        best_val_mae_str=bvm_s,
        final_val_loss_str=fvl_s,
        final_val_mae_str=fvm_s,
    ))

print()
print("=" * 60)
best = results[0]
print(f"  WINNER : {best['run']}")
print(f"  Best val_loss  : {best['best_val_loss']:.6f}")
print(f"  Best val_MAE   : {best['best_val_mae']:.6f}" if best["best_val_mae"] else "  Best val_MAE   : —")
print(f"  Trained epochs : {best['epochs']}")
print(f"  Checkpoint file: {best['run']}_best.h5  /  {best['run']}_final.keras")
print("=" * 60)
