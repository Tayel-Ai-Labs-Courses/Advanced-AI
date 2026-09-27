# Reliable AI Risk Intelligence - Session 5 runtime

The runtime is the productized continuation of the earlier course evidence chain:

`data contract -> legitimate model selection -> explanation evidence -> reliability controls -> persisted artifacts -> verified application`

Quick check:

```bash
python train_model.py
python verify_product.py
pytest -q
python app.py
```

See `DEPLOYMENT.md` for local, temporary-public, and hosted deployment paths.
