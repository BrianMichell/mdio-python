from mdio import open_mdio
from datetime import datetime, UTC
from mdio import to_mdio

input_mdio = "tmp.mdio"
output_mdio = "tmp_output.mdio"

ds = open_mdio(input_mdio, chunks={})

print("Dataset when first opened:")
print(ds)

for coord in ds.coords:
    ds[coord] = ds[coord].compute()

ds = ds.drop_vars(["raw_headers"])
for var in ds.data_vars:
    if "fast" in var:
        ds = ds.drop_vars([var])

ds = ds.rename_vars(amplitude="probability")
del ds["probability"].attrs["statsV1"]
ds["probability"].encoding["fill_value"] = 0
ds["probability"].encoding["_FillValue"] = 0

ds["support"] = ds["probability"].copy().astype("uint32")
ds["support"].encoding = ds["probability"].encoding
ds["support"].encoding.pop("dtype")

ds.attrs["createdOn"] = datetime.now(UTC).isoformat()
ds.attrs["name"] = "Sfm3DInfereneResult"
ds.attrs["attributes"]["defaultVariableName"] = "probability"

mode = "w"

print("Dataset before writing:")
print(ds)

to_mdio(ds, output_mdio, compute=False, mode=mode)