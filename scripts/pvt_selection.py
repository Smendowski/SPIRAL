import timm

pvt_variants = [
    "pvt_v2_b0",
    "pvt_v2_b1",
    "pvt_v2_b2",
    "pvt_v2_b3",
    "pvt_v2_b4",
    "pvt_v2_b5",
]

for variant in pvt_variants:
    model = timm.create_model(variant, pretrained=False, features_only=True)
    channels = model.feature_info.channels()
    print(f"{variant}: out_channels {channels}, last layer: {channels[-1]}")
