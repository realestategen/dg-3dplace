"""Hardcoded prompt presets for automated DG-3DPlace paper experiments.

These prompts are intentionally varied across object types, spatial positions,
and material attributes so the automation can support qualitative grids and
quantitative ablations in a paper setting.
"""

PROMPT_SPECS = [
    {
        "id": "car_center_red",
        "prompt": "a red car parked in the middle of the scene",
        "target_text": "A real-estate scene with a red car parked in the middle",
        "object_class": "car",
    },
    {
        "id": "car_left_blue",
        "prompt": "a compact blue car on the left side near the walkway",
        "target_text": "A real-estate scene with a compact blue car on the left side",
        "object_class": "car",
    },
    {
        "id": "car_right_white",
        "prompt": "a white car placed on the right side of the courtyard",
        "target_text": "A real-estate scene with a white car on the right side",
        "object_class": "car",
    },
    {
        "id": "bench_modern",
        "prompt": "a modern wooden bench near the central open area",
        "target_text": "A real-estate scene with a modern wooden bench",
        "object_class": "bench",
    },
    {
        "id": "bench_long",
        "prompt": "a long minimalist bench near the garden edge",
        "target_text": "A real-estate scene with a long minimalist bench",
        "object_class": "bench",
    },
    {
        "id": "planter_green",
        "prompt": "a large green planter with foliage near the front",
        "target_text": "A real-estate scene with a large green planter",
        "object_class": "plant",
    },
    {
        "id": "chair_white",
        "prompt": "a white lounge chair near the right side",
        "target_text": "A real-estate scene with a white lounge chair",
        "object_class": "chair",
    },
    {
        "id": "chair_black",
        "prompt": "a black lounge chair near the seating area",
        "target_text": "A real-estate scene with a black lounge chair",
        "object_class": "chair",
    },
    {
        "id": "table_round",
        "prompt": "a round coffee table beside the seating area",
        "target_text": "A real-estate scene with a round coffee table",
        "object_class": "table",
    },
    {
        "id": "sofa_gray",
        "prompt": "a gray two-seater sofa close to the center",
        "target_text": "A real-estate scene with a gray two-seater sofa",
        "object_class": "sofa",
    },
    {
        "id": "vase_decor",
        "prompt": "a decorative ceramic vase near the bench",
        "target_text": "A real-estate scene with a decorative ceramic vase",
        "object_class": "vase",
    },
    {
        "id": "lamp_floor",
        "prompt": "a tall floor lamp near the corner of the room",
        "target_text": "A real-estate scene with a tall floor lamp",
        "object_class": "lamp",
    },
]

SOURCE_TEXT = "A real-estate scene"

BASELINE_PROMPT = {
    "id": "baseline",
    "prompt": "no added object, keep the scene unchanged",
    "target_text": "A real-estate scene",
    "object_class": "none",
}
