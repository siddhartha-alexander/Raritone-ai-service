"""
3D Asset Generation Interface

This module defines the interface for future AI-based
3D asset generation.

The current Raritone 3D pipeline is CPU-based and works
with existing permitted GLB/GLTF assets.

Future GPU-enabled models such as TripoSR or another
approved image-to-3D model can be connected here without
changing the validation, optimization, preview, metadata,
or review pipeline.
"""


def generate_3d_asset(
    product_image,
    product_id,
):
    """
    Future GPU-based 3D generation interface.

    Parameters
    ----------
    product_image : str
        Path or reference to the product image.

    product_id : str
        Unique Raritone product identifier.

    Returns
    -------
    dict
        Generation status information.

    Notes
    -----
    Actual AI 3D generation is intentionally not executed
    in the current CPU-only pipeline.
    """

    return {
        "success": False,
        "status": "GPU_GENERATION_NOT_ENABLED",
        "product_id": product_id,
        "product_image": str(product_image),
        "generated_asset": None,
        "message": (
            "GPU-based 3D asset generation is not enabled "
            "in the current CPU pipeline. This interface "
            "is reserved for future model integration."
        ),
    }


if __name__ == "__main__":

    result = generate_3d_asset(
        product_image="example_product.jpg",
        product_id="PROD-TEST-001",
    )

    print(result)