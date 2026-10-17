"""Image-processing service for the PPE (EPI) detection project.

Receives a raw image, prepares it for the downstream YOLO model, and returns a lossless PNG
with the same pixel grid, so the model's boxes stay valid on the original image.
"""
