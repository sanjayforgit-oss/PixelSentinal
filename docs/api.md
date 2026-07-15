# PixelSentinel API Reference

## Base URL

```
http://localhost:8000/api/v1
```

## Endpoints

### Upload

- **POST** `/upload/image` - Upload single image
- **POST** `/upload/batch` - Upload multiple images

### Inference

- **POST** `/inference/enhance` - Enhance image
- **POST** `/inference/colorize` - Colorize image
- **POST** `/inference/segment` - Segment image

### Evaluation

- **POST** `/evaluation/compare` - Compare images

### Health

- **GET** `/health/status` - Check API status