/* JPEG orientation belongs to image decoding, before geometry or patch order. */
typedef enum { GLM53_EXIF_LE, GLM53_EXIF_BE } glm53_exif_order;
typedef enum {
    GLM53_EXIF_NORMAL = 1, GLM53_EXIF_MIRROR_X, GLM53_EXIF_HALF_TURN,
    GLM53_EXIF_MIRROR_Y, GLM53_EXIF_TRANSPOSE, GLM53_EXIF_CLOCKWISE,
    GLM53_EXIF_TRANSVERSE, GLM53_EXIF_COUNTERCLOCKWISE
} glm53_exif_orientation;
enum {
    GLM53_JPEG_APP1 = 0xe1, GLM53_JPEG_SOS = 0xda, GLM53_JPEG_EOI = 0xd9,
    GLM53_TIFF_MAGIC = 42, GLM53_EXIF_ORIENTATION = 0x0112, GLM53_EXIF_SHORT = 3,
    GLM53_EXIF_ENTRY_BYTES = 12
};

static uint16_t glm53_exif16(const uint8_t *p, glm53_exif_order order) {
    return order == GLM53_EXIF_LE ? (uint16_t)(p[0] | (p[1] << 8))
                                  : (uint16_t)((p[0] << 8) | p[1]);
}

static uint32_t glm53_exif32(const uint8_t *p, glm53_exif_order order) {
    return order == GLM53_EXIF_LE
        ? (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24)
        : ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}

static glm53_exif_orientation glm53_jpeg_orientation(const uint8_t *data, size_t bytes) {
    if (bytes < 4u || data[0] != 0xff || data[1] != 0xd8) { return GLM53_EXIF_NORMAL; }
    size_t pos = 2u;
    while (pos < bytes && data[pos] == 0xff) {
        while (pos < bytes && data[pos] == 0xff) { pos++; }
        if (pos >= bytes) { break; }
        const uint8_t marker = data[pos++];
        if (marker == GLM53_JPEG_SOS || marker == GLM53_JPEG_EOI) { break; }
        if (marker == 0x01 || marker == 0xd8 || (marker >= 0xd0 && marker <= 0xd7)) { continue; }
        if (bytes - pos < 2u) { break; }
        const size_t segment = ((size_t)data[pos] << 8) | data[pos + 1u];
        if (segment < 2u || segment > bytes - pos) { break; }
        if (marker != GLM53_JPEG_APP1 || segment < 16u ||
            memcmp(data + pos + 2u, "Exif\0\0", 6u) != 0) { pos += segment; continue; }

        const uint8_t *tiff = data + pos + 8u;
        const size_t size = segment - 8u;
        const glm53_exif_order order = tiff[0] == 'I' && tiff[1] == 'I' ? GLM53_EXIF_LE : GLM53_EXIF_BE;
        if ((order == GLM53_EXIF_BE && (tiff[0] != 'M' || tiff[1] != 'M')) ||
            glm53_exif16(tiff + 2u, order) != GLM53_TIFF_MAGIC) { break; }
        const uint32_t ifd = glm53_exif32(tiff + 4u, order);
        if (ifd > size || size - ifd < 2u) { break; }
        const uint16_t count = glm53_exif16(tiff + ifd, order);
        if (count > (size - ifd - 2u) / GLM53_EXIF_ENTRY_BYTES) { break; }
        for (uint16_t i = 0u; i < count; i++) {
            const uint8_t *entry = tiff + ifd + 2u + (size_t)i * GLM53_EXIF_ENTRY_BYTES;
            if (glm53_exif16(entry, order) != GLM53_EXIF_ORIENTATION ||
                glm53_exif16(entry + 2u, order) != GLM53_EXIF_SHORT ||
                glm53_exif32(entry + 4u, order) != 1u) { continue; }
            const uint16_t value = glm53_exif16(entry + 8u, order);
            return value >= GLM53_EXIF_NORMAL && value <= GLM53_EXIF_COUNTERCLOCKWISE
                ? (glm53_exif_orientation)value : GLM53_EXIF_NORMAL;
        }
        pos += segment;
    }
    return GLM53_EXIF_NORMAL;
}

static uint8_t *glm53_orient_rgb(uint8_t *rgb, int *width, int *height,
                                glm53_exif_orientation orientation) {
    if (orientation == GLM53_EXIF_NORMAL) { return rgb; }
    const uint32_t w = (uint32_t)*width, h = (uint32_t)*height;
    const uint32_t out_w = orientation >= GLM53_EXIF_TRANSPOSE ? h : w;
    const uint32_t out_h = orientation >= GLM53_EXIF_TRANSPOSE ? w : h;
    uint8_t *out = xmalloc((size_t)out_w * out_h * 3u);
    for (uint32_t y = 0u; y < h; y++) {
        for (uint32_t x = 0u; x < w; x++) {
            uint32_t dx = x, dy = y;
            switch (orientation) {
                case GLM53_EXIF_MIRROR_X: dx = w - 1u - x; break;
                case GLM53_EXIF_HALF_TURN: dx = w - 1u - x; dy = h - 1u - y; break;
                case GLM53_EXIF_MIRROR_Y: dy = h - 1u - y; break;
                case GLM53_EXIF_TRANSPOSE: dx = y; dy = x; break;
                case GLM53_EXIF_CLOCKWISE: dx = h - 1u - y; dy = x; break;
                case GLM53_EXIF_TRANSVERSE: dx = h - 1u - y; dy = w - 1u - x; break;
                case GLM53_EXIF_COUNTERCLOCKWISE: dx = y; dy = w - 1u - x; break;
                default: break;
            }
            memcpy(out + ((size_t)dy * out_w + dx) * 3u,
                   rgb + ((size_t)y * w + x) * 3u, 3u);
        }
    }
    stbi_image_free(rgb);
    *width = (int)out_w;
    *height = (int)out_h;
    return out;
}
