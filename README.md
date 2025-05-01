# paper_machine
An efficient and effective personalized paper recommendation solution

## TODO: Custom Fonts for Open Graph (OG) Images

Currently, the Open Graph (OG) images (used for social media previews) are generated using default fonts (`LXGW Bright`, `IBM Plex Mono`) loaded via the `website/src/utils/loadGoogleFont.ts` utility. The primary site font defined in `website/src/config.ts` (`FONT_CONFIG`) is loaded via CSS CDN for browser rendering but cannot be reliably used for build-time OG image generation due to CDN restrictions (HTTP 403 Forbidden errors when fetching font files directly).

To enable custom fonts (like the one specified in `FONT_CONFIG.fontFamily`) in the generated OG images, follow these steps:

1.  **Download Font Files**: Obtain the necessary font files (e.g., `MapleMono-CN-Light.ttf` or `.otf`) from a trusted source (like the font's official repository).
    *   Also, download any fallback fonts specified in `FONT_CONFIG.fallbackFonts` (e.g., `IBMPlexMono-Regular.ttf`, `IBMPlexMono-Bold.ttf`) if they are needed to render all characters in your site title/description.
2.  **Place Font Files**: Create a directory like `website/src/assets/fonts/` and place the downloaded `.ttf` or `.otf` files inside it. This directory is only used during build time and won't be included in the final deployed site assets.
3.  **Modify OG Templates**: Edit the OG template files (`website/src/utils/og-templates/site.tsx` and potentially `website/src/utils/og-templates/post.tsx`):
    *   Import Node.js built-in modules: `import fs from "fs/promises";` and `import path from "path";`.
    *   Import `FONT_CONFIG` from `@config`.
    *   Remove the import and usage of `loadGoogleFonts`.
    *   Implement a helper function (e.g., `readLocalFont`) to read font files using `fs.readFile`. Ensure you construct the correct absolute path to the files in `src/assets/fonts/`, for example: `path.join(process.cwd(), 'src/assets/fonts/MapleMono-CN-Light.ttf')`.
    *   Create an array for Satori's `fonts` option. For each font file you placed in `src/assets/fonts/`, call `readLocalFont` and add an object to the array like:
        ```typescript
        {
          name: 'Maple Mono CN Light', // Must match the fontFamily name used in CSS/FONT_CONFIG
          data: await readLocalFont('src/assets/fonts/MapleMono-CN-Light.ttf'),
          weight: 400, // Adjust weight/style as needed
          style: 'normal',
        }
        ```
        *   Include entries for *all* required fonts (primary and fallbacks, different weights/styles if used).
    *   Pass this `fonts` array to the `satori()` function's options.
    *   Update the `fontFamily` style properties within the JSX template to use `FONT_CONFIG.fontFamily` and its fallbacks correctly, e.g., `"${FONT_CONFIG.fontFamily}", ${FONT_CONFIG.fallbackFonts.map(f => `'${f}'`).join(", ")}`.
4.  **Test Build**: Run the build command (e.g., `npm run build`) and verify that OG images are generated without errors and use the custom font.
