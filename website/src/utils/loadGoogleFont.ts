import type { FontStyle, FontWeight } from "satori";

export type FontOptions = {
  name: string;
  data: ArrayBuffer;
  weight: FontWeight | undefined;
  style: FontStyle | undefined;
};

async function loadGoogleFont(
  font: string,
  text: string,
): Promise<ArrayBuffer> {
  const API = `https://fonts.googleapis.com/css2?family=${font}&text=${encodeURIComponent(text)}`;

  const css = await (
    await fetch(API, {
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; de-at) AppleWebKit/533.21.1 (KHTML, like Gecko) Version/5.0.5 Safari/533.21.1",
      },
    })
  ).text();

  const resource = css.match(
    /src: url\((.+)\) format\('(opentype|truetype)'\)/,
  );

  if (!resource) throw new Error("Failed to download dynamic font");

  const res = await fetch(resource[1]);

  if (!res.ok) {
    throw new Error("Failed to download dynamic font. Status: " + res.status);
  }

  const fonts: ArrayBuffer = await res.arrayBuffer();
  return fonts;
}

async function loadGoogleFonts(
  text: string,
): Promise<
  Array<{ name: string; data: ArrayBuffer; weight: number; style: string }>
> {
  const fontsConfig = [
    {
      name: "LXGW Bright",
      font: "LXGW+Bright",
      weight: 400,
      style: "normal",
    },
    {
      name: "IBM Plex Mono",
      font: "IBM+Plex+Mono",
      weight: 400,
      style: "normal",
    },
    {
      name: "IBM Plex Mono",
      font: "IBM+Plex+Mono:wght@700",
      weight: 700,
      style: "bold",
    },
  ];

  const fontPromises = fontsConfig.map(async ({ name, font, weight, style }) => {
    try {
      console.log(`Attempting to load font: ${name} (${font})`);
      const data = await loadGoogleFont(font, text);
      console.log(`Successfully loaded font: ${name}`);
      return { name, data, weight, style };
    } catch (error) {
      console.warn(`WARN: Failed to load font '${name}' (${font}) for OG image generation. Error: ${error instanceof Error ? error.message : error}`);
      return null;
    }
  });

  const loadedFonts = await Promise.all(fontPromises);

  const successfulFonts = loadedFonts.filter(font => font !== null) as Array<{
    name: string;
    data: ArrayBuffer;
    weight: number;
    style: string;
  }>;

  if (successfulFonts.length === 0) {
    console.warn("WARN: No fonts could be loaded for OG image generation. Satori will use fallback fonts.");
  }

  return successfulFonts;
}

export default loadGoogleFonts;
