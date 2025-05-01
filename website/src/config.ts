import type { Site, SocialObjects } from "./types";
import type { GiscusProps } from "@giscus/react";

export const FONT_CONFIG = {
  fontUrl: 'https://chinese-font.netlify.app/font-cdn/packages/maple-mono-cn/dist/MapleMono-CN-Light/result.css',
  fontFamily: 'Maple Mono CN Light',
  fallbackFonts: [
    'IBM Plex Mono',
    'Microsoft YaHei',
    'PingFang SC',
    'Hiragino Sans GB',
    'Source Han Sans SC',
    'Noto Sans CJK SC',
    'sans-serif',
    'monospace'
  ]
};

export const SITE: Site = {
  website: import.meta.env.PUBLIC_WEBSITE || "https://yikai-liao.github.io/paper_machine/", // can be overridden by PUBLIC_WEBSITE env var
  base: import.meta.env.PUBLIC_BASE || "/paper_machine", // can be overridden by PUBLIC_BASE env var
  author: "Yikai Liao",
  profile: "https://yikai-liao.github.io/academicpages/",
  desc: "An efficient and effective personalized paper recommendation solution",
  title: "Daily Paper Machine",
  ogImage: "/labtalks/smlab-og.png", // Open Graph Image
  lightAndDarkMode: true,
  postPerIndex: 6,
  postPerPage: 6,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
};

export const LOCALE = {
  lang: "en", // html lang code. Set this empty and default will be "en"
  langTag: ["en-EN"], // BCP 47 Language Tags. Set this empty [] to use the environment default
} as const;

export const LOGO_IMAGE = {
  enable: false,
  svg: true,
  width: 216,
  height: 46,
};

export const SOCIALS: SocialObjects = [
  {
    name: "Twitter",
    href: "https://x.com/RealYikaiLiao",
    linkTitle: `Yikai Liao on Twitter`,
    active: true,
  },
  {
    name: "Mail",
    href: "yikai003@e.ntu.edu.sg",
    linkTitle: `Yikai Liao on Mail`,
    active: true,
  },
  {
    name: "Github",
    href: "https://github.com/Yikai-Liao",
    linkTitle: `Yikai Liao on Github`,
    active: true,
  },
];

export const GISCUS: GiscusProps = {
  repo: "Yikai-Liao/paper_machine", // 格式为 username/repo
  repoId: "R_kgDOOiKqrg", // 在 https://giscus.app 上配置后获取
  category: "Giscus", // 通常使用 "Announcements" 或 "General"
  categoryId: "DIC_kwDOOiKqrs4CpqMc", // 在 https://giscus.app 上配置后获取
  mapping: "pathname", // 可选 'pathname', 'url', 'title', 'og:title'
  reactionsEnabled: "1", // 启用评论表情反应: '1' = 启用, '0' = 禁用
  emitMetadata: "0", // 不发送额外的元数据
  inputPosition: "bottom", // 评论框位置: 'top' 或 'bottom'
  lang: LOCALE.lang, // 使用网站默认语言设置
};
