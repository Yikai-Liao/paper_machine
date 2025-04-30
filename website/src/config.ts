import type { Site, SocialObjects } from "./types";

export const SITE: Site = {
  website: "https://yikai-liao.github.io/paper_machine/", // replace this with your deployed domain
  base: "/paper_machine", // replace this with your project root on the server
  author: "Yikai Liao",
  profile: "https://yikai-liao.github.io/academicpages/",
  desc: "An efficient and effective personalized paper recommendation solution",
  title: "Daily Paper Machine",
  ogImage: "/labtalks/smlab-og.png", // Open Graph Image
  lightAndDarkMode: true,
  postPerIndex: 5,
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
    href: "https://x.com/mishra_lab",
    linkTitle: `SMLab on Twitter`,
    active: true,
  },
  {
    name: "Mail",
    href: "mailto:smlab@niser.ac.in",
    linkTitle: `Email us at ${SITE.title}`,
    active: true,
  },
  {
    name: "Github",
    href: "https://github.com/JeS24/smlab-talks/",
    linkTitle: `SMLab Weekly Talks on Github`,
    active: true,
  },
  {
    name: "GitLab",
    href: "https://gitlab.niser.ac.in/",
    linkTitle: `SMLab on GitLab`,
    active: true,
  },
];
