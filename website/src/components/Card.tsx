import { slugifyStr } from "@utils/slugify";
import Subtitle from "./Subtitle";
import type { CollectionEntry } from "astro:content";
import { SITE } from "@config";

export interface Props {
  href?: string;
  frontmatter: CollectionEntry<"blog">["data"];
  secHeading?: boolean;
}

export default function Card({ href, frontmatter, secHeading = true }: Props) {
  const { title, author, pubDatetime, modDatetime, description, tags } = frontmatter;

  const headerProps = {
    style: { viewTransitionName: slugifyStr(title) },
    className: "text-lg font-medium decoration-dashed hover:underline",
  };

  return (
    <li className="my-6">
      <a
        href={href}
        className="inline-block text-lg font-medium text-skin-accent decoration-dashed underline-offset-4 focus-visible:no-underline focus-visible:underline-offset-0"
      >
        {secHeading ? (
          <h2 {...headerProps}>{title}</h2>
        ) : (
          <h3 {...headerProps}>{title}</h3>
        )}
      </a>
      <div>
        <Subtitle
          subtitleTransitionName={author + pubDatetime}
          author={author}
          pubDatetime={pubDatetime}
          modDatetime={modDatetime}
        />
        
        {/* 标签放在这里，紧跟在作者和时间信息后面 */}
        {tags && tags.length > 0 && (
          <div className="mt-0.5 text-sm italic opacity-65">
            {tags.map((tag, i) => (
              <span key={tag}>
                <a 
                  href={`${SITE.base}/tags/${slugifyStr(tag)}/`}
                  className="hover:text-skin-accent"
                >
                  #{tag}
                </a>
                {i < tags.length - 1 && ", "}
              </span>
            ))}
          </div>
        )}
      </div>
      
      <p className="mt-2">{description}</p>
    </li>
  );
}
