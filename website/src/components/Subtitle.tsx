import { slugifyStr } from "@utils/slugify";
import Author from "./Author";
import Datetime from "./Datetime";

interface SubtitleProps {
  subtitleTransitionName: string;
  author: string;
  pubDatetime: string | Date;
  modDatetime: string | Date | undefined | null;
}

interface Props extends SubtitleProps {
  size?: "sm" | "lg";
  className?: string;
}

export default function Subtitle({
  subtitleTransitionName,
  author,
  pubDatetime,
  modDatetime,
  size = "sm",
  className = "",
}: Props) {

  const spanProps = {
    style: { viewTransitionName: slugifyStr(subtitleTransitionName) },
  };

  return (
    <span {...spanProps} className={`flex items-center justify-between space-x-20 opacity-80 ${className}`.trim()}>
      <Author author={author} size={size} className={className} />
      <Datetime
        pubDatetime={pubDatetime}
        modDatetime={modDatetime}
        size={size}
        className={className}
      />
    </span>
  );
}
