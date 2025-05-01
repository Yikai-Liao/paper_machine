import { slugifyStr } from "@utils/slugify";
import Author from "./Author";
import Datetime from "./Datetime";

interface SubtitleProps {
  subtitleTransitionName: string;
  author: string;
  pubDatetime: string | Date;
  modDatetime: string | Date | undefined | null;
  score?: number;
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
  score,
  size = "sm",
  className = "",
}: Props) {
  const spanProps = {
    style: { viewTransitionName: slugifyStr(subtitleTransitionName) },
  };

  return (
    <span
      {...spanProps}
      className={`flex items-center flex-wrap justify-between opacity-80 ${className}`.trim()}
    >
      <Author author={author} size={size} />
      {score !== undefined && (
        <span className={`italic opacity-80 ${size === 'sm' ? 'text-sm' : 'text-base'}`}>
          Score: {score.toFixed(2)}
        </span>
      )}
      <Datetime
        pubDatetime={pubDatetime}
        modDatetime={modDatetime}
        size={size}
      />
    </span>
  );
}
