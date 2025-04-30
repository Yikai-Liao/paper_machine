interface AuthorProps {
  author: string;
}

interface Props extends AuthorProps {
  size?: "sm" | "lg";
  className?: string;
}

export default function Author({ author, size = "sm", className = "" }: Props) {
  return (
    <div
      className={`italic opacity-80 ${className} ${size === "sm" ? "text-sm" : "text-base"}`.trim()}
    >
      {author}
    </div>
  );
}
