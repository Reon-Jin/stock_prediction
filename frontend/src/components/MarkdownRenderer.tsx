import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Props = {
  content: string;
};

function normalizeMarkdown(input: string): string {
  if (!input) return "";

  let text = input.replace(/\r\n/g, "\n").trim();

  // 去掉行尾空格
  text = text
    .split("\n")
    .map((line) => line.replace(/[ \t]+$/g, ""))
    .join("\n");

  // 连续 3 个及以上空行 => 压成 1 个
  text = text.replace(/\n{3,}/g, "\n\n");

  // 修复这种情况：
  // 1.
  //
  // 对于所有投资者：
  //
  // - xxx
  //
  // 变成：
  // 1. 对于所有投资者：
  //    - xxx
  text = text.replace(/^(\d+)\.\s*\n+\s*([^\n].*)$/gm, "$1. $2");

  // 如果列表项之间被插了多余空行，压缩掉
  text = text.replace(/^([*-] .+)\n{2,}([*-] )/gm, "$1\n$2");

  // 有序列表项之间的多余空行压缩
  text = text.replace(/^(\d+\.\s.+)\n{2,}(\d+\.\s)/gm, "$1\n$2");

  // 标题后如果出现连续空行，只保留一个
  text = text.replace(/^(#{1,6}\s.*)\n{2,}/gm, "$1\n");

  // 列表标记和它后面的正文之间，去掉多余空行
  text = text.replace(/^(\d+\.\s.*)\n{2,}([^\n-#*])/gm, "$1\n$2");

  // 列表前面的空行收紧
  text = text.replace(/\n{2,}([*-] )/g, "\n$1");
  text = text.replace(/\n{2,}(\d+\. )/g, "\n$1");
  text = text.replace(/\n{2,}([^\s#>*-])/g, "\n$1");
  text = text.replace(/^([ \t]*[-*+]|\d+\.)\s*\n+\s*/gm, "$1 ");

  return text.trim();
}

export function MarkdownRenderer({ content }: Props) {
  const normalized = normalizeMarkdown(content);

  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p>{children}</p>,
          ul: ({ children }) => <ul>{children}</ul>,
          ol: ({ children }) => <ol>{children}</ol>,
          li: ({ children }) => <li>{children}</li>,
          h1: ({ children }) => <h1>{children}</h1>,
          h2: ({ children }) => <h2>{children}</h2>,
          h3: ({ children }) => <h3>{children}</h3>,
          h4: ({ children }) => <h4>{children}</h4>,
          blockquote: ({ children }) => <blockquote>{children}</blockquote>,
          pre: ({ children }) => <pre>{children}</pre>,
          table: ({ children }) => <table>{children}</table>,
        }}
      >
        {normalized}
      </ReactMarkdown>
    </div>
  );
}
