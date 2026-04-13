/**
 * Parse citation references [1], [2], etc. from text into segments.
 *
 * Returns an array of { type: "text"|"citation", content/index/label }
 */
export function parseCitations(text) {
  if (!text) return [{ type: "text", content: "" }];

  const segments = [];
  const regex = /\[(\d+)\]/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    const num = parseInt(match[1], 10);
    if (num < 1 || num > 10) continue;

    if (match.index > lastIndex) {
      segments.push({ type: "text", content: text.slice(lastIndex, match.index) });
    }

    segments.push({ type: "citation", index: num - 1, label: match[0] });
    lastIndex = regex.lastIndex;
  }

  if (lastIndex < text.length) {
    segments.push({ type: "text", content: text.slice(lastIndex) });
  }

  if (segments.length === 0) {
    segments.push({ type: "text", content: text });
  }

  return segments;
}

/** Colors for evidence chunk indices [0]-[4] */
export const CHUNK_COLORS = [
  { bg: "bg-blue-500",    text: "text-white", border: "border-blue-500",    light: "bg-blue-50"   },
  { bg: "bg-violet-500",  text: "text-white", border: "border-violet-500",  light: "bg-violet-50" },
  { bg: "bg-teal-500",    text: "text-white", border: "border-teal-500",    light: "bg-teal-50"   },
  { bg: "bg-rose-500",    text: "text-white", border: "border-rose-500",    light: "bg-rose-50"   },
  { bg: "bg-amber-500",   text: "text-white", border: "border-amber-500",   light: "bg-amber-50"  },
];
