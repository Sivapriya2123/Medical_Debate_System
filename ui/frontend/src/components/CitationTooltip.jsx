import { CHUNK_COLORS } from "../utils/citationParser";

export default function CitationTooltip({ chunkIndex, chunkText, position }) {
  if (chunkIndex == null || !chunkText) return null;
  const color = CHUNK_COLORS[chunkIndex] || CHUNK_COLORS[0];

  return (
    <div
      className="fixed z-50 max-w-xs bg-white border border-gray-200 rounded-lg shadow-lg p-3 pointer-events-none"
      style={{ left: position.x, top: position.y - 8, transform: "translateY(-100%)" }}
    >
      <div className="flex items-center gap-2 mb-1">
        <span
          className={`inline-flex items-center justify-center w-5 h-5 rounded-full text-xs font-bold ${color.bg} ${color.text}`}
        >
          {chunkIndex + 1}
        </span>
        <span className="text-xs font-medium text-gray-500">Evidence</span>
      </div>
      <p className="text-xs text-gray-600 leading-relaxed">
        {chunkText.length > 120 ? chunkText.slice(0, 120) + "..." : chunkText}
      </p>
    </div>
  );
}
