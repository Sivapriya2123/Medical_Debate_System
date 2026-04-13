import { useState } from "react";
import { ChevronDown, ChevronUp, AlertTriangle, Clock } from "lucide-react";
import { CHUNK_COLORS } from "../utils/citationParser";

export default function EvidenceChunk({ chunk, isActive, onClick }) {
  const [expanded, setExpanded] = useState(false);
  const color = CHUNK_COLORS[chunk.index] || CHUNK_COLORS[0];
  const showFull = expanded || isActive;

  return (
    <div
      id={`evidence-${chunk.index}`}
      onClick={onClick}
      className={`border rounded-lg p-3 cursor-pointer transition-all duration-300 ${
        isActive
          ? `${color.light} ${color.border} border-2`
          : "bg-white border-gray-200 hover:border-gray-300"
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span
            className={`inline-flex items-center justify-center w-7 h-7 rounded-full text-xs font-bold ${color.bg} ${color.text}`}
          >
            {chunk.index + 1}
          </span>
          <span className="text-xs text-gray-400 font-mono">
            {chunk.source || "PubMed"}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          {chunk.is_outdated && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-50 text-amber-700">
              <Clock className="w-3 h-3" /> Outdated
            </span>
          )}
          {chunk.has_conflict && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-rose-50 text-rose-700">
              <AlertTriangle className="w-3 h-3" /> Conflict
            </span>
          )}
        </div>
      </div>

      {/* Text */}
      <p className={`text-sm text-gray-600 leading-relaxed ${showFull ? "" : "line-clamp-3"}`}>
        {chunk.text}
      </p>

      {/* Footer */}
      <div className="flex items-center justify-between mt-2">
        {/* Relevance bar */}
        <div className="flex items-center gap-2 flex-1">
          <span className="text-xs text-gray-400">Relevance</span>
          <div className="flex-1 h-1.5 bg-gray-100 rounded-full max-w-[100px]">
            <div
              className={`h-full rounded-full ${color.bg}`}
              style={{ width: `${Math.min(chunk.relevance_score * 100, 100)}%` }}
            />
          </div>
          <span className="text-xs text-gray-400">{chunk.relevance_score.toFixed(2)}</span>
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation();
            setExpanded(!expanded);
          }}
          className="text-gray-400 hover:text-gray-600 ml-2"
        >
          {showFull ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>
      </div>
    </div>
  );
}
