import { useState } from "react";
import { motion } from "framer-motion";
import { DoctorAAvatar, DoctorBAvatar } from "./Avatars";
import { parseCitations, CHUNK_COLORS } from "../utils/citationParser";
import CitationTooltip from "./CitationTooltip";

const POSITION_STYLES = {
  yes: "bg-emerald-50 text-emerald-700 border-emerald-200",
  no: "bg-rose-50 text-rose-700 border-rose-200",
  maybe: "bg-amber-50 text-amber-700 border-amber-200",
};

export default function DoctorMessage({
  doctor,
  round,
  position,
  confidence,
  reasoning,
  evidenceCited,
  onCitationClick,
  chunks,
}) {
  const [tooltip, setTooltip] = useState(null);
  const isA = doctor === "A";
  const segments = parseCitations(reasoning);

  const handleCitationHover = (index, e) => {
    const chunk = chunks?.find((c) => c.index === index);
    if (chunk) {
      setTooltip({
        chunkIndex: index,
        chunkText: chunk.text,
        position: { x: e.clientX, y: e.clientY },
      });
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3, delay: (round - 1) * 0.15 }}
      className={`border rounded-lg p-4 shadow-sm ${
        isA ? "bg-white border-blue-200" : "bg-white border-violet-200"
      }`}
      style={{ borderLeftWidth: "4px", borderLeftColor: isA ? "#2563EB" : "#7C3AED" }}
    >
      {/* Header */}
      <div className="flex items-start gap-3 mb-3">
        <div className="shrink-0">
          {isA ? <DoctorAAvatar size={44} /> : <DoctorBAvatar size={44} />}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className={`text-sm font-semibold ${isA ? "text-blue-700" : "text-violet-700"}`}>
              {isA ? "Dr. Chen" : "Dr. Patel"}
            </span>
            <span className="text-xs text-gray-400">
              {isA ? "Conservative Clinician" : "Diagnostic Generalist"}
            </span>
            <span className="text-xs text-gray-400 ml-auto">
              {round === 1 ? "Opening" : `Rebuttal (R${round})`}
            </span>
          </div>

          {/* Position + confidence */}
          <div className="flex items-center gap-3 mt-2">
            <span
              className={`inline-flex px-2.5 py-0.5 rounded-full text-xs font-semibold border ${POSITION_STYLES[position]}`}
            >
              {position.toUpperCase()}
            </span>
            <div className="flex items-center gap-2 flex-1">
              <span className="text-xs text-gray-400">Confidence</span>
              <div className="flex-1 h-1.5 bg-gray-100 rounded-full max-w-[120px]">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    isA ? "bg-blue-500" : "bg-violet-500"
                  }`}
                  style={{ width: `${confidence * 100}%` }}
                />
              </div>
              <span className="text-xs text-gray-500 font-medium">
                {(confidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Reasoning with citation pills */}
      <div className="text-sm text-gray-600 leading-relaxed">
        {segments.map((seg, i) =>
          seg.type === "text" ? (
            <span key={i}>{seg.content}</span>
          ) : (
            <button
              key={i}
              onClick={() => onCitationClick(seg.index)}
              onMouseEnter={(e) => handleCitationHover(seg.index, e)}
              onMouseLeave={() => setTooltip(null)}
              className={`inline-flex items-center justify-center px-1.5 py-0.5 mx-0.5 rounded-full text-xs font-bold cursor-pointer transition-colors ${
                CHUNK_COLORS[seg.index]?.bg || "bg-gray-500"
              } ${CHUNK_COLORS[seg.index]?.text || "text-white"} hover:opacity-80`}
            >
              {seg.label}
            </button>
          )
        )}
      </div>

      {/* Evidence cited summary */}
      {evidenceCited?.length > 0 && (
        <div className="mt-2 flex items-center gap-1.5">
          <span className="text-xs text-gray-400">Cited:</span>
          {evidenceCited.map((idx) => (
            <span
              key={idx}
              className={`inline-flex items-center justify-center w-5 h-5 rounded-full text-xs font-bold ${
                CHUNK_COLORS[idx]?.bg || "bg-gray-400"
              } ${CHUNK_COLORS[idx]?.text || "text-white"}`}
            >
              {idx + 1}
            </span>
          ))}
        </div>
      )}

      {tooltip && (
        <CitationTooltip
          chunkIndex={tooltip.chunkIndex}
          chunkText={tooltip.chunkText}
          position={tooltip.position}
        />
      )}
    </motion.div>
  );
}
