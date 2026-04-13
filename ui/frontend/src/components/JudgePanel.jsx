import { motion } from "framer-motion";
import { JudgeAvatar } from "./Avatars";
import { parseCitations, CHUNK_COLORS } from "../utils/citationParser";

const VERDICT_STYLES = {
  yes: "from-emerald-500 to-emerald-600",
  no: "from-rose-500 to-rose-600",
  maybe: "from-amber-500 to-amber-600",
};

export default function JudgePanel({ judge, onCitationClick }) {
  if (!judge) return null;

  const segments = parseCitations(judge.reasoning);
  const gradientClass = VERDICT_STYLES[judge.prediction] || VERDICT_STYLES.maybe;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
      className="bg-white border border-gray-200 rounded-lg shadow-sm overflow-hidden"
    >
      {/* Verdict header */}
      <div className={`bg-gradient-to-r ${gradientClass} px-6 py-5 text-white`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <JudgeAvatar size={44} />
            <div>
              <p className="font-semibold text-sm text-white/90">GRPO Judge</p>
              <p className="text-xs text-white/70">decisive_v1</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-xs text-white/70 uppercase tracking-wide">Final Verdict</p>
            <p className="text-3xl font-bold">{judge.prediction.toUpperCase()}</p>
          </div>
        </div>
      </div>

      {/* Explanation */}
      <div className="p-5">
        <h4 className="text-sm font-semibold text-gray-900 mb-2">Explanation</h4>
        <div className="text-sm text-gray-600 leading-relaxed">
          {segments.map((seg, i) =>
            seg.type === "text" ? (
              <span key={i}>{seg.content}</span>
            ) : (
              <button
                key={i}
                onClick={() => onCitationClick(seg.index)}
                className={`inline-flex items-center justify-center px-1.5 py-0.5 mx-0.5 rounded-full text-xs font-bold cursor-pointer transition-colors ${
                  CHUNK_COLORS[seg.index]?.bg || "bg-gray-500"
                } ${CHUNK_COLORS[seg.index]?.text || "text-white"} hover:opacity-80`}
              >
                {seg.label}
              </button>
            )
          )}
        </div>

        {judge.full_response && (
          <>
            <h4 className="text-sm font-semibold text-gray-900 mt-4 mb-2">Debate Summary</h4>
            <p className="text-sm text-gray-500 leading-relaxed">{judge.full_response}</p>
          </>
        )}
      </div>
    </motion.div>
  );
}
