import { motion } from "framer-motion";
import { Database } from "lucide-react";
import EvidenceChunk from "./EvidenceChunk";

export default function RetrievalPanel({ retrieval, activeChunkIndex, onChunkClick }) {
  if (!retrieval) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      className="h-full"
    >
      <div className="flex items-center gap-2 mb-3">
        <Database className="w-4 h-4 text-blue-600" />
        <h3 className="text-sm font-semibold text-gray-900">Retrieved Evidence</h3>
      </div>

      <p className="text-xs text-gray-500 mb-4">
        {retrieval.total_candidates} candidates &rarr; cross-encoder reranking &rarr;{" "}
        <span className="font-medium text-gray-700">top {retrieval.after_reranking} selected</span>
      </p>

      <div className="space-y-3">
        {retrieval.chunks.map((chunk) => (
          <EvidenceChunk
            key={chunk.index}
            chunk={chunk}
            isActive={activeChunkIndex === chunk.index}
            onClick={() => onChunkClick(chunk.index)}
          />
        ))}
      </div>
    </motion.div>
  );
}
