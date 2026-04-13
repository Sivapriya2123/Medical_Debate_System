import { motion } from "framer-motion";
import { Shield } from "lucide-react";

function TrustBar({ label, value, weight, color }) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-sm text-gray-600 w-44 shrink-0">{label}</span>
      <div className="flex-1 h-2.5 bg-gray-100 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className={`h-full rounded-full ${color}`}
        />
      </div>
      <span className="text-sm font-semibold text-gray-700 w-12 text-right">
        {value.toFixed(2)}
      </span>
      <span className="text-xs text-gray-400 w-14 text-right">w={weight}</span>
    </div>
  );
}

export default function TrustPanel({ trust }) {
  if (!trust) return null;

  const isHigh = trust.composite_score > 0.85;
  const isMed = trust.composite_score > 0.6;

  const ringColor = isHigh
    ? "from-emerald-400 to-teal-500"
    : isMed
    ? "from-amber-400 to-orange-500"
    : "from-rose-400 to-red-500";

  const textColor = isHigh
    ? "text-emerald-600"
    : isMed
    ? "text-amber-600"
    : "text-rose-600";

  const bgColor = isHigh
    ? "from-emerald-50/50 to-teal-50/30"
    : isMed
    ? "from-amber-50/50 to-orange-50/30"
    : "from-rose-50/50 to-red-50/30";

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      className={`bg-gradient-to-br ${bgColor} border border-gray-200 rounded-xl shadow-sm p-5`}
    >
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-full bg-teal-50 flex items-center justify-center">
            <Shield className="w-4 h-4 text-teal-600" />
          </div>
          <h3 className="text-sm font-bold text-gray-900">Trust Score</h3>
        </div>

        {/* Big score circle */}
        <div className="relative">
          <div className={`w-16 h-16 rounded-full bg-gradient-to-br ${ringColor} p-[3px]`}>
            <div className="w-full h-full rounded-full bg-white flex items-center justify-center">
              <span className={`text-lg font-bold ${textColor}`}>
                {trust.composite_score.toFixed(2)}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="space-y-3">
        <TrustBar
          label="Agent Agreement"
          value={trust.agreement_score}
          weight={trust.weights.agreement}
          color="bg-gradient-to-r from-blue-400 to-blue-500"
        />
        <TrustBar
          label="Reasoning Consistency"
          value={trust.embedding_similarity}
          weight={trust.weights.similarity}
          color="bg-gradient-to-r from-violet-400 to-violet-500"
        />
        <TrustBar
          label="Confidence Stability"
          value={trust.confidence_stability}
          weight={trust.weights.stability}
          color="bg-gradient-to-r from-teal-400 to-teal-500"
        />
      </div>

      <p className="text-xs text-gray-400 mt-4 italic">
        High trust = doctors aligned = GRPO judge is more decisive, not cautious.
      </p>
    </motion.div>
  );
}
