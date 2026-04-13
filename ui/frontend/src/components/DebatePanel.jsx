import { motion } from "framer-motion";
import { MessageSquare } from "lucide-react";
import DoctorMessage from "./DoctorMessage";

export default function DebatePanel({ debate, onCitationClick, chunks }) {
  if (!debate) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <div className="flex items-center gap-2 mb-4">
        <MessageSquare className="w-4 h-4 text-violet-600" />
        <h3 className="text-sm font-semibold text-gray-900">Multi-Agent Debate</h3>
      </div>

      {debate.rounds.map((round) => (
        <div key={round.round} className="mb-6">
          <div className="flex items-center gap-2 mb-3">
            <div className="h-px flex-1 bg-gray-200" />
            <span className="text-xs font-medium text-gray-400 px-2">
              {round.round === 1 ? "Round 1 — Opening Arguments" : `Round ${round.round} — Rebuttals`}
            </span>
            <div className="h-px flex-1 bg-gray-200" />
          </div>

          <div className="space-y-3">
            {round.doctor_a && (
              <DoctorMessage
                doctor="A"
                round={round.round}
                position={round.doctor_a.position}
                confidence={round.doctor_a.confidence}
                reasoning={round.doctor_a.reasoning}
                evidenceCited={round.doctor_a.evidence_cited}
                onCitationClick={onCitationClick}
                chunks={chunks}
              />
            )}
            {round.doctor_b && (
              <DoctorMessage
                doctor="B"
                round={round.round}
                position={round.doctor_b.position}
                confidence={round.doctor_b.confidence}
                reasoning={round.doctor_b.reasoning}
                evidenceCited={round.doctor_b.evidence_cited}
                onCitationClick={onCitationClick}
                chunks={chunks}
              />
            )}
          </div>
        </div>
      ))}
    </motion.div>
  );
}
