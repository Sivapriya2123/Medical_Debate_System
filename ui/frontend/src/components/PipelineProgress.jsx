import { Check, Database, Filter, MessageSquare, Shield, Scale } from "lucide-react";

const STEPS = [
  { key: "retrieving", label: "Retrieve", color: "bg-blue-600", ring: "ring-blue-200", icon: Database, desc: "Searching PubMed evidence..." },
  { key: "filtering", label: "Filter", color: "bg-blue-600", ring: "ring-blue-200", icon: Filter, desc: "Filtering outdated & conflicting..." },
  { key: "debating", label: "Debate", color: "bg-violet-600", ring: "ring-violet-200", icon: MessageSquare, desc: "Doctors are debating..." },
  { key: "computing_trust", label: "Trust", color: "bg-teal-600", ring: "ring-teal-200", icon: Shield, desc: "Computing trust signals..." },
  { key: "judging", label: "Judge", color: "bg-amber-600", ring: "ring-amber-200", icon: Scale, desc: "GRPO judge deciding..." },
];

export default function PipelineProgress({ currentStep, onStepClick }) {
  const currentIdx = STEPS.findIndex((s) => s.key === currentStep);
  const isComplete = currentStep === "complete";

  return (
    <div className="w-full max-w-3xl mx-auto mb-8">
      <div className="flex items-center justify-between">
        {STEPS.map((step, i) => {
          const isDone = isComplete || i < currentIdx;
          const isActive = !isComplete && i === currentIdx;
          const Icon = step.icon;

          return (
            <div key={step.key} className="flex items-center flex-1 last:flex-none">
              <button
                onClick={() => onStepClick?.(step.key)}
                className="flex flex-col items-center gap-1.5 group"
              >
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium transition-all duration-300 ${
                    isDone
                      ? `${step.color} text-white shadow-sm`
                      : isActive
                      ? `${step.color} text-white ring-4 ${step.ring} animate-pulse`
                      : "bg-gray-100 text-gray-400"
                  }`}
                >
                  {isDone ? <Check className="w-4 h-4" /> : <Icon className="w-4 h-4" />}
                </div>
                <span
                  className={`text-xs font-semibold transition-colors ${
                    isDone || isActive ? "text-gray-900" : "text-gray-400"
                  }`}
                >
                  {step.label}
                </span>
              </button>

              {i < STEPS.length - 1 && (
                <div className="flex-1 h-1 mx-2 mt-[-18px] rounded-full overflow-hidden bg-gray-100">
                  <div
                    className={`h-full transition-all duration-700 rounded-full ${
                      isDone ? `${step.color}` : "bg-transparent"
                    }`}
                    style={{ width: isDone ? "100%" : "0%" }}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>

      {!isComplete && currentIdx >= 0 && (
        <div className="text-center mt-4">
          <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gray-50 border border-gray-200 text-sm text-gray-500">
            <span className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
            {STEPS[currentIdx].desc}
          </span>
        </div>
      )}
    </div>
  );
}
