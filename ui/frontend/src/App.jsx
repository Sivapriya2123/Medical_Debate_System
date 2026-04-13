import { useState, useCallback, useRef } from "react";
import { usePipeline } from "./hooks/usePipeline";
import QuestionInput from "./components/QuestionInput";
import PipelineProgress from "./components/PipelineProgress";
import RetrievalPanel from "./components/RetrievalPanel";
import DebatePanel from "./components/DebatePanel";
import TrustPanel from "./components/TrustPanel";
import JudgePanel from "./components/JudgePanel";
import { Plus, Clock, Zap, Stethoscope, BookOpen, Scale } from "lucide-react";

export default function App() {
  const { runPipeline, result, isLoading, error, currentStep, reset } = usePipeline();
  const [activeChunkIndex, setActiveChunkIndex] = useState(null);
  const evidenceRef = useRef(null);
  const highlightTimer = useRef(null);

  const handleCitationClick = useCallback((index) => {
    setActiveChunkIndex(index);
    const el = document.getElementById(`evidence-${index}`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
    if (highlightTimer.current) clearTimeout(highlightTimer.current);
    highlightTimer.current = setTimeout(() => setActiveChunkIndex(null), 3000);
  }, []);

  const handleNewQuestion = () => {
    reset();
    setActiveChunkIndex(null);
  };

  const showResults = result || isLoading;

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
      {/* Header */}
      <header className="border-b border-indigo-100 bg-white/80 backdrop-blur-sm sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 bg-gradient-to-br from-indigo-600 to-violet-600 rounded-lg flex items-center justify-center shadow-sm">
              <Plus className="w-4 h-4 text-white rotate-45" strokeWidth={3} />
            </div>
            <span className="text-base font-bold bg-gradient-to-r from-indigo-700 to-violet-600 bg-clip-text text-transparent">
              Medical Debate AI
            </span>
          </div>
          <div className="flex items-center gap-3">
            <span className="hidden sm:inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-gradient-to-r from-indigo-50 to-violet-50 text-indigo-600 border border-indigo-100">
              GRPO-Optimized
            </span>
            {showResults && (
              <button
                onClick={handleNewQuestion}
                className="px-3 py-1.5 text-sm bg-indigo-50 text-indigo-600 hover:bg-indigo-100 rounded-lg font-medium transition-colors"
              >
                New Question
              </button>
            )}
          </div>
        </div>
      </header>

      <main>
        {/* Welcome + Question Input */}
        {!showResults && (
          <>
            {/* Hero Section */}
            <div className="bg-gradient-to-br from-indigo-600 via-violet-600 to-purple-700 text-white">
              <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-20 text-center">
                <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/15 backdrop-blur-sm text-sm font-medium mb-6 border border-white/20">
                  <Zap className="w-3.5 h-3.5" />
                  79.0% accuracy on PubMedQA
                </div>
                <h1 className="text-4xl sm:text-5xl font-bold mb-4 leading-tight">
                  Trust-Aware Medical
                  <br />
                  <span className="text-indigo-200">Debate System</span>
                </h1>
                <p className="text-indigo-100 text-lg max-w-2xl mx-auto mb-10">
                  Ask a biomedical research question. Two AI doctors will debate it,
                  and a trust-aware judge will deliver the final verdict.
                </p>

                {/* How it works */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 max-w-2xl mx-auto mb-10">
                  <div className="flex flex-col items-center gap-2 p-4 rounded-xl bg-white/10 backdrop-blur-sm border border-white/10">
                    <div className="w-10 h-10 rounded-full bg-blue-400/20 flex items-center justify-center">
                      <BookOpen className="w-5 h-5 text-blue-200" />
                    </div>
                    <span className="text-sm font-semibold">Retrieve</span>
                    <span className="text-xs text-indigo-200">PubMed evidence</span>
                  </div>
                  <div className="flex flex-col items-center gap-2 p-4 rounded-xl bg-white/10 backdrop-blur-sm border border-white/10">
                    <div className="w-10 h-10 rounded-full bg-violet-400/20 flex items-center justify-center">
                      <Stethoscope className="w-5 h-5 text-violet-200" />
                    </div>
                    <span className="text-sm font-semibold">Debate</span>
                    <span className="text-xs text-indigo-200">Two AI doctors</span>
                  </div>
                  <div className="flex flex-col items-center gap-2 p-4 rounded-xl bg-white/10 backdrop-blur-sm border border-white/10">
                    <div className="w-10 h-10 rounded-full bg-amber-400/20 flex items-center justify-center">
                      <Scale className="w-5 h-5 text-amber-200" />
                    </div>
                    <span className="text-sm font-semibold">Judge</span>
                    <span className="text-xs text-indigo-200">GRPO verdict</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Search area */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 -mt-8">
              <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6 sm:p-8">
                <QuestionInput onSubmit={runPipeline} isLoading={isLoading} />
              </div>
            </div>

            <div className="py-4" />
          </>
        )}

        {/* Error */}
        {error && (
          <div className="max-w-2xl mx-auto mt-4 px-4">
            <div className="p-4 bg-rose-50 border border-rose-200 rounded-lg text-sm text-rose-700">
              <strong>Error:</strong> {error}
            </div>
          </div>
        )}

        {/* Results */}
        {showResults && (
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            {/* Question display */}
            <div className="text-center mb-6 bg-gradient-to-r from-indigo-50 via-violet-50 to-purple-50 rounded-xl p-5 border border-indigo-100">
              <p className="text-xs text-indigo-400 uppercase tracking-wider font-semibold mb-1">Analyzing Question</p>
              <p className="text-lg font-semibold text-gray-900">
                {result?.question || "Processing..."}
              </p>
            </div>

            <PipelineProgress currentStep={currentStep} />

            {result && (
              <div className="space-y-6">
                {/* Section label: Evidence & Debate */}
                <div className="flex items-center gap-3">
                  <div className="h-px flex-1 bg-gradient-to-r from-blue-200 to-transparent" />
                  <span className="text-xs font-semibold text-blue-600 uppercase tracking-wider">Evidence & Debate</span>
                  <div className="h-px flex-1 bg-gradient-to-l from-violet-200 to-transparent" />
                </div>

                {/* Evidence + Debate side by side */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                  <div
                    ref={evidenceRef}
                    className="lg:col-span-4 bg-gradient-to-b from-blue-50/50 to-white border border-blue-100 rounded-xl shadow-sm p-4 lg:max-h-[700px] lg:overflow-y-auto"
                  >
                    <RetrievalPanel
                      retrieval={result.retrieval}
                      activeChunkIndex={activeChunkIndex}
                      onChunkClick={setActiveChunkIndex}
                    />
                  </div>

                  <div className="lg:col-span-8 bg-gradient-to-b from-violet-50/30 to-white border border-violet-100 rounded-xl shadow-sm p-4">
                    <DebatePanel
                      debate={result.debate}
                      onCitationClick={handleCitationClick}
                      chunks={result.retrieval?.chunks}
                    />
                  </div>
                </div>

                {/* Section label: Analysis */}
                <div className="flex items-center gap-3 mt-2">
                  <div className="h-px flex-1 bg-gradient-to-r from-teal-200 to-transparent" />
                  <span className="text-xs font-semibold text-teal-600 uppercase tracking-wider">Trust Analysis & Verdict</span>
                  <div className="h-px flex-1 bg-gradient-to-l from-amber-200 to-transparent" />
                </div>

                {/* Trust + Judge in a grid */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <TrustPanel trust={result.trust} />
                  <JudgePanel judge={result.judge} onCitationClick={handleCitationClick} />
                </div>

                {/* Metadata footer */}
                {result.metadata && (
                  <div className="flex flex-wrap items-center justify-center gap-4 sm:gap-6 text-xs text-gray-400 py-4 px-4 bg-gray-50 rounded-xl border border-gray-100">
                    <span className="inline-flex items-center gap-1.5">
                      <Clock className="w-3 h-3" />
                      {result.metadata.total_time_seconds}s
                    </span>
                    <span className="w-1 h-1 rounded-full bg-gray-300" />
                    <span>{result.metadata.total_tokens?.toLocaleString() || "?"} tokens</span>
                    <span className="w-1 h-1 rounded-full bg-gray-300" />
                    <span>{result.metadata.num_calls || "?"} LLM calls</span>
                    <span className="w-1 h-1 rounded-full bg-gray-300" />
                    <span>Model: {result.metadata.model}</span>
                  </div>
                )}

              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
