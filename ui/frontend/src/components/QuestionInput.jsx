import { useState } from "react";
import { Search, Loader2, Sparkles } from "lucide-react";

const EXAMPLES = [
  { text: "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?", color: "from-blue-50 to-indigo-50 border-blue-100 hover:border-blue-300" },
  { text: "Is increased time to surgery associated with an increased risk of surgical site infection?", color: "from-violet-50 to-purple-50 border-violet-100 hover:border-violet-300" },
  { text: "Does physical activity reduce the risk of cognitive decline?", color: "from-teal-50 to-emerald-50 border-teal-100 hover:border-teal-300" },
  { text: "Is metformin effective for weight loss in non-diabetic patients?", color: "from-amber-50 to-orange-50 border-amber-100 hover:border-amber-300" },
];

export default function QuestionInput({ onSubmit, isLoading }) {
  const [question, setQuestion] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (question.trim() && !isLoading) onSubmit(question.trim());
  };

  const handleExample = (q) => {
    setQuestion(q);
    if (!isLoading) onSubmit(q);
  };

  return (
    <div className="max-w-3xl mx-auto">
      <form onSubmit={handleSubmit} className="relative">
        <div className="flex items-center border-2 border-gray-200 rounded-2xl shadow-sm bg-white focus-within:border-indigo-500 focus-within:shadow-md focus-within:shadow-indigo-100 transition-all">
          <Search className="ml-4 w-5 h-5 text-indigo-400 shrink-0" />
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a biomedical research question..."
            className="flex-1 px-4 py-4 text-gray-900 placeholder-gray-400 bg-transparent border-none outline-none text-base"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !question.trim()}
            className="mr-2 px-6 py-2.5 bg-gradient-to-r from-indigo-600 to-violet-600 text-white rounded-xl font-semibold text-sm hover:from-indigo-700 hover:to-violet-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2 shrink-0 shadow-sm"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Running
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                Analyze
              </>
            )}
          </button>
        </div>
      </form>

      <div className="mt-6">
        <p className="text-sm text-gray-400 mb-3 font-medium">Try an example:</p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {EXAMPLES.map((ex, i) => (
            <button
              key={i}
              onClick={() => handleExample(ex.text)}
              disabled={isLoading}
              className={`px-4 py-3 text-sm text-gray-600 bg-gradient-to-r ${ex.color} border rounded-xl disabled:opacity-50 transition-all text-left hover:shadow-sm`}
            >
              {ex.text.length > 70 ? ex.text.slice(0, 70) + "..." : ex.text}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
