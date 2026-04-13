import { useState, useRef, useCallback } from "react";

const STEP_ORDER = [
  "retrieving",
  "filtering",
  "debating",
  "computing_trust",
  "judging",
  "complete",
];

const STEP_DELAYS = [0, 1500, 3000, 5000, 7000];

export function usePipeline() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentStep, setCurrentStep] = useState(null);
  const timersRef = useRef([]);

  const clearTimers = useCallback(() => {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
  }, []);

  const runPipeline = useCallback(
    async (question) => {
      clearTimers();
      setResult(null);
      setError(null);
      setIsLoading(true);
      setCurrentStep("retrieving");

      // Simulate progressive steps while waiting for API
      STEP_DELAYS.forEach((delay, i) => {
        if (i === 0) return; // already set to "retrieving"
        const timer = setTimeout(() => {
          setCurrentStep((prev) => {
            if (prev === "complete") return prev;
            return STEP_ORDER[i];
          });
        }, delay);
        timersRef.current.push(timer);
      });

      try {
        const res = await fetch("/api/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question }),
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `Server error: ${res.status}`);
        }

        const data = await res.json();
        clearTimers();
        setResult(data);
        setCurrentStep("complete");
      } catch (err) {
        clearTimers();
        setError(err.message);
        setCurrentStep(null);
      } finally {
        setIsLoading(false);
      }
    },
    [clearTimers]
  );

  const reset = useCallback(() => {
    clearTimers();
    setResult(null);
    setError(null);
    setIsLoading(false);
    setCurrentStep(null);
  }, [clearTimers]);

  return { runPipeline, result, isLoading, error, currentStep, reset };
}
