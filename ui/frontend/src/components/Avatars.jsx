/** Inline SVG avatars for Dr. Chen, Dr. Patel, and the Judge. */

export function DoctorAAvatar({ size = 48 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Background */}
      <circle cx="50" cy="50" r="48" fill="#EFF6FF" />
      {/* Head */}
      <circle cx="50" cy="38" r="16" fill="#F5D6B8" />
      {/* Hair */}
      <path d="M34 32 Q34 22 50 22 Q66 22 66 32 Q66 28 50 28 Q34 28 34 32Z" fill="#374151" />
      {/* Glasses */}
      <rect x="38" y="33" width="10" height="8" rx="3" fill="none" stroke="#374151" strokeWidth="1.5" />
      <rect x="52" y="33" width="10" height="8" rx="3" fill="none" stroke="#374151" strokeWidth="1.5" />
      <line x1="48" y1="37" x2="52" y2="37" stroke="#374151" strokeWidth="1.5" />
      {/* Eyes */}
      <circle cx="43" cy="37" r="1.5" fill="#374151" />
      <circle cx="57" cy="37" r="1.5" fill="#374151" />
      {/* Smile */}
      <path d="M44 44 Q50 48 56 44" fill="none" stroke="#374151" strokeWidth="1.5" strokeLinecap="round" />
      {/* Lab coat */}
      <path d="M28 82 Q28 60 50 56 Q72 60 72 82" fill="#FFFFFF" stroke="#E5E7EB" strokeWidth="1" />
      {/* Coat lapels */}
      <path d="M44 56 L50 68 L56 56" fill="none" stroke="#E5E7EB" strokeWidth="1" />
      {/* Stethoscope */}
      <path d="M38 62 Q32 68 34 76" fill="none" stroke="#2563EB" strokeWidth="2" strokeLinecap="round" />
      <circle cx="34" cy="78" r="3" fill="#2563EB" />
    </svg>
  );
}

export function DoctorBAvatar({ size = 48 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Background */}
      <circle cx="50" cy="50" r="48" fill="#F5F3FF" />
      {/* Head */}
      <circle cx="50" cy="38" r="16" fill="#D4A574" />
      {/* Hair */}
      <path d="M34 34 Q34 20 50 20 Q66 20 66 34 Q64 26 50 26 Q36 26 34 34Z" fill="#1F2937" />
      {/* Eyes */}
      <circle cx="43" cy="36" r="2" fill="#1F2937" />
      <circle cx="57" cy="36" r="2" fill="#1F2937" />
      {/* Confident smile */}
      <path d="M43 44 Q50 49 57 44" fill="none" stroke="#1F2937" strokeWidth="1.5" strokeLinecap="round" />
      {/* Lab coat */}
      <path d="M28 82 Q28 60 50 56 Q72 60 72 82" fill="#FFFFFF" stroke="#E5E7EB" strokeWidth="1" />
      {/* Coat lapels */}
      <path d="M44 56 L50 68 L56 56" fill="none" stroke="#E5E7EB" strokeWidth="1" />
      {/* Clipboard */}
      <rect x="60" y="62" width="12" height="16" rx="2" fill="#7C3AED" />
      <rect x="62" y="60" width="8" height="4" rx="1" fill="#7C3AED" stroke="#F5F3FF" strokeWidth="0.5" />
      <line x1="63" y1="68" x2="69" y2="68" stroke="#F5F3FF" strokeWidth="1" />
      <line x1="63" y1="72" x2="69" y2="72" stroke="#F5F3FF" strokeWidth="1" />
    </svg>
  );
}

export function JudgeAvatar({ size = 48 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Background */}
      <circle cx="50" cy="50" r="48" fill="#FFFBEB" />
      {/* Scales base */}
      <line x1="50" y1="25" x2="50" y2="55" stroke="#D97706" strokeWidth="3" strokeLinecap="round" />
      {/* Beam */}
      <line x1="30" y1="35" x2="70" y2="35" stroke="#D97706" strokeWidth="3" strokeLinecap="round" />
      {/* Left pan */}
      <path d="M24 40 Q27 50 36 50 L24 50 Z" fill="#D97706" />
      <line x1="30" y1="35" x2="24" y2="40" stroke="#D97706" strokeWidth="2" />
      <line x1="30" y1="35" x2="36" y2="40" stroke="#D97706" strokeWidth="2" />
      <path d="M24 40 L36 40 Q36 50 30 50 Q24 50 24 40Z" fill="#FCD34D" />
      {/* Right pan */}
      <line x1="70" y1="35" x2="64" y2="40" stroke="#D97706" strokeWidth="2" />
      <line x1="70" y1="35" x2="76" y2="40" stroke="#D97706" strokeWidth="2" />
      <path d="M64 40 L76 40 Q76 50 70 50 Q64 50 64 40Z" fill="#FCD34D" />
      {/* Base */}
      <rect x="42" y="55" width="16" height="4" rx="2" fill="#D97706" />
      <rect x="38" y="59" width="24" height="4" rx="2" fill="#D97706" />
      {/* Gavel */}
      <rect x="55" y="68" width="16" height="8" rx="3" fill="#92400E" />
      <rect x="61" y="76" width="4" height="10" rx="1" fill="#92400E" />
      {/* Label */}
      <text x="50" y="96" textAnchor="middle" fontSize="8" fill="#D97706" fontFamily="system-ui" fontWeight="600">GRPO</text>
    </svg>
  );
}
